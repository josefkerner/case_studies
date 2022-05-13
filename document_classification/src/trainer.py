from traceback import print_exc

try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, multilabel_confusion_matrix
    import seaborn as sns
    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    import pickle
except ImportError as e:
    # print_exc()
    pass


class Trainer:
    def __init__(self):
        self.sparkSession = None

    def load_data(self):

        df = self.sparkSession.read.option("header", True).csv('dbfs:/FileStore/data/')
        return df.toPandas()

    def analyze_data(self,df):
        spark_df = self.sparkSession.createDataFrame(df)
        spark_df.registerTempTable("docs")
        sql = f"""
            with docs_tf as (
            select textContent, category,
            if(locate('http',textContent) = 0,0,1) has_link,
            size(array_distinct(split(textContent,' '))) words_content_count,
            size(array_distinct(split(title,' '))) words_title_count
            from docs
            )
            select avg(words_content_count),
            avg(words_title_count),
            sum(has_link) links_sum,
            category from docs_tf group by category
        """
        df = self.sparkSession.sql(sql)
        df.show()

    def load_dataframe(self):
        sql = "SELECT * FROM docs"
        df = self.sparkSession.sql(sql).toPandas()
        return df

    @staticmethod
    def remove_stop_words(content):
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        ps = PorterStemmer()
        output_words = []
        unique_words = []
        content = str(content).lower().split(' ')
        stop_words = ['the', 'is', 'a', 'of', 'and', '–', '&', ',', '_', '.', ':', '"', '', 'an', '=']
        for word in content:
            word = str(word).replace('.', '').replace('!', '').replace('"', '').replace('(', '').replace(')',
                                                                                                         '').replace(
                ':', '')
            word = ps.stem(word)
            if word not in stop_words:
                output_words.append(word)
                if word not in unique_words:
                    unique_words.append(word)
        return ' '.join(output_words)

    def processData(self, df):
        '''
        Processes data
        :return:
        '''
        df['textContent'] = df.textContent.apply(Trainer.remove_stop_words)
        df['title'] = df.title.apply(Trainer.remove_stop_words)
        return df

    def get_tf_idf_truncated(self, df, col):
        '''
        Obtains a tf idf vector compressed to 100 dimensions
        :param df:
        :param col:
        :return:
        '''
        vectorizer = TfidfVectorizer()
        tf_idf_matrix = vectorizer.fit_transform(df[col])
        svd = TruncatedSVD(n_components=100)
        res = svd.fit_transform(tf_idf_matrix)
        return res

    def perform_pca(self, df):
        '''
        Performs a PCA visualization in 2D space
        :param df:
        :return:
        '''
        y = df['category']

        x = self.get_tf_idf_truncated(df, 'textContent')
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, df[['category']]], axis=1)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = ['ok', 'jobs', 'shop', 'download', 'forum']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['category'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=50)
        ax.legend(targets)
        ax.grid()

    def start_train_workflow(self):
        df = self.load_data()
        #self.analyze_data(df)
        df = self.processData(df)
        #self.analyze_data(df)
        #self.perform_pca(df)
        self.train_pipeline(train_df=df)

    def train_pipeline(self, train_df):
        train_df['content_joined'] = train_df.textContent.str.cat(train_df.title, sep=" ")
        cleanup_nums = {"category": {"ok": 1, "jobs": 2, "shop": 3, "download": 4, "forum": 5}}
        train_df.replace(cleanup_nums, inplace=True)
        # import scikitplot as skplt
        y = train_df['category']
        X = train_df.loc[:, ~train_df.columns.isin(['category'])]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_train.shape)
        pipeline_log = Pipeline([
            ('count', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
        ])
        # Train model using the created sklearn pipeline
        model_name = 'logistic regression classifier'
        model_lgr = pipeline_log.fit(X_train['content_joined'], y_train)
        # Evaluate model performance
        self.evaluate_results(model_lgr, X_test, y_test)
        self.save_model_pipeline(model_lgr)

    def save_model_pipeline(self, model):
        pickle.dump(model, open('/dbfs/FileStore/sv_model.pkl', 'wb'))

    def evaluate_results(self, model, X_test, y_test):
        # Predict class labels using the learner function
        X_test['pred'] = model.predict(X_test['content_joined'])
        y_true = y_test
        y_pred = X_test['pred']
        print(y_pred)
        target_names = ['ok', 'jobs', 'shop', 'download', 'forum']

        # Print the Confusion Matrix
        results_log = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        results_df_log = pd.DataFrame(results_log).transpose()
        print(results_df_log)
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(pd.DataFrame(matrix),
                    annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
        plt.xlabel('Predictions')
        plt.xlabel('Actual')

        model_score = accuracy_score(y_pred, y_true)
        print(model_score)


#trainer = Trainer()
#trainer.sparkSession = spark
#trainer.start_train_workflow()