import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, KFold, train_test_split
from surprise import accuracy
from surprise import KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore

class Recommender:
  def __init__(self):
    pass
  def start_workflow(self):
    self.load_data()
    self.descriptive_analysis()
    data, books_index = self.preprocess_data()
    model = self.fit_model(data)
    self.save_model(model,books_index)



  def descriptive_analysis(self):
    '''
    Visualizes data
    :return:
    '''
    #1. perform long tail analysis - how severe the cold start problem is for low rated items
    #2. plot books into 2D space by using their books-users interaction vectors
    # ..to see how rich or poor the interactions are
    pass

  def load_data(self):
    '''
    Loads data into memory
    :return:
    '''
    self.books = pd.read_csv("/dbfs/FileStore/BX_Books.csv", sep=';', encoding="latin-1", error_bad_lines=False)
    self.ratings = pd.read_csv("/dbfs/FileStore/BX_Book_Ratings.csv", sep=';', encoding="latin-1", error_bad_lines=False)

  def preprocess_data(self):
    '''
    Preprocesses data
    :return:
    '''

    # consider only users who gave more than 200 ratings
    x = self.ratings['User-ID'].value_counts() > 200
    y = x[x].index  # user_ids
    print(y.shape)
    ratings = self.ratings[self.ratings['User-ID'].isin(y)]

    # and items with at least 20 ratings
    x = ratings['ISBN'].value_counts() > 20
    y = x[x].index  # item ids
    print(y.shape)
    ratings = ratings[ratings['ISBN'].isin(y)]

    ratings.drop_duplicates(['User-ID', 'ISBN'], inplace=True)
    self.books.drop_duplicates(['ISBN'], inplace=True)

    grouped = ratings.groupby(by=["ISBN"]).sum()
    grouped.reset_index(level=0, inplace=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print('grouped', grouped.shape)

    joined = grouped.join(self.books.set_index('ISBN'), on='ISBN')
    joined.drop_duplicates(['ISBN'], inplace=True)
    joined['book_number'] = joined.index

    reader = Reader(rating_scale=(0, 10))

    ratings['User-ID'] = pd.to_numeric(ratings['User-ID'], downcast='integer')
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    return data,joined

  def fit_model(self,data):
    '''
    Fits the rec engine model
    :param book_sparse:
    :param book_pivot:
    :return:
    '''

    sim_options = {'name': 'pearson_baseline',
                   'min_support': 5,
                   'user_based': False}
    train, test = train_test_split(data, test_size=.2)

    model = KNNBaseline(k=21, sim_options=sim_options)
    model.fit(train)
    base1_preds = model.test(test)
    print(accuracy.rmse(base1_preds))
    model.compute_similarities()

    return model

  def save_model(self,model,books_list):
    '''
    Will register the model and save the books index
    books index is later used in the inference to map book title to numeric ID
    :param model:
    :param books_list:
    :return:
    '''
    pass

Recommender().start_workflow()
