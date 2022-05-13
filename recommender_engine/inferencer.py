from traceback import print_exc
class Inferencer:
    def __init__(self):
        self.model = self.load_model()
        self.book_titles = self.load_book_titles()
    def load_model(self):
        '''
        Retrives latest model version from model repository such as ML Flow
        :return:
        '''
        pass
    def load_book_titles(self):
        '''
        Will load existing book titles index from the CSV file
        :return:
        '''
        pass
    @staticmethod
    def score_book(title,model,book_titles):
        '''
        Will return 10 most similar items based on the KNN model
        :param title:
        :param model
        :param book_titles
        :return: array with recommended book titles (which is passed as API output)
        '''
        try:
            book_titles_arr = []
            book_index = book_titles[book_titles['Book-Title'] == title].index.tolist()
            suggested_indexes = model.get_neighbors(book_index,10)
            suggested_books_df = book_titles[book_titles['book_number'].isin(suggested_indexes)]
            for index, row in suggested_books_df.iterrows():
                book_titles_arr.append(row['Book-Title'])
            return book_titles_arr
        except Exception as e:
            print_exc()
            message = f"Failed to recommend book becuase of error {e}"
            raise ValueError(message)

