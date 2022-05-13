import pickle
from traceback import print_exc
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
app = FastAPI()

from trainer import Trainer

class DocScoringRequest(BaseModel):
    title: str
    content: str

@app.get('/')
def index():
    return {'message': 'this is a Semantic Vision API'}

@app.post("/score")
def score_data(request: DocScoringRequest):
    '''
    Will score data based on the input in request
    :return:
    '''
    try:
        content = Trainer.remove_stop_words(request.content)
        title = Trainer.remove_stop_words(request.title)
        model = load_model()
        predict_cat = model.predict([f"{title} {content}"])
        categories = {1:"ok", 2: "jobs", 3: "shop", 4: "download", 5: "forum"}
        if predict_cat[0] != 1:
            res = "unsuitable category"
        else:
            res = "ok"
        response = {"category": res}
        return response
    except Exception as e:
        print_exc()
        return {"category": "error"}

def load_model():
    '''
    Loads a model
    :return:
    '''
    file = open("sv_model.pkl","rb")
    return pickle.load(file)

