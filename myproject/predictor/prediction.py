import pandas as pd
import numpy as np
import re
import joblib
import pickle
import spacy
from django.conf import settings
import os 


class Predictor:
    def __init__(self, file_names, resume) -> None:
       self.file_names = file_names
       self.b_pres_reverse = [
            {0: 'I', 1: 'E'},
            {0: 'N', 1: 'S'},
            {0: 'T', 1: 'F'},
            {0: 'J', 1: 'P'}
            ]
       __path_to_model_vec = os.path.join(settings.MEDIA_ROOT, "machine_learning")
       __model_path = os.path.join(__path_to_model_vec, "model/model.pkl")

       self.actual_personality = {
                    'I': "Introvert",
                    'E': "Extrovert",
                    'N': "Intution",
                    'S': "Sensing",
                    "T": "Thinking",
                    'F': "Feeling",
                    "J": "Judging",
                    "P": "Perceiving"
                }

       self.count_vec = joblib.load(os.path.join(__path_to_model_vec, "vectorizer/vectorizer.pkl"))
       self.tfizer = joblib.load(os.path.join(__path_to_model_vec, "vectorizer/tfidf_vectorizer.pkl"))
       self.model = pickle.load(open(__model_path, "rb"))
       self.my_post = [resume.replace('\n', ' ').strip()]

       __pd_structure = {"name" :pd.Series([], dtype=pd.StringDtype())}
       __pd_structure.update({value: pd.Series([])  for value in self.actual_personality.values()})
       __pd_structure.update({"personality_cat": pd.Series([], dtype=pd.StringDtype())})
       self.data_result = pd.DataFrame().from_dict(__pd_structure)

       self.__pipeline_def()


    def __pipeline_def(self):
        self.my_post = self.__pre_process()
        self.__transformation()
        self.__prob_and_pers()
        self.__result_data()


    def __translate_back(self, personality):
        s = ""
        for i, j in enumerate(personality):
            s += self.b_pres_reverse[i][j]
        return s


    def __result_data(self):
        for key in self.result:
            temp = []
            personality_ = self.__translate_back(self.result[key])
            probs_ = list(np.concatenate(self.probs[key]).flat)
            temp.append(key)
            temp += probs_
            temp.append(personality_)
            self.data_result.loc[len(self.data_result)] = temp


    def  __pre_process(self):
        nlp = spacy.load('en', disable = ['parse', 'ner'])
        post =  self.my_post[0]
            
        # remove url
        temp = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', post)

        # remove non-words
        temp = re.sub(r'[^A-Za-z]', " ", temp)

        #remove more than one space
        temp = re.sub(' +', " ", temp).lower()

        #remove multiple letter repeating words
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        #remove stop words
        doc = nlp(temp)
        temp = " ".join([tok.lemma_ for tok in doc if not tok.is_stop])

        return np.array([temp])
    

    def __transformation(self):
        self.my_X_cnt = self.count_vec.transform(self.my_post)
        self.my_X_tfidf = self.tfizer.transform(self.my_X_cnt).toarray()
    
    
    def __prob_and_pers(self):
        self.result = {}
        self.probs = {}
        for fname, x_tfidf in zip([i.split("/")[-1] 
                                    for i in self.file_names], 
                            self.my_X_tfidf):
            pred_result = []
            prob_result = []
            for i in self.model:
                y_pred = i.predict([x_tfidf])
                y_pred_prob = i.predict_proba([x_tfidf])
                    
                pred_result.append(y_pred[0])
                prob_result.append(y_pred_prob[0] * 100)    
            
            self.result[fname] = pred_result
            self.probs[fname] = prob_result

