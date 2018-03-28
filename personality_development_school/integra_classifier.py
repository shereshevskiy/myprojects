__author__ = 'dmitry_sh'

import pickle as pkl
import os
import numpy as np
import pymorphy2 # с русским языком, в отличие от nltk.stem.WordNetLemmatizer

#abs_path = 'C:/Users/dsher/OneDrive/Documents/!_SkillsEvolution/ML/\
#mipt_ya_ml_spec/practice/6_Final_project/sentiment_analysis/week7_final_project_APP'
abs_path = os.path.dirname(os.path.realpath(__file__))
#abs_path = ''
# определяем и используем абсолютный путь, потому что в консоле на винде
# относительный путь не "прокатывает", как в ноутбуке или в отладчике на Spyder

path_to_data = ('data' )


class IntegraClassifier(object):
    def __init__(self):
        with open(os.path.join(abs_path, path_to_data, 'int_clf_threeclass_classification.pkl'), 'rb') as clf_pkl, \
            open(os.path.join(abs_path, path_to_data, 'int_vectorizer.pkl'), 'rb') as vectorizer_pkl, \
            open(os.path.join(abs_path, path_to_data, 'LR_multi_bal_clf.pkl'), 'rb') as multi_clf, \
            open(os.path.join(abs_path, path_to_data, 'vectorizer_for_product.pkl'), 'rb') as vect_for_prod_pkl, \
            open(os.path.join(abs_path, path_to_data, 'le.pkl'), 'rb') as le_pkl\
            :
            self.model = pkl.load(clf_pkl)
            self.vectorizer = pkl.load(vectorizer_pkl)
            self.vectorizer_for_product = pkl.load(vect_for_prod_pkl)
            self.multi_clf = pkl.load(multi_clf)
            self.le = pkl.load(le_pkl)
            
        self.classes_dict = {0: " не", 
                             1: "", 
                             2: "",
                            -1: "Извините, возникла ошибка. Повторите, пожалуйста."}
        self.value_dict = {0: "", 
                           1: " на сумму до 30 тыс. Р", 
                           2: " на сумму свыше 30 тыс. Р",
                          -1: ""}


    @staticmethod
    def get_probability_words(probability):
        if probability < 0.363:
            return " возможно"
        if probability < 0.462:
            return " вероятно"
        if probability > 0.627:
            return " очень вероятно"
        else:
            return ""
       
    def text_preprocessing(self, text):
        # предобработка текста
        def del_any_symbols(text):
            return (text.replace('\n', ' ').replace('\r', ' ').replace('<br />', ' ').replace('\t', ' ')
                    .replace('&quot;', ' ').replace('!', ' ! ').replace('?', ' ? ').replace('\xa0', ' '))
        # удаление пунктуации кроме '!' и '?', они могут нести информацию
        def del_punctuation(text):
            return ''.join(' ' if symbol in '"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~' else symbol for symbol in text)
        # лемматизация
        def lemmatizer(text):
            pymorph = pymorphy2.MorphAnalyzer()
            return' '.join([pymorph.parse(word)[0].normal_form for word in text.split(' ') if word not in ['']])
        # общий препроцессинг текста

        return lemmatizer(del_punctuation(del_any_symbols(text)))
    
    
    
    def predict_text(self, text):

        try:
            vectorized = self.vectorizer.transform([self.text_preprocessing(text)])
#             vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0],\
                   self.model.predict_proba(vectorized)[0].max()
        except:
            print ("prediction error")
            return -1, 0.35

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print ('prediction error')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return 'клиент{}{} планирует покупку{}'.format(self.get_probability_words(prediction_probability), 
                                                self.classes_dict[class_prediction],
                                                self.value_dict[class_prediction]) if class_prediction != -1 else \
                                                self.classes_dict[class_prediction]
        
    def predict_curses(self, text, n=3):
        text = self.text_preprocessing(text)
        vectorized = self.vectorizer_for_product.transform([text])
        three_curs_numbers = np.argsort(-self.multi_clf.predict_proba(vectorized))[0][:n]
        three_curs_names = self.le.inverse_transform(three_curs_numbers)
        return list(three_curs_names)