import pandas as pd
from tbrs import TermBasedRandomSampling
from preprocessing2 import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold
from confusionmatrix import ConfusionMatrix
import time
import pickle

start = time.time()

data_train_filename = join(dirname(__file__), "data_train")
data_test_filename = join(dirname(__file__), "data_test")
stopwords_filename = join(dirname(__file__), "sw_tala")
ncd_filename = join(dirname(__file__), "new_cleaned_data_tala")
nt_filename = join(dirname(__file__), "new_terms_tala")
rw_filename = join(dirname(__file__), "removed_words_tala")

i=0

result_data_file = open(data_train_filename, 'rb')
data_train = pickle.load(result_data_file)
result_data_file.close()

result_data_file = open(data_test_filename, 'rb')
data_test = pickle.load(result_data_file)
result_data_file.close()

result_data_file = open(ncd_filename, 'rb')
new_cleaned_data = pickle.load(result_data_file)
result_data_file.close()

result_data_file = open(nt_filename, 'rb')
new_terms = pickle.load(result_data_file)
result_data_file.close()

result_data_file = open(rw_filename, 'rb')
removed_words = pickle.load(result_data_file)
result_data_file.close()

result_data_file = open(stopwords_filename, 'rb')
stopwords = pickle.load(result_data_file)
result_data_file.close()

weight = Weighting(new_cleaned_data, new_terms)
tfidf = weight.get_tf_idf_weighting()
idf = weight.get_idf()

nb = NBMultinomial()
nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],stopwords,idf,tfidf)

def predict(input_tweet):
    prediction, negatif, netral, positif = nb.predict(input_tweet)
    result_dict = {}
    result_dict["success"] = True
    result_dict["data"] = {"prediction":prediction,"negatif":negatif,"netral":netral,"positif":positif,"stopwords":stopwords,"removed_words":removed_words}
    return result_dict

pred = predict("Rasanya mau berhenti kuliah saja kalau daring begini, seperti bayar cuma cuma, materi dikasih secara online, disuruh baca sendiri tanpa ada yang menjelaskan, berasa otodidak")
print(pred)