import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes2 import NBMultinomial
from weighting import Weighting
import pprint
pp = pprint.PrettyPrinter(indent=4)
# kfold = 5

# for i in range(kfold):
#     data = pd.read_excel(
#         r'skripsi.xlsx',"Data Manualisasi")
#     data_tweet = data['Tweet']
#     data_target = data['Klasifikasi']

#     prepro = Preprocessing()
#     cleaned_data, terms = prepro.preprocessing(data_tweet)

#     tbrs = TermBasedRandomSampling(L=10,Y=100)
#     stopwords = tbrs.create_stopwords(cleaned_data,terms)
#     new_cleaned_data, new_terms = prepro.remove_stopword(cleaned_data,stopwords)

#     nb = NBMultinomial()

#     nb.fit(new_cleaned_data,new_terms,data_target)
#     nb.predict("Rasanya mau berhenti kuliah saja kalau daring begini, seperti bayar cuma cuma, materi dikasih secara online, disuruh baca sendiri tanpa ada yang menjelaskan, berasa otodidak :","Negatif")

########################################################
# FIX SKRIPSI TANPA KFOLD
data = pd.read_excel(
    r'Skripsi.xlsx',"Data Manualisasi")
data_tweet = data['Tweet']
data_target = data['Klasifikasi']
# TAHAP PEMBUATAN STOPWORD
prepro = Preprocessing()
cleaned_data, terms = prepro.preprocessing(data_tweet)
# print(cleaned_data)
# print(terms)
tbrs = TermBasedRandomSampling(L=20)
stopwords = tbrs.create_stopwords(cleaned_data,terms)
# print("STOPWORDS")
# print("STOPWORDS " + str(stopwords))
# TAHAP PELATIHAN
# print()
prepro2 = Preprocessing()
new_cleaned_data, new_terms = prepro2.preprocessing(data_tweet, stopwords)
# print(new_cleaned_data)
# print(new_terms)
weight = Weighting(new_cleaned_data, new_terms)
tfidf = weight.get_tf_idf_weighting()
idf = weight.get_idf()
# pp.pprint(tfidf)
nb = NBMultinomial()
nb.fit(new_cleaned_data,new_terms,data_target,stopwords,idf,tfidf)
# # TAHAP PENGUJIAN
nb.predict("Apa saya saja yang merasa kalau selama kuliah daring nyaman banget sampai saya tidak ingin masuk kuliah karena takut panik","Positif")
