import pandas as pd
from sklearn.model_selection import train_test_split
from tbrsmanualisasi import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayesmanualisasi import NBMultinomial
from weighting import Weighting
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

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
print("FIRST PREPRO DONE")
tbrs = TermBasedRandomSampling(L=20)
stopwords = tbrs.create_stopwords(cleaned_data,terms)
# TAHAP PELATIHAN
prepro2 = Preprocessing()
new_cleaned_data, new_terms = prepro.remove_stopword(cleaned_data, stopwords)
# print("new_terms")
# print(new_terms)
weight = Weighting(new_cleaned_data, new_terms)
tfidf = weight.get_tf_idf_weighting()
idf = weight.get_idf()

nb = NBMultinomial()
nb.fit(new_cleaned_data,new_terms,data_target,stopwords,idf,tfidf)
# print("stopwords")
print(stopwords)
# TAHAP PENGUJIAN

data_test = ["Apa saya saja yang merasa kalau selama kuliah daring nyaman banget sampai saya tidak ingin masuk kuliah karena takut panik",
"Aku merasa lebih leluasa dengan kuliah daring, tidak capek harus siap-siap berangkat. Hanya tinggal makan, beres didepan komputer sudah siap nyimak. Buat materi, selama online emang tidak pernah mengandalkan dosen atau temen. Jadi lebih banyak waktu buat searching sama buka textbook",
"Jujur tidak ada senang-senangnya kuliah daring. Aku butuh praktik lapangan. Apalagi semester depan magang. Apa magang online juga? Bisa stres gara-gara banyak deadline",
"Tatap langsung aja kadang tidak paham, apalagi kuliah daring, belum lagi jaringan lambat­ ditambah beberapa dosen yang jarang memberi kuliah online, atau cuma memberi tugas saja... Fix kampus ku belum siap menerapkan kuliah daring!  pic.twitter.com/UHdReyLgh8",
"Orang lain pada ribut sama keadaan kosan yang sudah ditinggal berbulan-bulan terus ribut gimana caranya balik ke kosan. Aku anteng-anteng saja jadi penghuni kos dari awal pemerintah nyuruh dirumah saja dan kuliah jadi daring"]
y_true = ["Positif","Positif","Negatif","Negatif","Netral"]
y_pred = []
for i in range(len(data_test)):
    y_pred.append(nb.predict(data_test[i],y_true[i]))


target_names = ['Negatif', 'Netral', 'Positif']
cm = confusion_matrix(y_true, y_pred)

print('accuracysss: {}'.format(cm.diagonal()/cm.sum(axis=1)))



precision, recall, fscore, support = score(y_true, y_pred, labels=["Negatif", "Netral", "Positif"])
print('accuracy: {}'.format(accuracy_score(y_true, y_pred)))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
