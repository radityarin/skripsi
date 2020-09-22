import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting

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
    


















data = pd.read_excel(
    r'skripsi.xlsx',"Data Manualisasi")
data_tweet = data['Tweet']
data_target = data['Klasifikasi']

prepro = Preprocessing()
# cleaned_data, terms = prepro.preprocessing(data_tweet)

# weight = Weighting(cleaned_data, terms)
# tfidf = weight.get_tf_idf_weighting()

# tbrs = TermBasedRandomSampling(L=20,Y=50)
# stopwords = tbrs.create_stopwords(cleaned_data,terms)
# stopwords_tbrs_manualisasi = ['kuliah','online','daring','yang','saja','tetap','mau','offline','aku','materi','ada','ini','kelas','cara','sama','jadi','kasih','tapi','lebih','kalau','dosen','beberapa','kenapa','new','normal','zona','lagi','sendiri','dengan','masih','jujur','corona','paham','sekali','nilai','bisa','ya','selesai','jelas','mohon','kaji']			

text_file = open('stopword_tala.txt', 'r')
stopword_tala = text_file.read().split('\n')
text_file.close()

# new_cleaned_data, new_terms = prepro.remove_stopword(cleaned_data,stopwords)
new_cleaned_data, new_terms = prepro.preprocessing(data_tweet,stopword_tala)

nb = NBMultinomial()

nb.fit(new_cleaned_data,new_terms,data_target,stopword_tala)
nb.predict("Apa saya saja yang merasa kalau selama kuliah daring nyaman banget sampai saya tidak ingin masuk kuliah karena takut panik","Positif")
# nb.predict(data_tweet[4],"Netral")
# nb.predict(data_tweet[7],"Positif")
# nb.predict("Saya bisa paham tapi kadang suka lupa","Netral")
