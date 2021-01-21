import pandas as pd
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayess import NBMultinomial
from weighting import Weighting
from confusionmatrix import ConfusionMatrix

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
    r'Skripsi.xlsx',"Fold 3 Train")
data_tweet = data['Tweet']
data_target = data['Label']

# TAHAP PEMBUATAN STOPWORD
prepro = Preprocessing()
cleaned_data, terms = prepro.preprocessing(data_tweet)
print("FIRST PREPRO DONE")
tbrs = TermBasedRandomSampling(X=40, Y=10, L=30)
stopwords = tbrs.create_stopwords(cleaned_data,terms)

# TAHAP PELATIHAN
prepro2 = Preprocessing()
new_cleaned_data, new_terms = prepro2.remove_stopword(cleaned_data, stopwords)
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

tes = pd.read_excel(
    r'Skripsi.xlsx',"Fold 3 Test")
tes_tweet = tes['Tweet']
tes_target = tes['Label']

y_test = tes_target
y_pred = []
for i in range(len(tes_tweet)):
    pred, negatif, netral, positif, pangka, pterm,lneg,lnet,lpos,termnya = nb.predict(tes_tweet[i],tes_target[i])
    y_pred.append(pred)
    if pred != tes_target[i]:
        print()
        print("Actual : " + tes_target[i])
        print("Pred   : " + pred)
        print("Term   : " + str(termnya))
        print("Neg    : " + str(lneg))
        print("Neg    : " + str(negatif))
        print("Net    : " + str(lnet))
        print("Net    : " + str(netral))
        print("Pos    : " + str(lpos))
        print("Pos    : " + str(positif))
        print("Pgrh A : " + str(pangka))
        print("Pgrh T : " + str(pterm))
        print(tes_tweet[i])
    

cm = ConfusionMatrix()
accuracy, accuracy_each_class, precision_each_class, recall_each_class, fmeasure_each_class = cm.score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))
print('accuracy: {}'.format(accuracy_each_class))
print('precision: {}'.format(precision_each_class))
print('recall: {}'.format(recall_each_class))
print('fscore: {}'.format(fmeasure_each_class))
