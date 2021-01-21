import pandas as pd
from tbrs import TermBasedRandomSampling
from preprocessing2 import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold
from confusionmatrix import ConfusionMatrix
import time

start = time.time()

data = pd.read_excel(
    r'C:\Users\PPATK\Desktop\Code 2\Code\Skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,10)
data_train, data_test = kfold.get_data_sequence()
i=0
print("kfold")
print(time.time() - start)
start = time.time()

prepro = Preprocessing()
cleaned_data, terms,asd = prepro.preprocessing(data_train[i]["tweet"])
print("preprocessing")
print(time.time() - start)
start = time.time()

tbrs = TermBasedRandomSampling(X=10, Y=10, L=40)
stopwords = tbrs.create_stopwords(cleaned_data,terms)

print("remove stopword")
print(time.time() - start)
start = time.time()

prepro2 = Preprocessing()
new_cleaned_data, new_terms, removed_words = prepro2.remove_stopword(cleaned_data, stopwords)

print("create stopword")
print(time.time() - start)
start = time.time()

weight = Weighting(new_cleaned_data, new_terms)
tfidf = weight.get_tf_idf_weighting()
idf = weight.get_idf()

nb = NBMultinomial()
nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],stopwords,idf,tfidf)

print("nb fit")
print(time.time() - start)
start = time.time()

y_test = []
y_pred = []

for j in range(len(data_test[i]["tweet"])):
    prediction = nb.predict(data_test[i]["tweet"][j],data_test[i]["target"][j])
    y_test.append(data_test[i]["target"][j])
    y_pred.append(prediction)

print("nb pred")
print(time.time() - start)
start = time.time()

cm = ConfusionMatrix()
accuracy, precision, recall, fmeasure = cm.score(y_test, y_pred)

print("Stopwords")
print(stopwords)
print("\nRemoved Stopwords")
print(removed_words)
print("\nAccuracy     : {}".format(accuracy))
print("Precision    : {}".format(precision))
print("Recall       : {}".format(recall))
print("FMeasure     : {}".format(fmeasure))

# df = pd.DataFrame({'X':x_array,'Y':y_array,'L':l_array,'K-Fold':kfold_per_combination,'Accuracy':list_acc,'Precision':list_prec,'Recall':list_recall,'F-Measure':list_fmeasure,'Fold Accuracy':fold_accuracy,'Fold Precision':fold_precision,'Fold Recall':fold_recall,'Fold F-Measure':fold_fmeasure})
# print(df)
# df.to_excel(r'cobabarunih.xlsx', index = False, header=True)
