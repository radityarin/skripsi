import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfoldnew import KFold
import time
from sklearn.metrics import precision_recall_fscore_support as score
from confusionmatrix import ConfusionMatrix

data = pd.read_excel(
    r'Skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,10)
data_train, data_test = kfold.get_data_sequence()

start = time.time()

accuracy_total_accumulation = 0
accuracy_per_fold = []
accuracy_each_class_per_fold = []
precision_per_fold = []
recall_per_fold = []
fmeasure_per_fold = []

fold = list(range (1,11))

for i in range(len(data_train)):
    print("Fold ke " + str(i+1))
    # print(len(data_train[i]["tweet"]))
    # print(len(data_test[i]["tweet"]))
    y_test = []
    y_pred = []
    # TAHAP PEMBUATAN STOPWORD
    prepro = Preprocessing()
    new_cleaned_data, new_terms = prepro.preprocessing(data_train[i]["tweet"])
    
    # TAHAP PELATIHAN
    
    weight = Weighting(new_cleaned_data, new_terms)
    tfidf = weight.get_tf_idf_weighting()
    idf = weight.get_idf()

    nb = NBMultinomial()
    nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],[],idf,tfidf)
    
    correct_ans = 0
    for j in range(len(data_test[i]["tweet"])):
        print("Test ke " + str(j))
        prediction = nb.predict(data_test[i]["tweet"][j],data_test[i]["target"][j])
        y_test.append(data_test[i]["target"][j])
        y_pred.append(prediction)
        if prediction == data_test[i]["target"][j]:
            correct_ans+=1
        # break

    accuracy = (float(correct_ans) / len(data_test[i]["tweet"])) * 100
    accuracy_per_fold.append(accuracy)
    accuracy_total_accumulation+=accuracy

    cm = ConfusionMatrix()
    accuracy, accuracy_each_class, precision_each_class, recall_each_class, fmeasure_each_class = cm.score(y_test, y_pred)


    # precision, recall, fscore, support = score(y_test, y_pred, labels=["Negatif", "Netral", "Positif"])
    accuracy_each_class_per_fold.append(accuracy_each_class)
    precision_per_fold.append(precision_each_class)
    recall_per_fold.append(recall_each_class)
    fmeasure_per_fold.append(fmeasure_each_class)
    
    print("Accuracy             : {}".format(accuracy))
    print("Accuracy Each Class  : {}".format(accuracy_each_class))
    print("Precision Each Class : {}".format(precision_each_class))
    print("Recall Each Class    : {}".format(recall_each_class))
    print("FMeasure Each Class  : {}".format(fmeasure_each_class))


df = pd.DataFrame({'Fold':fold,'Accuracy per Fold':accuracy_per_fold,'Accuracy Class per Fold':accuracy_each_class_per_fold,'Precision per Fold':precision_per_fold,'Recall per Fold':recall_per_fold,'F-Measure per Fold':fmeasure_per_fold,'Accuracy':accuracy_total_accumulation})
print(df)

df.to_excel(r'outputneweditedkfoldwithoutstopwords2.xlsx', index = False, header=True)
