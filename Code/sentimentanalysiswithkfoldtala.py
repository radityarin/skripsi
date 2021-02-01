import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold
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

acc_neg = []
acc_net = []
acc_pos = []
prec_neg = []
prec_net = []
prec_pos = []
recall_neg = []
recall_net = []
recall_pos = []
fmeasure_neg = []
fmeasure_net = []
fmeasure_pos = []
acc_per_fold = []

f = open("stopword_tala.txt", "r")
stopwords = f.read().split()

fold = list(range (1,11))

for i in range(len(data_train)):
    print("Fold ke " + str(i+1))
    y_test = []
    y_pred = []
    # TAHAP PEMBUATAN STOPWORD
    prepro = Preprocessing()
    new_cleaned_data, new_terms = prepro.preprocessing(data_train[i]["tweet"],stopwords=stopwords)
    
    # TAHAP PELATIHAN
    
    weight = Weighting(new_cleaned_data, new_terms)
    tfidf = weight.get_tf_idf_weighting()
    idf = weight.get_idf()

    nb = NBMultinomial()
    nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],stopwords,idf,tfidf)
    
    for j in range(len(data_test[i]["tweet"])):
        print("Test ke " + str(j))
        prediction = nb.predict(data_test[i]["tweet"][j],data_test[i]["target"][j])
        y_test.append(data_test[i]["target"][j])
        y_pred.append(prediction)

    cm = ConfusionMatrix()
    accuracy, accuracy_each_class, precision_each_class, recall_each_class, fmeasure_each_class = cm.score(y_test, y_pred)
    
    acc_neg.append(accuracy_each_class[0])
    acc_net.append(accuracy_each_class[1])
    acc_pos.append(accuracy_each_class[2])
    prec_neg.append(precision_each_class[0])
    prec_net.append(precision_each_class[1])
    prec_pos.append(precision_each_class[2])
    recall_neg.append(recall_each_class[0])
    recall_net.append(recall_each_class[1])
    recall_pos.append(recall_each_class[2])
    fmeasure_neg.append(fmeasure_each_class[0])
    fmeasure_net.append(fmeasure_each_class[1])
    fmeasure_pos.append(fmeasure_each_class[2])
    acc_per_fold.append(accuracy)


df = pd.DataFrame({'K-Fold':fold,'AccNeg':acc_neg,'AccNet':acc_net,'AccPos':acc_pos,'PreNeg':prec_neg,'PreNet':prec_net,'PrePos':prec_pos,'RecNeg':recall_neg,'RecNet':recall_net,'RecPos':recall_pos,'FMeNeg':fmeasure_neg,'FMeNet':fmeasure_net,'FmePos':fmeasure_pos,'Accuracy':acc_per_fold})
print(df)

df.to_excel(r'outputnewtabletala.xlsx', index = False, header=True)
