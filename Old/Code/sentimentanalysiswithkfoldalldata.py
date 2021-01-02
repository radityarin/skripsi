import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfoldnew import KFold
import time
from sklearn.metrics import precision_recall_fscore_support as score

data = pd.read_excel(
    r'Skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,10)
data_train, data_test = kfold.get_data_sequence()

x_array = []
y_array = []
l_array = []
accuracy_per_fold_per_xyl_combination = []
accuracy_total_per_xyl_combination = []
precision_per_fold_per_xyl_combination = []
recall_per_fold_per_xyl_combination = []
fmeasure_per_fold_per_xyl_combination = []

start = time.time()

count=1

prepro = Preprocessing()
cleaned_data, terms = prepro.preprocessing(data_tweet)
                
for l in range(10,60,10):
    for y in range (10,60,10):
        for x in range (10,60,10):
            print("PERULANGAN " + str(count))
            count+=1
            print('X={}, Y={}, L={}'.format(x,y,l))
            x_array.append(x)
            y_array.append(y)
            l_array.append(l)

            accuracy_total_accumulation = 0
            accuracy_per_fold = []
            precision_per_fold = []
            recall_per_fold = []
            fmeasure_per_fold = []

            for i in range(len(data_train)):
                print("Fold ke " + str(i+1))
                print(len(data_train[i]["tweet"]))
                print(len(data_test[i]["tweet"]))
                y_test = []
                y_pred = []
                # TAHAP PEMBUATAN STOPWORD
                # prepro = Preprocessing()
                # cleaned_data, terms = prepro.preprocessing(data_tweet)
                
                tbrs = TermBasedRandomSampling(X=x, Y=y, L=l)
                stopwords = tbrs.create_stopwords(cleaned_data,terms)

                # TAHAP PELATIHAN
                prepro2 = Preprocessing()
                new_cleaned_data, new_terms = prepro2.preprocessing(data_train[i]["tweet"], stopwords)

                weight = Weighting(new_cleaned_data, new_terms)
                tfidf = weight.get_tf_idf_weighting()
                idf = weight.get_idf()

                nb = NBMultinomial()
                nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],stopwords,idf,tfidf)
                
                correct_ans = 0
                for j in range(len(data_test[i]["tweet"])):
                    # print("Test ke " + str(j))
                    prediction = nb.predict(data_test[i]["tweet"][j],data_test[i]["target"][j])
                    y_test.append(data_test[i]["target"][j])
                    y_pred.append(prediction)
                    if prediction == data_test[i]["target"][j]:
                        correct_ans+=1
                    # break

                accuracy = (float(correct_ans) / len(data_test[i]["tweet"])) * 100
                accuracy_per_fold.append(accuracy)
                accuracy_total_accumulation+=accuracy

                precision, recall, fscore, support = score(y_test, y_pred, labels=["Negatif", "Netral", "Positif"])
                precision_per_fold.append(precision)
                recall_per_fold.append(recall)
                fmeasure_per_fold.append(fscore)
                # break

            accuracy_per_fold_per_xyl_combination.append(accuracy_per_fold)
            precision_per_fold_per_xyl_combination.append(precision_per_fold)
            recall_per_fold_per_xyl_combination.append(recall_per_fold)
            fmeasure_per_fold_per_xyl_combination.append(fmeasure_per_fold)
            accuracy_total = float(accuracy_total_accumulation/len(accuracy_per_fold))
            accuracy_total_per_xyl_combination.append(accuracy_total)
            end = time.time()

            print("Accuracy Total : "+str(accuracy_total))
            print(str(count) + " combination time : " + str(end - start))
    #         break
    #     break
    # break

df = pd.DataFrame({'X':x_array,'Y':y_array,'L':l_array,'Accuracy per Fold':accuracy_per_fold_per_xyl_combination,'Precision per Fold':precision_per_fold_per_xyl_combination,'Recall per Fold':recall_per_fold_per_xyl_combination,'F-Measure per Fold':fmeasure_per_fold_per_xyl_combination,'Accuracy':accuracy_total_per_xyl_combination})
print(df)

df.to_excel(r'outputneweditedkfoldusingalldata.xlsx', index = False, header=True)
