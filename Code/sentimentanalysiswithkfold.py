import pandas as pd
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold
from confusionmatrix import ConfusionMatrix
import time

data = pd.read_excel(
    r'C:\Users\PPATK\Documents\skripsi\Code\skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,10)
data_train, data_test = kfold.get_data_sequence()

x_array = []
y_array = []
l_array = []
kfold_per_combination = []
list_acc = []
list_prec = []
list_recall = []
list_fmeasure = []
fold_accuracy = []
fold_precision = []
fold_recall = []
fold_fmeasure = []

count=1
for l in range(10,60,10):
    for y in range (10,60,10):
        for x in range (10,60,10):
            print("PERULANGAN " + str(count))
            count+=1
            print('X={}, Y={}, L={}'.format(x,y,l))
            x_array.append(x)
            y_array.append(y)
            l_array.append(l)
            for i in range(9):
                x_array.append(" ")
                y_array.append(" ")
                l_array.append(" ")                

            accuracy_total_accumulation = 0
            precision_total_accumulation = 0
            recall_total_accumulation = 0
            fmeasure_total_accumulation = 0

            start_time_combination = time.time()

            for i in range(len(data_train)):
                start_time = time.time()
                print("Fold ke " + str(i))
                kfold_per_combination.append(i+1)
                y_test = []
                y_pred = []

                prepro = Preprocessing()
                cleaned_data, terms = prepro.preprocessing(data_train[i]["tweet"])
                
                tbrs = TermBasedRandomSampling(X=x, Y=y, L=l)
                stopwords = tbrs.create_stopwords(cleaned_data,terms)

                prepro2 = Preprocessing()
                new_cleaned_data, new_terms = prepro2.remove_stopword(cleaned_data, stopwords)

                weight = Weighting(new_cleaned_data, new_terms)
                tfidf = weight.get_tf_idf_weighting()
                idf = weight.get_idf()

                nb = NBMultinomial()
                nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],stopwords,idf,tfidf)
                
                for j in range(len(data_test[i]["tweet"])):
                    print("Uji ke- " + str(j))
                    prediction = nb.predict(data_test[i]["tweet"][j])
                    y_test.append(data_test[i]["target"][j])
                    y_pred.append(prediction)

                cm = ConfusionMatrix()
                accuracy, precision, recall, fmeasure = cm.score(y_test, y_pred)
                list_acc.append(accuracy)
                list_prec.append(precision)
                list_recall.append(recall)
                list_fmeasure.append(fmeasure)

                accuracy_total_accumulation+=accuracy
                precision_total_accumulation+=precision
                recall_total_accumulation+=recall
                fmeasure_total_accumulation+=fmeasure
                print("--- %s seconds per fold ---" % (time.time() - start_time))

            print("--- %s seconds per combination---" % (time.time() - start_time_combination))
            accuracy_total = float(accuracy_total_accumulation/len(data_train))
            precision_total = float(precision_total_accumulation/len(data_train))
            recall_total = float(recall_total_accumulation/len(data_train))
            fmeasure_total = float(fmeasure_total_accumulation/len(data_train))
            for i in range(len(data_train)):
                fold_accuracy.append(accuracy_total)
                fold_precision.append(precision_total)
                fold_recall.append(recall_total)
                fold_fmeasure.append(fmeasure_total)
    #         break
    #         if count >= 2:
    #             break
    #     break
    # break

df = pd.DataFrame({'X':x_array,'Y':y_array,'L':l_array,'K-Fold':kfold_per_combination,'Accuracy':list_acc,'Precision':list_prec,'Recall':list_recall,'F-Measure':list_fmeasure,'Fold Accuracy':fold_accuracy,'Fold Precision':fold_precision,'Fold Recall':fold_recall,'Fold F-Measure':fold_fmeasure})
print(df)
# df.to_excel(r'cobabarunih.xlsx', index = False, header=True)
