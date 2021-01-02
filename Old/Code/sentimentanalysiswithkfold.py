import pandas as pd
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold
from confusionmatrix import ConfusionMatrix


data = pd.read_excel(
    r'Skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,10)
data_train, data_test = kfold.get_data_sequence()

x_array = []
y_array = []
l_array = []
kfold_per_combination = []
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
combination_accuracy = []

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

            for i in range(len(data_train)):
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
                
                correct_ans = 0
                for j in range(len(data_test[i]["tweet"])):
                    prediction = nb.predict(data_test[i]["tweet"][j],data_test[i]["target"][j])
                    y_test.append(data_test[i]["target"][j])
                    y_pred.append(prediction)
                    if prediction == data_test[i]["target"][j]:
                        correct_ans+=1

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
                accuracy_total_accumulation+=accuracy

            accuracy_total = float(accuracy_total_accumulation/len(data_train))
            for i in range(len(data_train)):
                combination_accuracy.append(accuracy_total)
            if count >= 3:
                break
        break
    break

df = pd.DataFrame({'X':x_array,'Y':y_array,'L':l_array,'K-Fold':kfold_per_combination,'AccNeg':acc_neg,'AccNet':acc_net,'AccPos':acc_pos,'PreNeg':prec_neg,'PreNet':prec_net,'PrePos':prec_pos,'RecNeg':recall_neg,'RecNet':recall_net,'RecPos':recall_pos,'FMeNeg':fme_neg,'FMeNet':fme_net,'FmePos':fme_pos,'Accuracy':acc_per_fold,'Combination Acc':combination_accuracy})
print(df)
df.to_excel(r'outputnew.xlsx', index = False, header=True)
