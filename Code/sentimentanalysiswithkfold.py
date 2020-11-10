import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold

data = pd.read_excel(
    r'Skripsi.xlsx',"Data Coding Latihan")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,5)
data_train, data_test = kfold.get_data_sequence()

for i in range(5):
    # TAHAP PEMBUATAN STOPWORD
    print("FIRST PREPRO")
    prepro = Preprocessing()
    cleaned_data, terms = prepro.preprocessing(data_train[i]["tweet"])
    
    print("CREATE TBRS")
    tbrs = TermBasedRandomSampling(L=20)
    stopwords = tbrs.create_stopwords(cleaned_data,terms)

    # TAHAP PELATIHAN
    print("REMOVE STOPWORDS")
    new_cleaned_data, new_terms = prepro.remove_stopword(cleaned_data, stopwords)
    print("START WEIGHTING")
    weight = Weighting(new_cleaned_data, new_terms)
    tfidf = weight.get_tf_idf_weighting()
    idf = weight.get_idf()

    print("START FITTING")
    nb = NBMultinomial()
    nb.fit(new_cleaned_data,new_terms,data_train[i]["target"],stopwords,idf,tfidf)
    
    print("START PREDICTING")
    print(data_test[i])

    correct_ans = 0
    for j in range(len(data_test[i]["tweet"])):
        prediction = nb.predict(data_test[i]["tweet"][j],data_test[i]["target"][j])
        if prediction == data_test[i]["target"][j]:
            correct_ans+=1

    accuracy = (float(correct_ans) / len(data_test[i]["tweet"])) * 100
    print("Accuracy fold "+ str(i) + " : " + str(accuracy) + "%")