import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from preprocessing import Preprocessing
from weighting import Weighting
import numpy as np

class NBGaussian(object):

    def __init__(self):
        self.a=[]
        self.terms = []
        self.used_terms = []
        self.cleaned_data = []
        self.weighted_terms = {}
        self.used_terms_with_con_prob = {}
        self.total = []
        self.terms_con_prob = {}
        self.con_prob_negative = []
        self.con_prob_neutral = []
        self.con_prob_positive = []
        self.target = []
        self.class_name = []
        self.means= dict()
        self.stdev= dict()

    def calculate_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def get_class(self,target):
        for label in target:
            if label not in self.class_name:
                self.class_name.append(label)

    def get_value_with_specific_label(self,list_value,label):
        temp = []
        index_value = 0
        for value in list_value:
            if self.target[index_value] == label:
                temp.append(value)
            index_value+=1
        return temp

    def fit(self, cleaned_data, terms, target):
        self.cleaned_data = cleaned_data
        self.terms = terms
        self.target = target
        weight = Weighting(self.cleaned_data, self.terms)
        tfidf = weight.get_tf_idf_weighting()
        self.weighted_terms = tfidf
        self.get_class(target)
        for label in self.class_name:
            for term in self.terms:
                value = self.get_value_with_specific_label(self.weighted_terms[term],label)
                temp_mean = {}
                temp_stdev = {}
                temp_mean[term] = np.mean(value)
                temp_stdev[term] = np.std(value,ddof=1)
                self.means[label] = temp_mean
                self.stdev[label] = temp_stdev

        for term in self.terms:
            self.con_prob_negative.append(self.calculate_probability(term, 'Negatif'))
            self.con_prob_neutral.append(self.calculate_probability(term, 'Netral'))
            self.con_prob_positive.append(self.calculate_probability(term, 'Positif'))


    def getTotalDocument(self):
        return len(self.cleaned_data)

    def getTotalDocumentWithSpecificCategory(self,category):
        if category == 'Negatif':
            return int(len(self.cleaned_data)/3)
        elif category == 'Netral':
            return int(len(self.cleaned_data)/3)
        elif category == 'Positif':
            return int(len(self.cleaned_data)/3)
        else:
            return 0

    def separate_by_class(self,weighting,term):
        separated = dict()
        for label in self.class_name:
            temp = {}
            for t in term:
                # print(t + str(self.get_value_with_specific_label(weighting[t],label)))
                temp[t] = self.get_value_with_specific_label(weighting[t],label)
            separated[label] = temp

        return separated

    def predict(self,data_test,expected_result):
        prepro = Preprocessing()
        cleaned_data_test, terms_test = prepro.preprocessing([data_test])
        token = prepro.get_token()

        # weight_test = Weighting(cleaned_data_test, terms_test)
        # tfidf_test = weight_test.get_tf_idf_weighting()
        # tfidf_separated_by_class = self.separate_by_class(tfidf_test,terms_test)

        # print(tfidf_separated_by_class)
        prob = {}
        for label in self.class_name:
            for tok in token:
                prob[label]
































        # for term in terms_test:
        #     if term in self.terms:
        #         self.used_terms.append(term)

        # for term in self.used_terms:
        #     temp = []
        #     temp.append(self.terms_con_prob[term][0])
        #     temp.append(self.terms_con_prob[term][1])
        #     temp.append(self.terms_con_prob[term][2])
        #     self.used_terms_with_con_prob[term] = temp

        probabiltyNegatif = self.getTotalDocumentWithSpecificCategory(
            'Negatif') / self.getTotalDocument()
        probabiltyNetral = self.getTotalDocumentWithSpecificCategory(
            'Netral') / self.getTotalDocument()
        probabiltyPositif = self.getTotalDocumentWithSpecificCategory(
            'Positif') / self.getTotalDocument()

        negatif = 1
        netral = 1
        positif = 1
        for term in self.used_terms:
            negatif *= self.used_terms_with_con_prob[term][0]
            netral *= self.used_terms_with_con_prob[term][1]
            positif *= self.used_terms_with_con_prob[term][2]

        negatif = negatif * probabiltyNegatif
        netral = netral * probabiltyNetral
        positif = positif * probabiltyPositif

        finalResult = "" 
        if (positif > negatif and positif > netral):
            finalResult = "Positif" 
        elif negatif > positif and negatif > netral:
            finalResult = "Negatif" 
        elif netral > positif and netral > negatif:
            finalResult = "Netral" 

        print()
        print('Komentar yang diuji : ' + data_test)
        print('Expected Result : ' + expected_result)
        print('Output Result : ' + finalResult)

        return finalResult
 

import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting

data = pd.read_excel(
    r'skripsi.xlsx',"Data Manualisasi")
data_tweet = data['Tweet']
data_target = data['Klasifikasi']

prepro = Preprocessing()
stopwords_tbrs_manualisasi = ['kuliah','online','daring','yang','saja','tetap','mau','offline','aku','materi','ada','ini','kelas','cara','sama','jadi','kasih','tapi','lebih','kalau','dosen','beberapa','kenapa','new','normal','zona','lagi','sendiri','dengan','masih','jujur','corona','paham','sekali','nilai','bisa','ya','selesai','jelas','mohon','kaji']			

text_file = open('stopword_tala.txt', 'r')
stopword_tala = text_file.read().split('\n')
text_file.close()

new_cleaned_data, new_terms = prepro.preprocessing(data_tweet,stopword_tala)

nb = NBGaussian()

nb.fit(new_cleaned_data,new_terms,data_target)
nb.predict("Rasanya mau berhenti kuliah saja kalau daring begini, seperti bayar cuma cuma, materi dikasih secara online, disuruh baca sendiri tanpa ada yang menjelaskan, berasa otodidak :","Negatif")
# nb.predict(data_tweet[4],"Netral")
# nb.predict(data_tweet[7],"Positif")
# nb.predict("Saya bisa paham tapi kadang suka lupa","Netral")

