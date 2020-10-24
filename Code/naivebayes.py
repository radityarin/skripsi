import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from preprocessing import Preprocessing
from weighting import Weighting

class NBMultinomial(object):

    def __init__(self):
        self.a=[]
        self.terms = []
        self.used_terms = []
        self.cleaned_data = []
        self.weighted_terms = {}
        self.used_terms_with_likelihood = {}
        self.total = []
        self.likelihood = {}
        self.con_prob_negative = []
        self.con_prob_neutral = []
        self.con_prob_positive = []
        self.prior_negative = []
        self.prior_neutral = []
        self.prior_positive = []
        self.target = []
        self.stopwords = []

    def countWord(self,term,document):
        documentArray = document.split()
        count = 0
        for word in documentArray:
            if term == word:
                count += 1
        return count

    def countSpecificWordInCategory(self,word,category):
        counter = 0
        indexDocument = 0
        if category == 'Negatif':
            for tf in self.weighted_terms[word]:
                if self.target[indexDocument]=="Negatif":
                    counter = counter + tf
                indexDocument += 1
        elif category == 'Netral':
            for tf in self.weighted_terms[word]:
                if self.target[indexDocument]=="Netral":
                    counter = counter + tf
                indexDocument += 1
        elif category == 'Positif':
            for tf in self.weighted_terms[word]:
                if self.target[indexDocument]=="Positif":
                    counter = counter + tf
                indexDocument += 1
        return counter

    def countAllWordInCategory(self,category):
        counter = 0
        if category == 'Negatif':
            indexDocument = 0
            for totalTiapDokumen in self.total:
                if self.target[indexDocument]=="Negatif":
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        elif category == 'Netral':
            indexDocument = 0
            for totalTiapDokumen in self.total:
                if self.target[indexDocument]=="Netral":
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        elif category == 'Positif':
            indexDocument = 0
            for totalTiapDokumen in self.total:
                if self.target[indexDocument]=="Positif":
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        return counter

    def getTotalTerm(self):
        return len(self.terms)

    def calculate_probability_multinomial(self,word, category):
        return (self.countSpecificWordInCategory(word, category) + 1) / (self.countAllWordInCategory(category) + self.getTotalTerm())

    def fit(self, cleaned_data, terms, target, stopwords, weight = None):
        self.cleaned_data = cleaned_data
        self.terms = terms
        self.target = target
        if weight == None:
            weighting = Weighting(self.cleaned_data, self.terms)
            self.weighted_terms = weighting.get_tf_idf_weighting()
        else:
            self.weighted_terms = weight
        self.stopwords = stopwords

        for i in range(len(self.cleaned_data)):
            total_word = 0
            for term in self.terms:
                total_word += self.weighted_terms[term][i]
            self.total.append(total_word)
        
        for term in self.terms:
            self.con_prob_negative.append(self.calculate_probability_multinomial(term, 'Negatif'))
            self.con_prob_neutral.append(self.calculate_probability_multinomial(term, 'Netral'))
            self.con_prob_positive.append(self.calculate_probability_multinomial(term, 'Positif'))
            
        self.likelihood = {}
        indexKomentar = 0
        for term in self.terms:
            temp = []
            temp.append(self.con_prob_negative[indexKomentar])
            temp.append(self.con_prob_neutral[indexKomentar])
            temp.append(self.con_prob_positive[indexKomentar])
            self.likelihood[term] = temp
            # print(term +","+str(temp))
            indexKomentar += 1

        self.prior_negative = self.getTotalDocumentWithSpecificCategory(
            'Negatif') / self.getTotalDocument()
        self.prior_neutral = self.getTotalDocumentWithSpecificCategory(
            'Netral') / self.getTotalDocument()
        self.prior_positive = self.getTotalDocumentWithSpecificCategory(
            'Positif') / self.getTotalDocument()
        

    def getTotalDocument(self):
        return len(self.cleaned_data)

    def getTotalDocumentWithSpecificCategory(self,category):
        if category == 'Negatif':
            return len([tgt for tgt in self.target if tgt == "Negatif"])
        elif category == 'Netral':
            return len([tgt for tgt in self.target if tgt == "Netral"])
        elif category == 'Positif':
            return len([tgt for tgt in self.target if tgt == "Positif"])
        else:
            return 0

    def predict(self,data_test,expected_result):
        prepro = Preprocessing()
        cleaned_data_test, terms_test = prepro.preprocessing([data_test],self.stopwords)
        terms_test = prepro.get_token()
        for term in terms_test:
            if term in self.terms:
                self.used_terms.append(term)

        for term in self.used_terms:
            temp = []
            temp.append(self.likelihood[term][0])
            temp.append(self.likelihood[term][1])
            temp.append(self.likelihood[term][2])
            # print(term +","+str(temp))
            self.used_terms_with_likelihood[term] = temp

        # a = str(probabiltyNegatif) + " * "
        # b = str(probabiltyNetral) + " * "
        # c = str(probabiltyPositif) + " * "
        negatif = 1
        netral = 1
        positif = 1
        for term in self.used_terms:
            # print(term)
            # a+= str(self.used_terms_with_likelihood[term][0]) + " * "
            # b+= str(self.used_terms_with_likelihood[term][1]) + " * "
            # c+= str(self.used_terms_with_likelihood[term][2]) + " * "
            negatif *= self.used_terms_with_likelihood[term][0]
            netral *= self.used_terms_with_likelihood[term][1]
            positif *= self.used_terms_with_likelihood[term][2]
        
        # print(a)
        # print(b)
        # print(c)
        
        negatif = negatif * self.prior_negative
        netral = netral * self.prior_neutral
        positif = positif * self.prior_positive
        # print(str(negatif) + ", " + str(netral) + ", " +str(positif))
        finalResult = "" 
        if (positif > negatif and positif > netral):
            finalResult = "Positif" 
        elif negatif > positif and negatif > netral:
            finalResult = "Negatif" 
        elif netral > positif and netral > negatif:
            finalResult = "Netral" 

        print('Komentar yang diuji : ' + data_test)
        print('Expected Result : ' + expected_result)
        print('Output Result : ' + finalResult)
        print()

        return finalResult
 
