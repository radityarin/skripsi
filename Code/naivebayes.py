import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from preprocessing import Preprocessing

class NaiveBayes(object):

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
        print(self.weighted_terms[word])
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

    def calculate_probability_gaussian(self, x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def fit(self, cleaned_data, terms, weighted_terms, target):
        self.cleaned_data = cleaned_data
        self.terms = terms
        self.weighted_terms = weighted_terms
        self.target = target
        
        for i in range(len(self.cleaned_data)):
            total_word = 0
            for term in self.terms:
                total_word += self.weighted_terms[term][i]
            self.total.append(total_word)
        
        for term in self.terms:
            self.con_prob_negative.append(self.calculate_probability_multinomial(term, 'Negatif'))
            self.con_prob_neutral.append(self.calculate_probability_multinomial(term, 'Netral'))
            self.con_prob_positive.append(self.calculate_probability_multinomial(term, 'Positif'))
            
        self.terms_con_prob = {}
        indexKomentar = 0
        for term in self.terms:
            temp = []
            temp.append(self.con_prob_negative[indexKomentar])
            temp.append(self.con_prob_neutral[indexKomentar])
            temp.append(self.con_prob_positive[indexKomentar])
            self.terms_con_prob[term] = temp
            print(term +","+str(temp))
            indexKomentar += 1

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

    def predict(self,data_test,expected_result):
        prepro = Preprocessing()
        cleaned_data_test, terms_test = prepro.preprocessing([data_test])

        for term in terms_test:
            if term in self.terms:
                self.used_terms.append(term)

        for term in self.used_terms:
            temp = []
            temp.append(self.terms_con_prob[term][0])
            temp.append(self.terms_con_prob[term][1])
            temp.append(self.terms_con_prob[term][2])
            self.used_terms_with_con_prob[term] = temp

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
 
