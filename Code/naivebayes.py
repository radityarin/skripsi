import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from preprocessing import Preprocessing

class NaiveBayes(object):

    def __init__(self):
        self.a=[]
        self.terms = []
        self.used_terms = []
        self.cleaned_data = []
        self.terms_raw_tf = {}
        self.used_terms_with_con_prob = {}
        self.total = []
        self.terms_con_prob = {}
        self.con_prob_positive = []
        self.con_prob_neutral = []
        self.con_prob_negative = []

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
        if category == 'Positif':
            for tf in self.terms_raw_tf[word]:
                if indexDocument < 15:
                    counter = counter + tf
                indexDocument += 1
        elif category == 'Negatif':
            for tf in self.terms_raw_tf[word]:
                if indexDocument >= 15:
                    counter = counter + tf
                indexDocument += 1
        return counter

    def countAllWordInCategory(self,category):
        counter = 0
        if category == 'Positif':
            indexDocument = 0
            for totalTiapDokumen in self.total:
                if indexDocument < 15:
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        elif category == 'Negatif':
            indexDocument = 0
            for totalTiapDokumen in self.total:
                if indexDocument >= 15:
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        return counter

    def getTotalTerm(self):
        return len(self.terms)

    def countConditionalProbablility(self,word, category):
        return (self.countSpecificWordInCategory(word, category) + 1) / (self.countAllWordInCategory(category) + self.getTotalTerm())

    def fit(self, cleaned_data, terms):
        self.cleaned_data = cleaned_data
        self.terms = terms

        for term in self.terms:
            temp = []
            for data in self.cleaned_data:
                temp.append(self.countWord(term, data))
            self.terms_raw_tf[term] = temp
        
        for i in range(len(self.cleaned_data)):
            total_word = 0
            for term in self.terms:
                total_word += self.terms_raw_tf[term][i]
            self.total.append(total_word)
        
        self.con_prob_positive = []
        self.con_prob_negative = []

        for term in self.terms:
            self.con_prob_positive.append(self.countConditionalProbablility(term, 'Positif'))
            self.con_prob_negative.append(self.countConditionalProbablility(term, 'Negatif'))

        self.terms_con_prob = {}
        indexKomentar = 0
        for term in self.terms:
            temp = []
            temp.append(self.con_prob_positive[indexKomentar])
            temp.append(self.con_prob_negative[indexKomentar])
            self.terms_con_prob[term] = temp
            indexKomentar += 1

    def getTotalDocument(self):
        return len(self.cleaned_data)

    def getTotalDocumentWithSpecificCategory(self,category):
        if category == 'Positif':
            return int(len(self.cleaned_data)/2)
        elif category == 'Neutral':
            return int(len(self.cleaned_data)/2)
        elif category == 'Negatif':
            return int(len(self.cleaned_data)/2)
        else:
            return 0

    def predict(self,data_test,expected_result):
        prepro = Preprocessing()
        cleaned_data_test, terms_test = prepro.preprocessing([data_test])
        self.used_terms = []
        for term in terms_test:
            if term in self.terms:
                self.used_terms.append(term)

        for term in self.used_terms:
            temp = []
            temp.append(self.terms_con_prob[term][0])
            temp.append(self.terms_con_prob[term][1])
            self.used_terms_with_con_prob[term] = temp

        probabiltyPositif = self.getTotalDocumentWithSpecificCategory(
            'Positif') / self.getTotalDocument()
        probabiltyNegatif = self.getTotalDocumentWithSpecificCategory(
            'Negatif') / self.getTotalDocument()

        positif = 1
        negatif = 1
        
        for term in self.used_terms:
            positif *= self.used_terms_with_con_prob[term][0]
            negatif *= self.used_terms_with_con_prob[term][1]

        positif = positif * probabiltyPositif
        negatif = negatif * probabiltyNegatif
        print()
        print(positif)
        print(negatif)

        finalResult = True if positif > negatif else False

        print()
        print('Komentar yang diuji : ' + data_test)
        print('Expected Result : ' + ('Positif' if expected_result == True else 'Negatif'))
        print('Output Result : ' + ('Positif' if finalResult == True else 'Negatif'))
