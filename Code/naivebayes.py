# import non alpha characters
import re
# import Stemmer Class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class NaiveBayes(object):

    def __init__(self):
        self.a=[]
        self.terms = []
        self.cleaned_data = []
        self.myTerms = {}
        self.total = []
        self.myConditionalProbability = {}

    def countWord(self,term,document):
        documentArray = document.split()
        count = 0
        for word in documentArray:
            if term == word:
                count += 1
        return count

    def countSpecificWordInCategory(self,word, category):
        counter = 0
        indexDocument = 0
        if category == 'Positif':
            for tf in self.myTerms[word]:
                if indexDocument < 15:
                    counter = counter + tf
                indexDocument += 1
        elif category == 'Negatif':
            for tf in self.myTerms[word]:
                if indexDocument >= 15:
                    counter = counter + tf
                indexDocument += 1
        return counter

    def countAllWordInCategory(self,category):
        counter = 0
        if category == 'Positif':
            indexDocument = 0
            for totalTiapDokumen in total:
                if indexDocument < 15:
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        elif category == 'Negatif':
            indexDocument = 0
            for totalTiapDokumen in total:
                if indexDocument >= 15:
                    counter = counter + totalTiapDokumen
                indexDocument += 1
        return counter

    def getTotalTerm(self):
        return len(termsTraining)

    def countConditionalProbablility(self,word, category):
        return (self.countSpecificWordInCategory(word, category) + 1) / (self.countAllWordInCategory(category) + self.getTotalTerm())

    def fit(self, cleaned_data, terms):
        self.cleaned_data = cleaned_data
        self.terms = terms

        for term in self.terms:
            temp = []
            for data in self.cleaned_data:
                temp.append(self.countWord(term, data))
            self.myTerms[term] = temp
        
        for i in range(len(self.cleaned_data)):
            total_word = 0
            for term in self.terms:
                total_word += self.myTerms[term][i]
            self.total.append(total_word)
        
        conProbPositive = []
        conProbNegative = []

        for term in self.terms:
            conProbPositive.append(self.countConditionalProbablility(term, 'Positif'))
            conProbNegative.append(self.countConditionalProbablility(term, 'Negatif'))

        self.myConditionalProbability = {}
        indexKomentar = 0
        for term in self.terms:
            temp = []
            temp.append(conProbPositive[indexKomentar])
            temp.append(conProbNegative[indexKomentar])
            self.myConditionalProbability[term] = temp
            indexKomentar += 1
