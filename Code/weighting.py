import math

class Weighting(object):

    def __init__(self,data,terms):

        '''
            Parameters:
            1. data ex: = ["lorem ipsum","dolor sit amet"]
            2. terms ex: = ["ipsum","sit"]
        '''

        self.data = data
        self.terms = terms
        self.raw_tf = {}
        self.log_tf = {}
        self.idf = []
        self.tf_idf = {}
    
    def countWord(self,term, document):
        count = 0
        for word in document.split():
            if term == word:
                count+=1
        return count

    def dftCount(self,numbers):
        count = len(self.data)
        for number in numbers:
            if number == 0:
                count-=1
        return count

    def get_raw_tf_weighting(self):
        for term in self.terms:
            temp = []
            for data in self.data:
                temp.append(self.countWord(term, data))
            self.raw_tf[term] = temp
        return self.raw_tf

    def get_log_tf_weighting(self):
        self.get_raw_tf_weighting()
        for term in self.terms:
            temp = []
            for i in range(len(self.raw_tf[term])):
                tf = 0 if (self.raw_tf[term])[i] == 0 else 1+math.log((self.raw_tf[term])[i],10)
                temp.append(tf)
            self.log_tf[term] = temp
        return self.log_tf

    def calculate_idf(self):
        for term in self.terms:
            df = self.dftCount(self.raw_tf[term])
            idf_value = math.log(len(self.data)/df,10)
            self.idf.append(idf_value)

    def get_tf_idf_weighting(self):
        self.get_log_tf_weighting()
        self.calculate_idf()
        count = 0
        for term in self.terms:
            temp = []
            for i in range(len(self.data)):
                tfidf_value = self.log_tf[term][i]*self.idf[count]
                temp.append(tfidf_value)
            self.tf_idf[term] = temp
            count+=1
        return self.tf_idf
