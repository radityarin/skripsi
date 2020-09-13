import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from random import randint

class TermBasedRandomSampling(object):

    '''
        TERM BASED RANDOM SAMPLING

        Parameters:
            1. Y (int) = the number of selection step Y times
            2. X (int) = the number of top ranked (least weighted) extraction in every looping
            3. L (int) = the number of stopwords to build

        Algorithm:
            Repeat Y times, where Y is a parameter:
                Randomly choose a term in the lexicon file, we shall call it ωrandom
                Retrieve all the documents in the corpus that contains ωrandom
                Use the refined Kullback-Leibler divergence measure to assign a weight to every term in the retrieved documents. The assigned weight will give us some indication of how important the term is.
                Divide each term’s weight by the maximum weight of all terms. As a result, all the weights are controlled within [0,1]. In other words, normalise each weighted term by the maximum weight.
                Rank the weighted terms by their associated weight in ascending order. Since the less informative a term is, the less useful a term is and hence, the more likely it is a stopword.
                Extract the top X top-ranked (i.e. least weighted), where X is a param- eter.
            You now have an array of length X ∗ Y . Each element in the array is associated to a weight.
            Shrink the array by merging the elements containing the same term and take the average of the term’s associated weights. For example, if the term “retrieval” appears three times in the array and its weights are 0.5, 0.4 and 0.3 respectively, we merge these three elements together into one single one and the weight of the term “retrieval” will become
            (0.5 + 0.4 + 0.3) = 0.4 3
            Rank the shrunk array in increasing order depending on the term’s weight. In other words, sort the array in ascending order.
            Extract the L top-ranked terms as stopword list for the collection. L is a parameter. Therefore, it is often a good idea to use trial and error.
    '''
    def __init__(self, Y = 50, X = 30, L = 200):
        self.cleaned_data = []
        self.terms = []
        self.stemmer = None
        self.setup_library()
        self.Y = Y
        self.X = X
        self.L = L
        self.used_token = []
        self.token_weight = []

    def setup_library(self):
        stemmerFactory = StemmerFactory()
        self.stemmer = stemmerFactory.create_stemmer()

    def generate_random_words(self,token):
        return token[randint(0,len(token)-1)]

    def get_documents_contains_words(self,words,documents):
        sampled_documents = []
        count = 1
        for tweet in documents:
            if words in tweet.split():
                sampled_documents.append(tweet)
            count+=1
        return sampled_documents

    def count_words(self,word, documents):
        count = 0
        for d in documents:
            for w in d.split():
                if word == w:
                    count+=1
        return float(count)

    def get_sum_of_the_length_document(self,documents):
        sum = 0
        for d in documents:
            sum+=len(d.split())
        return float(sum)

    def get_term(self,documents):
        term = []
        for d in documents:
            for word in d.split():
                if word not in term:
                    term.append(word)
        return term

    def get_total_token(self,token):
        return float(len(token))

    def kl_div(self,word,sampled_documents):
        tf_x = self.count_words(word,sampled_documents)
        l_x = self.get_sum_of_the_length_document(sampled_documents)
        p_x = tf_x / l_x
        F = self.count_words(word,self.cleaned_data)
        token_c = self.get_total_token(self.terms)
        p_c = F / token_c
        w_t = p_x * np.log2(p_x/p_c)
        return w_t

    def create_stopwords(self,cleaned_data,terms):

        '''
            Parameters:
            1. cleaned_data = array of documents
            2. terms = array of documents
            ex : ["Lorem ipsum","Dolor sit amet"]
        '''
        self.cleaned_data = cleaned_data
        self.terms = terms
        
        # Repeat Y times, where Y is a parameter:
        for i in range(self.Y):
            # Randomly choose a term in the lexicon file, we shall call it ωrandom
            w_random = self.generate_random_words(self.terms)
            
            # Retrieve all the documents in the corpus that contains ωrandom
            sampled_documents = self.get_documents_contains_words(w_random, self.cleaned_data)
            
            # Use the refined Kullback-Leibler divergence measure to assign a weight to every term in the retrieved documents. The assigned weight will give us some indication of how important the term is.
            term_sampled_documents = self.get_term(sampled_documents)

            token_w = {}
            for word in term_sampled_documents:
                token_w[word] = self.kl_div(word,sampled_documents)
                if word not in self.used_token:
                    self.used_token.append(word)

            # Divide each term’s weight by the maximum weight of all terms. As a result, all the weights are controlled within [0,1]. In other words, normalise each weighted term by the maximum weight.
            maximum = max(token_w, key=token_w.get)  
            minimum = min(token_w, key=token_w.get)
            max_weight_term = token_w[maximum]
            min_weight_term = token_w[minimum]

            normalized_term_weight = {}
            for k,v in token_w.items():
                normalized_term_weight[k] = ( v - min_weight_term) / (max_weight_term - min_weight_term)
            
            # Rank the weighted terms by their associated weight in ascending order. Since the less informative a term is, the less useful a term is and hence, the more likely it is a stopword.
            sort_term_weight = sorted(normalized_term_weight.items(), key=lambda x: x[1])

            # Extract the top X top-ranked (i.e. least weighted), where X is a param- eter.
            sorted_term_weight = {}
            count = 0
            for i in sort_term_weight:
                if count < self.X:
                    sorted_term_weight[i[0]] = i[1]
                else:
                    break
                count+=1

            self.token_weight.append(sorted_term_weight)

        weighted_token = {}
        for used_tok in self.used_token:
            temp = []
            for tok_w in self.token_weight:
                if used_tok in tok_w:
                    temp.append(tok_w[used_tok])
            weighted_token[used_tok] = temp

        # You now have an array of length X ∗ Y . Each element in the array is associated to a weight.
        # Shrink the array by merging the elements containing the same term and take the average of the term’s associated weights. For example, if the term “retrieval” appears three times in the array and its weights are 0.5, 0.4 and 0.3 respectively, we merge these three elements together into one single one and the weight of the term “retrieval” will become
        # (0.5 + 0.4 + 0.3) / 3 = 0.4
        merged_weighted_token = {}
        for k,v in weighted_token.items():
            if len(v) != 0:
                merged_weighted_token[k] = np.mean(v)
        
        # Rank the shrunk array in increasing order depending on the term’s weight. In other words, sort the array in ascending order.
        sorted_merged_weighted_token = sorted(merged_weighted_token.items(), key=lambda x: x[1])

        # Extract the L top-ranked terms as stopword list for the collection. L is a parameter. Therefore, it is often a good idea to use trial and error.
        sorted_final_weight = {}
        count = 0
        for i in sorted_merged_weighted_token:
            if count < self.L:
                sorted_final_weight[i[0]] = i[1]
            else:
                break
            count+=1

        stopwords = []
        for k, v in sorted_final_weight.items():
            print(k + " : "+str(v))
            stopwords.append(k)

        return stopwords
