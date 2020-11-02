import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessing(object):

    def __init__(self):
        self.cleaned_data = []
        self.terms = []
        self.setup_library()
        self.token = []

    def setup_library(self):
        stemmerFactory = StemmerFactory()
        self.stemmer = stemmerFactory.create_stemmer()

    def preprocessing(self,data,stopwords=None):
        for i in range(len(data)):
            case_folding = data[i].lower()
            # print()
            # print(case_folding)
            remove_newline = case_folding.replace("\n"," ")
            cleaning = re.sub(r'[^a-zA-Z]', " ",remove_newline)
            # print(cleaning)
            # print(cleaning.split())
            if stopwords != None:
                # print("stopwords")
                # print(stopwords)
                # filtered_words = [word for word in cleaning.split() if word not in stopwords]
                # print(filtered_words)
                stemming = self.stemmer.stem(cleaning)
                filtered_words = [word for word in stemming.split() if word not in stopwords]
                # print(stemming)
                self.cleaned_data.append(" ".join(filtered_words))
                tokenizing = [word for word in filtered_words if word.isalpha()]
            else:
                stemming = self.stemmer.stem(cleaning)
                # print(stemming)
                self.cleaned_data.append(stemming)
                tokenizing = [word for word in stemming.split() if word.isalpha()]
            # print("\n")
            # print(tokenizing)
            for word in tokenizing:
                self.token.append(word)
                if word not in self.terms:
                    self.terms.append(word)
        return self.cleaned_data, self.terms

    def get_token(self):
        return self.token

    def remove_stopword(self,cleaned_data,stopwords):
        new_cleaned_data = []
        new_terms = []
        for data in cleaned_data:
            filtered_words = [word for word in data.split() if word not in stopwords]
            new_cleaned_data.append(" ".join(filtered_words))
            for word in filtered_words:
                if word not in new_terms:
                    new_terms.append(word)
        return new_cleaned_data, new_terms
