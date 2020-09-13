import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessing(object):

    def __init__(self):
        self.cleaned_data = []
        self.terms = []
        self.setup_library()

    def setup_library(self):
        stemmerFactory = StemmerFactory()
        self.stemmer = stemmerFactory.create_stemmer()

    def preprocessing(self,data,stopwords=None):
        for i in range(len(data)):
            case_folding = data[i].lower()
            remove_newline = case_folding.replace("\n"," ")
            cleaning = re.sub(r'[^a-zA-Z]', " ",remove_newline)
            if stopwords != None:
                filtered_words = [word for word in cleaning.split() if word not in stopwords]
                stemming = self.stemmer.stem(" ".join(filtered_words))
                print(filtered_words)
            else:
                stemming = self.stemmer.stem(cleaning)
            print(stemming)
            tokenizing = [word for word in stemming.split() if word.isalpha()]
            self.cleaned_data.append(stemming)
            for word in tokenizing:
                if word not in self.terms:
                    self.terms.append(word)
        return self.cleaned_data, self.terms

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
