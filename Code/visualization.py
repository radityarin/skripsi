import pandas as pd
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfold import KFold
from confusionmatrix import ConfusionMatrix

from wordcloud import WordCloud, ImageColorGenerator
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_excel(
    r'C:\Users\PPATK\Documents\skripsi\Code\skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

# kfold = KFold(data_tweet,data_target,10)
# data_train, data_test = kfold.get_data_sequence()
prepro = Preprocessing()
cleaned_data, terms = prepro.preprocessing(data_tweet[0:100].to_list())
print(data_tweet[0:100])

tbrs = TermBasedRandomSampling(X=10, Y=10, L=40)
stopwords = tbrs.create_stopwords(cleaned_data,terms)

prepro2 = Preprocessing()
new_cleaned_data, new_terms = prepro2.remove_stopword(cleaned_data, stopwords)

allWordsNegative = []
for data in (new_cleaned_data):
    for d in data.split():
        allWordsNegative.append(d)

allWordsNegative = " ".join(allWordsNegative)



prepro = Preprocessing()
cleaned_data, terms = prepro.preprocessing(data_tweet[101:200].to_list())
print(data_tweet[101:200])

tbrs = TermBasedRandomSampling(X=10, Y=10, L=40)
stopwords = tbrs.create_stopwords(cleaned_data,terms)

prepro2 = Preprocessing()
new_cleaned_data, new_terms = prepro2.remove_stopword(cleaned_data, stopwords)

allWordsNetral = []
for data in (new_cleaned_data):
    for d in data.split():
        allWordsNetral.append(d)

allWordsNetral = " ".join(allWordsNetral)

prepro = Preprocessing()
cleaned_data, terms = prepro.preprocessing(data_tweet[201:300].to_list())
print(data_tweet[201:300])

tbrs = TermBasedRandomSampling(X=10, Y=10, L=40)
stopwords = tbrs.create_stopwords(cleaned_data,terms)

prepro2 = Preprocessing()
new_cleaned_data, new_terms = prepro2.remove_stopword(cleaned_data, stopwords)

allWordsPositive = []
for data in (new_cleaned_data):
    for d in data.split():
        allWordsPositive.append(d)

allWordsPositive = " ".join(allWordsPositive)



wordCloudNegative = WordCloud(colormap="Blues", width=1600, height=800, random_state=30, max_font_size=200, min_font_size=20).generate(allWordsNegative)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordCloudNegative, interpolation="bilinear")
plt.axis('off')
plt.show()


wordCloudNetral = WordCloud(colormap="Blues", width=1600, height=800, random_state=30, max_font_size=200, min_font_size=20).generate(allWordsNetral)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordCloudNetral, interpolation="bilinear")
plt.axis('off')
plt.show()


wordCloudPositive = WordCloud(colormap="Blues", width=1600, height=800, random_state=30, max_font_size=200, min_font_size=20).generate(allWordsPositive)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordCloudPositive, interpolation="bilinear")
plt.axis('off')
plt.show()