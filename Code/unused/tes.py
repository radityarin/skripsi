import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NBMultinomial
from weighting import Weighting
from kfoldnew import KFold
import time
from sklearn.metrics import precision_recall_fscore_support as score

data = pd.read_excel(
    r'Skripsi.xlsx',"Data Coding")
data_tweet = data['Tweet']
data_target = data['Label']

kfold = KFold(data_tweet,data_target,10)
data_train, data_test = kfold.get_data_sequence()

# for word in data_train[9]["tweet"]:
#     print(word)

for i in range(len(data_train)):
    print(len(data_train[i]["tweet"]))