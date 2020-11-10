from firebase import firebase
from uuid import UUID
import uuid
import json
import pandas as pd

def uuid_convert(o):
    if isinstance(o, UUID):
        return o.hex

firebase = firebase.FirebaseApplication('https://bantulabelindong.firebaseio.com/', None)

def insert_tweet(tweet):
    uid = uuid_convert(uuid.uuid4())
    data = dict()
    data['idTweet'] = uid
    data['tweet'] = tweet
    location = '/Data Tweet/'
    firebase.post(location,data)


data = pd.read_excel(
    r'/Users/radityarin/Documents/Kuliah/Skripsi/Code/helper/dataset_radit.xlsx',"Sheet1")
data_tweet = data['Tweet']
# data_target = data['Klasifikasi']

for tweet in data_tweet:
    print(tweet)
    insert_tweet(tweet)