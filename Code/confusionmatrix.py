import pandas as pd

# class ConfusionMatrix(object):

#     def __init__(self):


#     def add_data(self,actual,predicted):
conf = pd.DataFrame({'Negatif':{'Negatif':1,'Netral':4,'Positif':3},'Netral':{'Negatif':2,'Netral':0,'Positif':1},'Positif':{'Negatif':0,'Netral':0,'Positif':2},})
print(conf)        



# actual = ["Negatif","Positif","Netral","Positif","Positif","Positif","Negatif","Negatif","Netral","Netral"]
# predicted = ["Positif","Positif","Positif","Positif","Positif","Netral","Negatif","Positif","Netral","Positif"]