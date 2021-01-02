import pandas as pd
import pprint

class KFold(object):

    def __init__(self, data, target, fold, test_size=0.2):
        self.data = data
        self.target = target
        self.fold = fold
        self.NEGATIVE = "Negatif"
        self.NETRAL = "Netral"
        self.POSITIVE = "Positif"
        self.test_size = test_size
        self._prepare_data()

    def _prepare_data(self):
        self.data_negative = []
        self.data_netral = []
        self.data_positive = []

        for i in range(len(self.data)):
            if self.target[i] == self.NEGATIVE:
                self.data_negative.append(self.data[i])
            elif self.target[i] == self.NETRAL:
                self.data_netral.append(self.data[i])
            elif self.target[i] == self.POSITIVE:
                self.data_positive.append(self.data[i])
            else:
                return None

    def get_data_sequence(self):
        data_test_size = len(self.data_negative) / self.fold
        self.test_size = len(self.data_negative) * self.test_size
        data_test_start_index = 0
        data_test_end_index = self.test_size

        data_train = []
        data_test = []
        for i in range(self.fold):
            data_neg_test = []
            data_net_test = []
            data_pos_test = []
            data_neg_train = []
            data_net_train = []
            data_pos_train = []
            
            check = False

            for j in range(len(self.data_negative)):
                if j >= data_test_start_index and j < data_test_end_index:
                    temp = []
                    temp.append(self.data_negative[j])
                    temp.append(self.NEGATIVE)
                    data_neg_test.append(temp)
                    temp = []
                    temp.append(self.data_netral[j])
                    temp.append(self.NETRAL)
                    data_net_test.append(temp)
                    temp = []
                    temp.append(self.data_positive[j])
                    temp.append(self.POSITIVE)
                    data_pos_test.append(temp)
                    if j == 99 and data_test_start_index == 90:
                        for k in range(10):
                            temp = []
                            temp.append(self.data_negative[k])
                            temp.append(self.NEGATIVE)
                            data_neg_test.append(temp)
                            temp = []
                            temp.append(self.data_netral[k])
                            temp.append(self.NETRAL)
                            data_net_test.append(temp)
                            temp = []
                            temp.append(self.data_positive[k])
                            temp.append(self.POSITIVE)
                            data_pos_test.append(temp)
                else:
                    if i!=9:
                        temp = []
                        temp.append(self.data_negative[j])
                        temp.append(self.NEGATIVE)
                        data_neg_train.append(temp)
                        temp = []
                        temp.append(self.data_netral[j])
                        temp.append(self.NETRAL)
                        data_net_train.append(temp)
                        temp = []
                        temp.append(self.data_positive[j])
                        temp.append(self.POSITIVE)
                        data_pos_train.append(temp)
                    else:
                        if j > 9:
                            temp = []
                            temp.append(self.data_negative[j])
                            temp.append(self.NEGATIVE)
                            data_neg_train.append(temp)
                            temp = []
                            temp.append(self.data_netral[j])
                            temp.append(self.NETRAL)
                            data_net_train.append(temp)
                            temp = []
                            temp.append(self.data_positive[j])
                            temp.append(self.POSITIVE)
                            data_pos_train.append(temp)

            data_combine_test = data_neg_test + data_net_test + data_pos_test
            data_combine_train = data_neg_train + data_net_train + data_pos_train

            data_combine_test_tweet = [data[0] for data in data_combine_test]
            data_combine_test_target = [data[1] for data in data_combine_test]

            data_combine_train_tweet = [data[0] for data in data_combine_train]
            data_combine_train_target = [data[1]
                                         for data in data_combine_train]

            data_dict_test = {}
            data_dict_train = {}
            data_dict_test["tweet"] = data_combine_test_tweet
            data_dict_test["target"] = data_combine_test_target

            data_dict_train["tweet"] = data_combine_train_tweet
            data_dict_train["target"] = data_combine_train_target

            data_train.append(data_dict_train)
            data_test.append(data_dict_test)
            data_test_start_index += data_test_size
            data_test_end_index += data_test_size

        return data_train, data_test


# data = pd.read_excel(
#     r'Skripsi.xlsx', "Data Coding Latihan")
# data_tweet = data['Tweet']
# data_target = data['Label']

# kfold = KFold(data_tweet, data_target, 10)
# data_train, data_test = kfold.get_data_sequence()
# print(data_train[2])
# print(data_test[2])
# print(len(data_train))
# for i in range(len(data_test)):
#     print("\n ==== FOLD " + str(i+1) + " ==== ")
#     print("TRAIN")
#     pp.pprint(data_train[i])
#     print("\nTEST")
#     pp.pprint(data_test[i])
#     input()
