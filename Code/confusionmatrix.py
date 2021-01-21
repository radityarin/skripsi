import pandas as pd
import numpy as np


class ConfusionMatrix(object):

    def __init__(self):
        self.NEGATIVE = "Negatif"
        self.NETRAL = "Netral"
        self.POSITIVE = "Positif"
        self.actual  = []
        self.predicted = []

    def create_confusion_matrix(self,actual,predicted):
        self.cm = pd.DataFrame(np.zeros((3, 3), dtype=int), index=['Actually Negatif', 'Actually Netral', 'Actually Positif'], columns=[
                               'Predicted Negatif', 'Predicted Netral', 'Predicted Positif'])
        for i in range(len(actual)):
            if actual[i] == self.NEGATIVE:
                if predicted[i] == self.NEGATIVE:
                    self.cm.loc["Actually " + self.NEGATIVE,
                                "Predicted " + self.NEGATIVE] += 1
                elif predicted[i] == self.NETRAL:
                    self.cm.loc["Actually " + self.NEGATIVE,
                                "Predicted " + self.NETRAL] += 1
                elif predicted[i] == self.POSITIVE:
                    self.cm.loc["Actually " + self.NEGATIVE,
                                "Predicted " + self.POSITIVE] += 1
            elif actual[i] == self.NETRAL:
                if predicted[i] == self.NEGATIVE:
                    self.cm.loc["Actually " + self.NETRAL,
                                "Predicted " + self.NEGATIVE] += 1
                elif predicted[i] == self.NETRAL:
                    self.cm.loc["Actually " + self.NETRAL,
                                "Predicted " + self.NETRAL] += 1
                elif predicted[i] == self.POSITIVE:
                    self.cm.loc["Actually " + self.NETRAL,
                                "Predicted " + self.POSITIVE] += 1
            elif actual[i] == self.POSITIVE:
                if predicted[i] == self.NEGATIVE:
                    self.cm.loc["Actually " + self.POSITIVE,
                                "Predicted " + self.NEGATIVE] += 1
                elif predicted[i] == self.NETRAL:
                    self.cm.loc["Actually " + self.POSITIVE,
                                "Predicted " + self.NETRAL] += 1
                elif predicted[i] == self.POSITIVE:
                    self.cm.loc["Actually " + self.POSITIVE,
                                "Predicted " + self.POSITIVE] += 1

    def find_tp_fn_fp_tn(self):
        self.tp_negatif = self.cm.loc["Actually " +
                                      self.NEGATIVE, "Predicted " + self.NEGATIVE]
        self.tp_netral = self.cm.loc["Actually " +
                                     self.NETRAL, "Predicted " + self.NETRAL]
        self.tp_positif = self.cm.loc["Actually " +
                                      self.POSITIVE, "Predicted " + self.POSITIVE]

        temp = self.cm.copy()
        temp.loc["Actually " + self.NEGATIVE, "Predicted " + self.NEGATIVE] = 0
        self.fn_negatif = sum(temp.loc["Actually " + self.NEGATIVE, :])
        self.fp_negatif = sum(temp.loc[:, "Predicted " + self.NEGATIVE])

        temp = self.cm.copy()
        temp.loc["Actually " + self.NETRAL, "Predicted " + self.NETRAL] = 0
        self.fn_netral = sum(temp.loc["Actually " + self.NETRAL, :])
        self.fp_netral = sum(temp.loc[:, "Predicted " + self.NETRAL])

        temp = self.cm.copy()
        temp.loc["Actually " + self.POSITIVE, "Predicted " + self.POSITIVE] = 0
        self.fn_positif = sum(temp.loc["Actually " + self.POSITIVE, :])
        self.fp_positif = sum(temp.loc[:, "Predicted " + self.POSITIVE])

        temp = self.cm.copy()
        temp = temp.drop("Actually " + self.NEGATIVE, axis = 0).drop("Predicted " + self.NEGATIVE, axis = 1)
        self.tn_negatif = sum(temp.sum())
        
        temp = self.cm.copy()
        temp = temp.drop("Actually " + self.NETRAL, axis = 0).drop("Predicted " + self.NETRAL, axis = 1)
        self.tn_netral = sum(temp.sum())
        
        temp = self.cm.copy()
        temp = temp.drop("Actually " + self.POSITIVE, axis = 0).drop("Predicted " + self.POSITIVE, axis = 1)
        self.tn_positif = sum(temp.sum())

    # def get_accuracy(self):
    #     total = float(self.tp_negatif + self.tp_netral + self.tp_positif) / len(self.actual)
    #     return total

    def get_accuracy(self):
        accuracy_each_class = []
        accuracy_each_class.append((self.tn_negatif + self.tp_negatif)/(self.tn_negatif + self.tp_negatif+ self.fn_negatif + self.fp_negatif))
        accuracy_each_class.append((self.tn_netral + self.tp_netral)/(self.tn_netral + self.tp_netral+ self.fn_netral + self.fp_netral))
        accuracy_each_class.append((self.tn_positif + self.tp_positif)/(self.tn_positif + self.tp_positif+ self.fn_positif + self.fp_positif))
        return np.mean(accuracy_each_class)

    def get_precision(self):
        precision_each_class = []
        precision_each_class.append((self.tp_negatif)/(self.tp_negatif+ self.fp_negatif))
        precision_each_class.append((self.tp_netral)/(self.tp_netral+ self.fp_netral))
        precision_each_class.append((self.tp_positif)/(self.tp_positif+ self.fp_positif))
        return np.mean(precision_each_class)

    def get_recall(self):
        recall_each_class = []
        recall_each_class.append((self.tp_negatif)/(self.tp_negatif+ self.fn_negatif))
        recall_each_class.append((self.tp_netral)/(self.tp_netral+ self.fn_netral))
        recall_each_class.append((self.tp_positif)/(self.tp_positif+ self.fn_positif))
        return np.mean(recall_each_class)

    def get_fmeasure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        fmeasure = (2 * precision * recall)/(precision+recall)
        return fmeasure

    def score(self, actual, predicted):
        self.actual = actual
        self.create_confusion_matrix(actual,predicted)
        self.find_tp_fn_fp_tn()
        return self.get_accuracy(),self.get_precision(),self.get_recall(),self.get_fmeasure()

    def get_confusion_matrix(self):
        return self.cm

# actual = ["Positif", "Positif", "Negatif", "Negatif", "Netral"]
# predicted = ["Positif", "Negatif", "Negatif", "Negatif", "Netral"]

# cm = ConfusionMatrix()
# accuracy, precision, recall, fmeasure = cm.score(actual, predicted)
# print(cm.get_confusion_matrix())

# print("Accuracy             : {}".format(accuracy))
# print("precision            : {}".format(precision))
# print("recall               : {}".format(recall))
# print("fmeasure             : {}".format(fmeasure))