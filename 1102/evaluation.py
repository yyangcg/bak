import matplotlib.pyplot as plt
from sklearn import ensemble,metrics
import pandas as pd
import numpy as np
class Evaluation(object):
    def __init__(self, n = 10):
        self.n = n
        self.results = None


    def calc_ks(self, data, n, prediction="prediction", label="label"):
        """
        calculate ks value
        :param data: DataFrame[prediction,label]
        :param n : int number of cut
        :param prediction: string name of prediction
        :param label: string name of label
        :return: array
        """
        data = data.sort_values (prediction, ascending=False)
        boolean = True
        while boolean:
            try:
                data[prediction] = pd.Series (pd.qcut (data[prediction], n, labels=np.arange (n).astype (str)))
            except ValueError:
                boolean = True
                n -= 1
            else:
                boolean = False
        count = data.groupby (prediction)[label].agg ({'bad': np.count_nonzero, 'obs': np.size})
        count["good"] = count["obs"] - count["bad"]
        t_bad = np.sum (count["bad"])
        t_good = np.sum (count["good"])
        ks_vector = np.abs (np.cumsum (count["bad"]) / t_bad - np.cumsum (count["good"]) / t_good)
        return ks_vector

    def calc_continus_ks(self, data, prediction="prediction", label="label"):
        """

        :param data:
        :param prediction:
        :param label:
        :return:
        """
        data = data.sort_values (prediction, ascending=False)
        count = data.groupby (prediction, sort=False)[label].agg ({'bad': np.count_nonzero, 'obs': np.size})
        count["good"] = count["obs"] - count["bad"]
        t_bad = np.sum (count["bad"])
        t_good = np.sum (count["good"])
        ks_vector = np.abs (np.cumsum (count["bad"]) / t_bad - np.cumsum (count["good"]) / t_good)
        population = np.sum (count["obs"])
        bad_number = np.sum (count["bad"])
        bad_rate = bad_number / population
        return ks_vector, population, bad_number, bad_rate


    def eval(self, data_label, data_proba, label, prediction, threshold = 0.5):
        probas = data_proba[prediction]
        target = data_label[label]
        data = pd.DataFrame({'prediction': probas,'label':target})
        data['preds'] = 0
        data.loc[(data['prediction'] >= threshold), 'preds'] = 1
        data = data.sort_values (by='prediction', ascending=False)
        probas = data['prediction']
        target = data['label']
        preds = data['preds']
        # ks_dis = self.calc_ks (data, self.n)
        ks_cont, population, bad_number, bad_rate = self.calc_continus_ks (data)
        ACC = metrics.accuracy_score(target,preds)
        f1_score = metrics.f1_score (target, preds)
        cm = metrics.confusion_matrix(target,preds)
        Precision = cm[1, 1] / (cm[1, 1] + cm[0, 1] * 1.1)
        Recall = cm[1, 1] / (cm[1, 1] + cm[1, 0] * 1.1)
        auc = metrics.roc_auc_score (target, probas)
        gini = 2*auc - 1
        print ("Population:", population)
        print ("Number of bad:", bad_number)
        print ("Bad rate:", bad_rate)
        # print ("ks_dis:", max (ks_dis))
        print("ks_cont:", max(ks_cont))
        print ("AUC:", auc)
        print ("Gini:", gini)
        print("ACCURACY:", ACC)
        print("F1 Score:", f1_score)
        print( "Precision:", Precision)
        print( "Recall:", Recall)
        self.results = pd.DataFrame ({'Population': [population], 'Number of bad': [bad_number], 'Bad rate':[bad_rate],
                                      'KS': [max(ks_cont)], 'AUC':[auc], 'Gini':[gini], 'ACCURACY':[ACC],
                                      'F1 Score':[f1_score], 'Precision': [Precision], 'Recall': [Recall]})
        return self.results


    def save_csv(self,filename):
        self.results.to_csv(filename, encoding='utf-8',index=False,index_label=True)

if __name__ == '__main__':

    data_label = pd.read_csv ('huoke_yuhan07.csv')
    data_proba = pd.read_csv ('huoke_yuhan07.csv')
    Eval = Evaluation()
    Eval.eval(data_label, data_proba, label='label', prediction='prediction')
    Eval.save_csv('result_evaluation.csv')


