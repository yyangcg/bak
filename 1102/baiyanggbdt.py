
"""ted on Fri Oct 20 12:59:58 2017

@author: yuhan
"""
def calc_continus_ks(data, prediction="prediction", label="label"):
    """

    :param data:
    :param prediction:
    :param label:
    :return:
    """
    data = data.sort_values(prediction, ascending=False)
    count = data.groupby(prediction, sort=False)[label].agg({'bad': np.count_nonzero, 'obs': np.size})
    count["good"] = count["obs"] - count["bad"]
    t_bad = np.sum(count["bad"])
    t_good = np.sum(count["good"])
    ks_vector = np.abs(np.cumsum(count["bad"]) / t_bad - np.cumsum(count["good"]) / t_good)
    return ks_vector

import pandas as pd
import numpy as np
from multiprocessing import cpu_count

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


class EnsembleGBDT (object):
    def __init__(self, n_gbdts=1, sample_size=0.1):
        self.n_gbdts = n_gbdts
        self.model = []
        self.sample_size = sample_size
        #        self.model = None
        self.n_jobs = cpu_count()

    def __repr__(self):
        return "Number of GBDT: {0}, Sample size: {1}.".format (self.n_gbdts, self.sample_size)

    @staticmethod
    def gbdt(arg):
        """
        对X,y 建立random_state = i 的GBDT
        :param data: X,y
        :return: Model
        """
        i, x, y = arg
        gbdt = GradientBoostingClassifier (learning_rate=0.1,
                                           n_estimators=100,
                                           subsample=1,
                                           min_samples_split=20,
                                           min_samples_leaf=10,
                                           max_depth=5,
                                           max_features=None,
                                           verbose=0,
                                           max_leaf_nodes=None,
                                           warm_start=False,
                                           random_state=i).fit (X=x, y=y)
        return gbdt

    def fit(self, train_set_x, train_set_y):
        """
        split training data,fit gbdt 
        sample: X,y
        sample size: self.sample_size  
        :param data: train_set_x, train_set_y
        :return: Model List
        """
        yyhgbdt = [(j, XY[:][0], XY[:][2]) for (XY, j) in
                   [(train_test_split(train_set_x, train_set_y, test_size=1 - self.sample_size, random_state=i), i) for i in
                    range (self.n_gbdts)]]
        from parallelComputing import myPool
        self.model = myPool(n_jobs=self.n_jobs, arg=yyhgbdt, function=self.gbdt)

    def predict_proba(self, test_set_x):
        """     
        预测Test的label为1 的概率
        :param data: test_set_x
        :return: array
        """
        prediction = [ele.predict_proba (test_set_x)[:, 1] for ele in self.model]
        prediction_test = np.mean (np.array (prediction), axis=0)
        return prediction_test


if __name__ == '__main__':
    Em = EnsembleGBDT(n_gbdts=3)
    data_train = pd.read_csv ('D:\\competition\\06_train.csv')
    data_test = pd.read_csv ('D:\\competition\\07_test_label.csv')
    y_test = data_test['label']
    y_train = data_train['label']
    X_train = pd.read_csv ('D:\\competition\\testing\\06_train_processed.csv')
    X_test = pd.read_csv ('D:\\competition\\testing\\07_test_processed.csv')

    import time
    start_time = time.time ()
    Em.fit(X_train, y_train)
    print ("Training {0} seconds.".format (time.time () - start_time))
    pre = Em.predict_proba(X_test)
    Data = pd.DataFrame()
    Data['label'] = y_test
    Data['prediction'] = pre
    Data = Data.sort_values ('prediction', ascending=False)
    ks_y = calc_continus_ks (Data)
    print(max (ks_y))
