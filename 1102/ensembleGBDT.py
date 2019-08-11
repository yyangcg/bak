
"""ted on Fri Oct 20 12:59:58 2017

@author: yuhan
"""
from __future__ import print_function


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


def compute_ks_h2o(df_pred, df_obs, pred_column_name='p1', target_column_name='isinvested'):
    print (df_pred[pred_column_name].as_data_frame (use_pandas=True, header=True).head ())
    print (df_obs[target_column_name].as_data_frame (use_pandas=True, header=True).head ())
    pred = df_pred[pred_column_name].as_data_frame (use_pandas=True, header=True)
    #    predproba = df_pred[pred_column_name].as_data_frame(use_pandas=True, header=True)
    target = df_obs[target_column_name].as_data_frame (use_pandas=True, header=True)
    # target_oot2 = target_oot2.reset_index()
    # del target_oot2['index']
    result = pd.concat ([pred, target], axis=1)
    # result.to_csv('model_h2o_v1.csv',encoding='utf-8')
    result.columns = ['prediction', 'label']
    ks_dis = calc_ks (result, 10, prediction="prediction")
    print ('discrete ks is: ', max (ks_dis))
    ks_cont = calc_continus_ks (result, prediction="prediction")
    print ('continuous ks is: ', max (ks_cont))

    return result, ks_dis, ks_cont

import pandas as pd
import numpy as np
import datetime
import time
from multiprocessing import cpu_count
from sklearn import ensemble, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch

h2o.init (max_mem_size = "4G", nthreads = -1, port=4869)

class EnsembleGBDT (object):
    def __init__(self, n_gbdts=100, sample_size=0.7):
        self.n_gbdts = n_gbdts
        self.model = []
        self.sample_size = sample_size
        # self.n_estimators = n_estimators
        # self.max_depth = max_depth
        self.n_jobs = cpu_count()
    def __repr__(self):
        return "Number of GBDT: {0}, Sample size: {1}.".format (self.n_gbdts, self.sample_size)

    @staticmethod
    def gbdt(arg):
        """
        for X,y  build GBDT(random_state = i )
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
        # self.model = [self.gbdt(elem) for elem in yyhgbdt]
        return self.model


    def fit_h2o(self,train_xy, label_train):

        train = h2o.H2OFrame (train_xy)
        self.x = train.columns
        self.y = label_train
        self.x = self.x.remove (self.y)
        train[self.y] = train[self.y].asfactor ()
        my_gbm = [H2OGradientBoostingEstimator (distribution="bernoulli",
                                                ntrees=10,
                                                max_depth=5,
                                                min_rows=2,
                                                learn_rate=0.2,
                                                nfolds=5,
                                                fold_assignment="Random",
                                                keep_cross_validation_predictions=True,
                                                seed=i) for i in range (self.n_gbdts)]
        model_gbm = [elem.train (x=self.x, y=self.y, training_frame=train) for elem in my_gbm]

        model = [elem.model_id for elem in my_gbm]
        ensemble_h2o = H2OStackedEnsembleEstimator (model_id="my_ensemble_gbm_ensemble",
                                                base_models=model)
        ensemble_h2o.train (x=self.x, y=self.y, training_frame=train)

        return ensemble_h2o

    def predict_proba_h2o(self,test_set_x):
        test = h2o.H2OFrame (test_set_x)
        test[self.y] = test[self.y].asfactor ()
        prediction = self.ensemble_h2o.predict(test)
        return prediction

    def predict_proba(self, test_set_x):
        """     
        predict_prob Test label:1 
        :param data: test_set_x
        :return: array
        """
        prediction = [ele.predict_proba (test_set_x)[:, 1] for ele in self.model]
        prediction_test = np.mean (np.array (prediction), axis=0)
        return prediction_test


if __name__ == '__main__':
    # sklearn
    # print ("Process running at {}".format (datetime.datetime.now ()))
    #
    # Em = EnsembleGBDT(n_gbdts=200)
    # data_train = pd.read_csv ('train.csv')
    # data_test = pd.read_csv ('oot.csv')
    # y_test = data_test['label']
    # y_train = data_train['label']
    # X_train = data_train.drop(['label'],axis=1)
    # X_test = data_test.drop(['label'],axis=1)
    #
    # # build model
    # start_time = time.time ()
    # Em.fit(X_train, y_train)
    # print ("Training {0} seconds.".format (time.time () - start_time))
    # # predict
    # pred = Em.predict_proba(X_test)
    # # ks
    # Data = pd.DataFrame()
    # Data['label'] = y_test
    # Data['prediction'] = pred
    # Data = Data.sort_values ('prediction', ascending=False)
    # ks_y = calc_continus_ks (Data)
    # print(max (ks_y))
    # # auc
    # target = y_test
    # prediction = Data['prediction']
    # auc = metrics.roc_auc_score (target, prediction)
    # print ("AUC:", auc)

    #h2o
    print ("Process running at {}".format (datetime.datetime.now ()))

    Em = EnsembleGBDT(n_gbdts=2)
    data_train = pd.read_csv ('train.csv')
    data_test = pd.read_csv ('oot.csv')
    y_test = data_test['label']
    y_train = data_train['label']
    X_train = data_train.drop(['label'],axis=1)
    X_test = data_test.drop(['label'],axis=1)

    # build model
    start_time = time.time ()
    Em.fit_h2o(data_train, label_train='label')
    print ("Training {0} seconds.".format (time.time () - start_time))
    # predict
    df_pred = Em.predict_proba_h2o(data_test)
    # ks
    pred = df_pred['p1'].as_data_frame (use_pandas=True, header=True)
    Data = pd.DataFrame ()
    Data['label'] = y_test
    Data['prediction'] = pred
    Data = Data.sort_values ('prediction', ascending=False)
    ks_y = calc_continus_ks (Data)
    print (max (ks_y))
    # auc
    target = Data['label']
    prediction = Data['prediction']
    auc = metrics.roc_auc_score (target, prediction)
    print ("AUC:", auc)
