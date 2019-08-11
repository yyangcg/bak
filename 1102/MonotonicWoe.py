#!/usr/bin/env python
# -*- coding: utf-8 -*-
from com.finupgroup.utils.Discretize import Discretize
from com.finupgroup.utils.woe import Woe
import scipy.stats as stats
import numpy as np
import pandas as pd

__author__ = 'jumingxing'


class MonotonicWoe (Discretize, Woe):
    def __init__(self):
        super (MonotonicWoe, self).__init__ ()
        self.spearman_coeffient = dict ()

    @staticmethod
    def spearman(x):
        """
        计算spearman coeffient
        :param x: pd.DataFrame contains number of bad, observition, lower, upper of cut labels
        :return: abs(spearman coeffient)
        """
        bad = x["bad"]
        count = x["obs"]
        lower = x["lower"]
        count = [1 if x == 0 else x for x in count]
        bad_rate = bad / count
        try:
            cor, pval = stats.spearmanr (a=bad_rate, b=lower, axis=0)
        except ValueError:
            print
            "bad_rate is %f" % bad_rate
            print
            "bad is %s" % repr (bad)
            print
            "lower is %s" % repr (lower)
        return abs (cor)

    def group_transform(self, column_data, target, labels):
        """
        对cut labels 进行合并迭代,寻找使得abs(spearman coeffient)最大的labels组合
        :param column_data: 原始数据列向量, pd.Series or np.array
        :param target: list or np.array or pd.Series
        :param labels: the cut labels of discretize model
        :return: new cut label and the best spearman coeffient
        """
        data = pd.DataFrame ({"var": column_data, "label": list (target)})
        data_matrix_1 = data.groupby ("var")["label"].agg (
            {'bad': np.count_nonzero, 'obs': np.size})
        if "bin_na" in data_matrix_1.index:
            data_matrix = data_matrix_1.drop ("bin_na").copy ()
        else:
            data_matrix = data_matrix_1.copy ()
        data_matrix["lower"] = np.append (labels[0] - 1, labels[1:-1])
        data_matrix["upper"] = np.append (labels[1:-1], labels[-1:][0] + 1)
        coef = self.spearman (data_matrix)
        new_label = list (np.copy (labels))
        data_matrix_best = data_matrix.copy ()
        while np.float64 (coef) < 0.9999:
            coef_list = []
            label_list = []
            data_matrix_list = []
            for i in range (len (new_label) - 2):
                label_temp = list (np.copy (new_label))
                label_temp.remove (new_label[i + 1])
                data_matrix_temp = data_matrix_best.copy ()
                series_index = data_matrix_temp.index
                data_matrix_temp["bad"].loc[series_index[i + 1]] += data_matrix_temp["bad"].loc[series_index[i]]
                data_matrix_temp["obs"].loc[series_index[i + 1]] += data_matrix_temp["obs"].loc[series_index[i]]
                data_matrix_temp["lower"].loc[series_index[i + 1]] = data_matrix_temp["lower"].loc[series_index[i]]
                data_matrix_temp = data_matrix_temp.drop (data_matrix_temp.index[i])
                data_matrix_list.append (data_matrix_temp)
                coef_list.append (self.spearman (data_matrix_temp))
                label_list.append (label_temp)
            index = np.argmax (coef_list)
            coef = coef_list[index]
            new_label = label_list[index]
            data_matrix_best = data_matrix_list[index]
        data_matrix_best.index = np.array (["bin_" + str (i) for i in range (data_matrix_best.shape[0])])
        if "bin_na" not in data_matrix_1.index:
            data_matrix_best.loc["bin_na"] = [0, 0, np.NaN, np.NaN]
        else:
            data_matrix_best.loc["bin_na"] = data_matrix_1.loc["bin_na"]
        data_matrix_best["good"] = data_matrix_best["obs"] - data_matrix_best["bad"]
        return new_label, coef, data_matrix_best

    def fit_method(self, train_set_x, train_set_y, discretize=None):
        """
        经过monotonic变换后的woe_fit_model
        :param train_set_x: pd.DataFrame with variable and label , continus variable
        :param train_set_y: string, the name of target
        :param discretize: discretize 对象
        :return: 返回Monotonic_woe对象
        """
        dict_dock_temp = dict ()
        dictionary = dict ()
        for variable in train_set_x.columns:
            if variable in discretize.dictionary.keys ():
                label = discretize.dictionary[variable]
            else:
                continue
            new_label, coef, data_matrix = self.group_transform (column_data=train_set_x[variable],
                                                                 target=train_set_y, labels=label)
            self.spearman_coeffient.update ({variable: coef})
            dictionary.update ({variable: new_label})
            dict_dock_temp.update ({variable: super (MonotonicWoe, self).calc_woe (data_matrix)})
        self.dict_dock = dict_dock_temp
        self.dictionary = dictionary
        return self

    def transform(self, test_set):
        """
        monotonic_woe transform
        :param test_set: 训练集 without "label" column
        :return: 返回monotonic变换后的woe_transform
        """
        result = Woe.transform (self, test_set)
        return result


if __name__ == '__main__':
    data = pd.read_csv ("/data/workspace/pyworkspace/robot_model_bj/src/data/test_data.csv", header=0)
    monotonic_model = MonotonicWoe ()
    discretization = Discretize ().fit (data.drop ("label", 1), bins=20)
    train_data_x = discretization.transform (data.drop ("label", 1))
    train_data_y = data["label"]
    model = monotonic_model.fit_method (train_set_x=train_data_x, train_set_y=train_data_y,
                                        discretize=discretization)
    spearman_coeffient = monotonic_model.spearman_coeffient
    print
    spearman_coeffient
    monotonic_data = monotonic_model.transform (test_set=data.drop ("label", 1))
    # woe_data.to_csv("/data/workspace/pyworkspace/robot_model_bj/src/data/test_data_result.csv")
    print
    monotonic_model.dict_dock
