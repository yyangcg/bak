#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

__author__ = 'jumingxing'


def calc_ks(data, n, prediction="prediction", label="label"):
    """
    calculate ks value
    :param data: DataFrame[prediction,label]
    :param n : int number of cut
    :param prediction: string name of prediction
    :param label: string name of label
    :return: array
    """
    data = data.sort_values(prediction, ascending=False)
    boolean = True
    while boolean:
        try:
            data[prediction] = pd.Series(pd.qcut(data[prediction], n, labels=np.arange(n).astype(str)))
        except ValueError:
            boolean = True
            n -= 1
        else:
            boolean = False
    count = data.groupby(prediction)[label].agg({'bad': np.count_nonzero, 'obs': np.size})
    count["good"] = count["obs"] - count["bad"]
    t_bad = np.sum(count["bad"])
    t_good = np.sum(count["good"])
    ks_vector = np.abs(np.cumsum(count["bad"]) / t_bad - np.cumsum(count["good"]) / t_good)
    return ks_vector


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

#
# if __name__ == '__main__':
#     data = pd.read_csv("/data/pre/part-00000", header=None)
#     data.columns = ["prediction", "label"]
#     # data = pd.read_excel("/data/workspace/pyworkspace/robot_model_bj/src/data/tmp/预测结果.xlsx")
#     # data.to_csv("/data/workspace/pyworkspace/robot_model_bj/src/data/test_ks.csv")
#     ks_dis = calc_ks(data, 20, prediction="prediction")
#     ks_cont = calc_continus_ks(data, prediction="prediction")
#     print "ks_dis is %f " % max(ks_dis)
#     print "ks_cont is %f " % max(ks_cont)
# cumsum(count["bad"]) / t_bad - np.cumsum(count["good"]) / t_good)