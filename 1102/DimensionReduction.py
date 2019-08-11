#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from scipy.optimize import minimize

import numpy as np
import pandas as pd

from com.finupgroup.utils import woe
from com.finupgroup.utils.Discretize import Discretize

__author__ = 'jumingxing'


class Instance (object):
    """
    object (label, [feature])
    """

    def __init__(self):
        self.label = None
        self.features = []


class dimension_reduction (object):
    def __init__(self, data_set):
        self.labels = []
        self.information_value = []
        self.result_matrix = None
        self.result = None
        self.size = 0
        self.maxent = None
        self.data = data_set

    @staticmethod
    def load_data(data):
        """
        载入数据,转化为待处理Instance对象
        :param data: pd.DataFrame with variables and label
        :return: list[instance object]
        """
        instanceList = []
        for idx, row in data.iterrows ():
            instance = Instance ()
            label = str (row.label)
            features = [element for element in row.drop ("label")]
            instance.label = label
            instance.features = features
            instanceList.append (instance)
        return instanceList

    @staticmethod
    def cal_new_var(X, weight, bins):
        """
        根据weight进行新变量的计算组合
        :param X: pd.DataFrame with variables
        :param weight: list , weight to transform
        :return: new variable with woe transform
        """
        result = []
        for index, row in X.iterrows ():
            result.append (sum (map (lambda (a, b): a * b, zip (row, weight))))
        result = pd.DataFrame (result)
        discretization = Discretize ().fit (x_train=result, bins=bins, labels=None, isToWrite=False)
        result = discretization.transform (result)
        return result[0]

    def entropy(self, instances):
        """
        计算熵值
        :param instances: 包含(features, label)键值对的Instance对象
        :return: dimension_reduction对象
        """
        feature_count = defaultdict (int)
        feature_label_pair = defaultdict (int)
        for instance in instances:
            label = instance.label
            if label not in self.labels:
                self.labels.append (label)
            for feature in instance.features:
                feature_count[feature] += 1
                feature_label_pair[(label, feature)] += 1
        self.size = len (instances)
        maxent = 0.0
        for instance in instances:
            label = instance.label
            for feature in instance.features:
                marginal_probability = float (feature_count[feature]) / self.size
                conditional_probability = float (feature_label_pair[(label, feature)]) / feature_count[feature]
                maxent += -marginal_probability * conditional_probability * np.log (conditional_probability)
        self.maxent = maxent
        return self

    def objective(self, weight):
        """
        搜索最优weight的目标函数
        :param weight: 待计算weight, list, weight to transform
        :return: 返回 self.maxent
        """
        data_var = self.data.drop ("label", 1)
        label = self.data.label
        data_temp = pd.DataFrame ({"new_var": self.cal_new_var (data_var, weight, 10), "label": label})
        instances = self.load_data (data_temp)
        self.entropy (instances)
        return self.maxent


if __name__ == '__main__':
    data = pd.read_csv ("/data/workspace/pyworkspace/robot_model_bj/src/data/TEMP.csv")
    colnames = pd.read_excel ("/data/workspace/pyworkspace/robot_model_bj/src/data/test_data_colname.xlsx")
    test_data = data[colnames["col_name_1"]]
    test_data = test_data.fillna (0)
    test_data.to_csv ("/data/workspace/pyworkspace/test_data.csv", index=False)
    dim = dimension_reduction (data_set=test_data)
    b = (0, 1)
    bnds = (
        b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b,
        b, b,
        b)
    weight0 = np.random.random (40)
    weight0 = weight0 / sum (weight0)
    print
    "开始最优化"
    solution = minimize (dim.objective, weight0, method='COBYLA')
    x = np.array (solution.x)
    discretize = Discretize ().fit (test_data, bins=10, labels=None, isToWrite=False)
    test_data_trans = discretize.transform (test_data.drop ("label", 1))
    information_value_2 = [woe.information_value (test_data_trans[column], test_data.label) for column in
                           test_data_trans.columns]
    data_temp_1 = pd.DataFrame (
        {"new_var": dimension_reduction.cal_new_var (test_data.drop ("label", 1), x, 10), "label": test_data.label})
    data_temp_1.to_csv ("/data/workspace/pyworkspace/new_data.csv", index=False)
    information_value_1 = woe.information_value (data_temp_1["new_var"], data_temp_1.label)
    print
    "new method iv is %f" % information_value_1
    data_result = pd.DataFrame (information_value_2, index=test_data_trans.columns).to_csv (
        "/data/workspace/pyworkspace/data_result.csv", index=True)
