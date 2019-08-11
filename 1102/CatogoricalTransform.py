#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

__author__ = 'jumingxing'


class CategoricalTransform:
    def __init__(self):
        self.categorical_var = []
        self.dictionary = dict ()

    def fit(self, data, Y):
        """
        分类变量进行连续化变换 bad_rate替换
        :param data: pd.DataFrame columns variable
        :param Y: pd.Series label
        :return: CategoricalTransform对象
        """
        for elem in data.columns:
            if data[elem].unique ().size > 1:
                if isinstance (data[elem].unique ()[1], unicode) or isinstance (data[elem].unique ()[1],
                                                                                str) or isinstance (
                        data[elem].unique ()[0], unicode):
                    self.categorical_var.append (elem)
            else:
                if isinstance (data[elem].unique ()[0], unicode) or isinstance (data[elem].unique ()[0], str):
                    self.categorical_var.append (elem)
        if len (self.categorical_var) > 0:
            data_cate = data.loc[:, self.categorical_var]
            nan_rate = data_cate.apply (self._nan_transform, axis=0)
            data_cate["label"] = Y
            for elem in self.categorical_var:
                temp = data_cate.groupby (elem)["label"].mean ().sort_values (ascending=False)
                temp.set_value (np.nan, value=nan_rate[elem])
                self.dictionary.update ({elem: temp})  # order
        return self

    @staticmethod
    def _nan_transform(x):
        return float (sum ([str (elem) == "nan" for elem in x])) / len (x)

    def transform(self, X):
        """
        对离散变量连续化做transform变换
        :param X: pd.DataFrame 待transform离散变量数据集
        :return: pd.DataFrame 连续化transform结果
        """
        keys = self.dictionary.keys ()
        for elem in X.columns:
            if elem not in keys:
                continue
            X[elem] = X[elem].apply (self._element_transform, col=elem)
        return X

    def _element_transform(self, element, col):
        if element in self.dictionary[col].index:
            return self.dictionary[col].loc[element]
        elif isinstance (element, str):
            return 0.0
