#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

__author__ = 'jumingxing'


class CategoricalTransform:
    def __init__(self):
        self.categorical_var = []
        self.dictionary = dict()

    def fit(self, data, Y):
        """
        分类变量进行连续化变换 bad_rate替换
        :param data: pd.DataFrame columns variable
        :param Y: pd.Series label
        :return: CategoricalTransform对象
        """
        self.categorical_var = [self.fitIml1(elem) for elem in data.columns]
        if len(self.categorical_var) > 0:
            data_cate = data.loc[:, self.categorical_var]
            nan_rate = data_cate.apply(self._nan_transform, axis=0)
            data_cate["label"] = Y
            arg = [(nan_rate, elem) for elem in self.categorical_var]
            self.dictionary = dict(self.fitImp2(elem) for elem in arg)
        return self

    @staticmethod
    def fitIml1(colName):
        if data[colName].unique().size > 1:
            if isinstance(data[colName].unique()[1], unicode) or isinstance(data[colName].unique()[1],
                                                                            str) or isinstance(
                    data[colName].unique()[0], unicode):
                return colName
        else:
            if isinstance(data[colName].unique()[0], unicode) or isinstance(data[colName].unique()[0], str):
                return colName

    @staticmethod
    def fitImp2(arg):
        nan_rate, colName = arg
        temp = data_cate.groupby(colName)["label"].mean().sort_values(ascending=False)
        temp.set_value(np.nan, value=nan_rate[colName])
        return [colName, temp]

    @staticmethod
    def _nan_transform(x):
        return float(sum([str(elem) == "nan" for elem in x])) / len(x)

    def transform(self, X):
        """
        对离散变量连续化做transform变换
        :param X: pd.DataFrame 待transform离散变量数据集
        :return: pd.DataFrame 连续化transform结果
        """
        keys = self.dictionary.keys()

        tempDict = dict(
            [[elem, X[elem].apply(self._element_transform, col=elem)] for elem in X.columns if elem in keys])

        for k, v in tempDict.iteritems():
            X[k] = v

        # for elem in X.columns:
        # 	if elem not in keys:
        # 		continue
        # 	X[elem] = X[elem].apply(self._element_transform, col=elem)
        return X

    def _element_transform(self, element, col):
        if element in self.dictionary[col].index:
            return self.dictionary[col].loc[element]
        elif isinstance(element, str):
            return 0.0
