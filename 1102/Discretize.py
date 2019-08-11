#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import sys
from datetime import datetime
from com.finupgroup.utils.woe import Woe

__author__ = 'jumingxing'


class Discretize (object):
    def __init__(self):
        self.bins = 10
        self.labels = None
        self.isToWrite = None
        self.dictionary = dict ()
        self.n_jobs = cpu_count ()

    def discretization(self, column, bins=10, labels=None, retbins=True, isToWrite=False, filename=None):
        """
        对连续变量列进行离散化,采用qcut方法
        :param column: 待离散化的连续变量列
        :param bins: 离散化的切分个数
        :param labels: 离散化后的label
        :param retbins: 是否返回离散化结果
        :param isToWrite: 是否将离散化结果bins和labels写出
        :param filename: 写出离散化结果bins和labels的路径
        :return: Discretize对象
        """
        column_set = set (column)
        column_set_nan = [elem.decode ("gbk").encode ("utf-8") if isinstance (elem, str) else elem for elem in
                          column_set]
        column_set_nan = [elem for elem in column_set_nan if str (elem) != "nan"]
        if len (column_set_nan) <= 5:
            column_set_nan.sort ()
            for elem in column_set_nan:
                if isinstance (elem, unicode):
                    raise Exception ("{} is not numeric".format (elem))
            self.labels = list (np.append (column_set_nan, float ("inf")))
        else:
            series_index = column.index
            for i in range (len (column)):
                try:
                    if isinstance (list (column)[i], str) or isinstance (list (column)[i], unicode):
                        column.loc[series_index[i]] = 0
                except ValueError:
                    print
                    column.loc[series_index[i]]
            bins = int (np.minimum (column.unique ().size, bins))
            boolean = True
            while boolean:
                try:
                    cuts, bin_s = pd.qcut (np.array (column), bins, labels=labels, retbins=retbins)
                except ValueError:
                    boolean = True
                    bins -= 1
                else:
                    boolean = False
            self.labels = bin_s
            result = pd.DataFrame ({"bins": bin_s, "labels": np.arange (len (bin_s)).astype (str)})
            if isToWrite:
                result.to_csv (filename, index=False)
        return self

    def data_transform(self, column):
        """
        对连续变量列进行离散化transform
        :param column: column of a pd.DataFrame 连续变量
        :return:返回离散化transform的结果
        """
        bin_s = np.append ((-float ("inf"),), self.labels[1:-1])
        labels = np.array (["bin_" + str (i) for i in xrange (len (bin_s))])
        result_temp = pd.cut (column, bins=np.append (bin_s, (float ("inf"),)), labels=labels)
        result = [elem if str (elem) != "nan" else "bin_na" for elem in result_temp]
        return pd.Series (pd.Categorical (result, categories=np.append (labels, "bin_na"), ordered=True),
                          index=column.index)

    def fit(self, x_train, bins=10, labels=None, isToWrite=False, filename=None):
        """
        对连续变量数据集进行离散化train过程
        :param x_train: pd.Dataframe 待离散化train的连续变量的dataFrame
        :param bins: 待离散化切分bin的个数
        :param labels: 离散化的labels,None,[]
        :param isToWrite: 是否将离散化结果bins和labels写出
        :param filename: 写出离散化结果bins和labels的路径
        :return: 返回Discretize对象
        """
        arg = [(self, x_train, name, bins, labels, isToWrite, filename) for name in x_train.columns]
        discretize_obj = [_get_discretize_object (elem) for elem in arg]
        for elem in discretize_obj:
            if elem is not None:
                self.dictionary.update (elem)
            else:
                continue
        return self

    def transform(self, x_test):
        """
        对待离散化的连续变量数据集进行离散化transform
        :param x_test: pd.DataFrame 待离散化连续变量数据集
        :return: pd.DataFrame 离散化结果
        """
        out = pd.DataFrame ()
        keys = self.dictionary.keys ()
        col_names = x_test.columns
        intersect_col = list (set (col_names).intersection (set (keys)))
        arg = [(self, x_test, name) for name in intersect_col]
        result_temp = [_get_columns_transform (elem) for elem in arg]
        if len (result_temp) == 0:
            return None
        else:
            for i in range (len (result_temp)):
                out[intersect_col[i]] = result_temp[i]
            return out


def _get_columns_transform(arg):
    self, x_test, name = arg
    self.labels = self.dictionary.get (name)
    if self.labels is None:
        raise ValueError ("column %s has no labels" % name)
    result_temp = self.data_transform (x_test[name])
    return result_temp


def _get_discretize_object(arg):
    self, x_train, name, bins, labels, isToWrite, filename = arg
    discretize_obj = None
    if x_train[name].unique ().size == 1:
        return None
    try:
        discretize_obj = self.discretization (column=x_train[name], bins=bins, labels=labels, isToWrite=isToWrite,
                                              filename=filename)
    except Exception:
        print
        "当前类名称是 %s, 方法名称是 %s :" % (__file__, sys._getframe ().f_code.co_name)
        print
        name
    return {name: discretize_obj.labels}
