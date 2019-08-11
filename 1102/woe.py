#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd

__author__ = 'jumingxing'


def information_value(column, Y):
    """

    :param column: pd.Series or list or np.array after discretization
    :param Y: label
    :return: information value
    """
    data = pd.DataFrame ({"var": column, "label": Y})
    woe_data = data.groupby ("var")["label"].agg (
        {'bad': np.count_nonzero, 'obs': np.size})
    woe_data['good'] = woe_data['obs'] - woe_data['bad']
    if 'bad' not in woe_data.columns:
        raise ValueError ("data columns don't has 'bad' column")
    if 'good' not in woe_data.columns:
        raise ValueError ("data columns don't has 'good' column")
    t_bad = np.maximum (woe_data['bad'].sum (), 0.5)
    t_good = np.maximum (woe_data['good'].sum (), 0.5)
    woe_data["woe"] = woe_data.apply (_bucket_woe, axis=1) + np.log (float (t_good) / float (t_bad))
    iv = (woe_data['bad'] / t_bad - woe_data['good'] / t_good) * woe_data['woe']
    return iv.sum ()


class Woe (object):
    def __init__(self):
        self.dict_dock = None
        self.n_jobs = cpu_count ()

    def _woe_elem_transform(self, col_name, elem):
        matrix = self.dict_dock.get (col_name)
        if elem in list (matrix.index):
            return matrix.loc[elem, "woe"]
        else:
            return 0.0

    @staticmethod
    def calc_woe(data):
        """

        :param data: DataFrame(Var:float,bad:int,good:int)
        :return: weight of evidence
        """
        if 'bad' not in data.columns:
            raise ValueError ("data columns don't has 'bad' column")
        if 'good' not in data.columns:
            raise ValueError ("data columns don't has 'good' column")
        t_bad = np.maximum (data['bad'].sum (), 0.5)
        t_good = np.maximum (data['good'].sum (), 0.5)
        data['woe'] = data.apply (_bucket_woe, axis=1) + np.log (float (t_good) / float (t_bad))
        woe_dock = pd.DataFrame (data["woe"])
        return woe_dock

    def woe_transform(self, data_var, col_name):
        """

        :param data_var: pd.Series
        :param col_name: string column name
        :return:
        """
        data = [self._woe_elem_transform (col_name=col_name, elem=elem) for elem in data_var]
        return pd.Series (data)

    def fit(self, X, Y):
        """

        :param X: pd.DataFrame variable data after discretize
        :param Y: pa.Series label
        :return: woe object
        """
        dict_dock = dict ()
        data = X.copy ()
        data["label"] = np.array (Y)
        for elem in X.columns:
            woe_data = data.groupby (elem)["label"].agg (
                {'bad': np.count_nonzero, 'obs': np.size})
            woe_data['good'] = woe_data['obs'] - woe_data['bad']
            dict_dock.update ({elem: self.calc_woe (woe_data)})
        self.dict_dock = dict_dock
        return self

    def transform(self, X):
        """

        :param X: pd.DataFrame after discretize
        :return: pd.DataFrame with result of woe transform
        """
        keys = self.dict_dock.keys ()
        if len (keys) < 0:
            return pd.DataFrame (np.zeros (X.shape), columns=X.columns)
        else:
            arg = [(self, X, name) for name in X.columns if name in keys]
            result_temp = [_get_woe_transform (elem) for elem in arg]
            result = [elem for elem in result_temp if elem is not None]
            if len (result) == 0:
                return pd.DataFrame (np.zeros (X.shape), columns=X.columns)
            else:
                try:
                    return pd.DataFrame (np.transpose (result), columns=X.columns)
                except ValueError:
                    print
                    "woe_transform error"
                    print
                    result


def _get_woe_transform(arg):
    self, X, name = arg
    result_temp = pd.Series ([self._woe_elem_transform (col_name=name, elem=elem) for elem in X[name]])
    return result_temp


def _bucket_woe(x):
    t_bad = x['bad']
    t_good = x['good']
    if t_bad == 0 and t_good == 0:
        woe = 0
    else:
        t_bad = 0.1 if t_bad == 0 else t_bad
        t_good = 0.1 if t_good == 0 else t_good
        woe = np.log (float (t_bad) / float (t_good))
    return woe
