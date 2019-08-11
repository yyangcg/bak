#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import random

from com.finupgroup.model.myLogisticRegression import MyLogisticRegression

__author__ = 'jumingxing'


def _choose_cluster(arg):
    """
    Choose subspace closest to the given variable
    The most similar subspace is choosen based on R^2
    Parameters
    ----------
    column: variable to be assigned
    pcas: orthogonal basis for different subspaces
    number_clusters: number of subspaces (clusters)
    :return: index number of subspace closest to variable
    """
    column, pcas, number_clusters = arg
    v1 = np.var (column)
    arg_1 = [(pcas[i], column, v1) for i in range (number_clusters)]
    choose_rule = [_find_best_rule (elem) for elem in arg_1]
    return np.argmax (choose_rule)


def _find_best_rule(arg):
    pca, column, v1 = arg
    try:
        model = LinearRegression ().fit (X=pca, y=column)
    except ValueError:
        print
        column
    coef = model.coef_
    interept = model.intercept_
    res = column - np.dot (pca, coef) - interept
    v2 = np.var (res)
    temp = -float ("inf") if v1 == 0 else (v1 - v2) / v1
    return temp


def _choose_cluster_BIC(arg):
    """
    Selects subspace closest to given variable (according to BIC)
    The most similar subspace is choosen based on BIC
    Parameters
    ----------
    column: variable to be assigned
    pcas: orthogonal basis for different subspaces
    number_clusters: number of subspaces (clusters)
    :return: index number of subspace closest to variable
    """
    column, pcas, number_clusters = arg
    arg_1 = [(pcas[i], column) for i in range (number_clusters)]
    BICs = [_find_best_BIC (elem) for elem in arg_1]
    return np.argmax (BICs)


def _find_best_BIC(arg):
    pca, column = arg
    nparams = pca.shape[1]
    n = len (column)
    model = LinearRegression ().fit (pca, column)
    coef = model.coef_
    interept = model.intercept_
    res = column - np.dot (pca, coef) - interept
    sigma_hat = np.sqrt (np.var (res))
    if sigma_hat < 1e-15:
        print
        "In function _choose_cluster_BIC: estimated value of noise in cluster is %f <1e-16." \
        " It might corrupt the result." % sigma_hat
    loglik = sum (np.log (stats.norm.cdf (res, 0, sigma_hat)))
    return 2 * loglik - nparams * np.log (n)


def _pca_new_BIC(data, k):
    """
    Computes the value of BIC-like criterion for given data set and
    number of factors. Assumes that number of variables is large
    compared to number of observations
    :param data:pd.DataFrame
    :param k: number of principal components fitted
    :return: BIC value of BIC criterion
    """
    d = data.shape[0]
    N = data.shape[1]
    m = d * k - k * (k + 1) / 2
    lamb = np.linalg.eigvals (np.cov (data.T))
    v = np.sum (lamb[range (k, d, 1)]) / (d - k)
    t0 = -N * d / 2 * np.log (2 * np.pi)
    t1 = -N / 2 * np.sum (np.log (lamb[range (k)]))
    t2 = -N * (d - k) / 2 * np.log (v)
    t3 = -N * d / 2
    pen = -(m + d + k + 1) / 2 * np.log (N)
    return t0 + t1 + t2 + t3 + pen


def variable_cluster(X, number_clusters=10, stop_criterion=1, max_iter=100, max_subspace_dim=4,
                     initial_segmentation=None, estimate_dimension=False):
    """
    Performs k-means based subspace clustering. Center of each cluster is some number of principal components.
    Similarity measure is R^2 coefficient
    :param X: a pd.DataFrame with only continuous variables
    :param number_clusters: an integer, number of clusters to be used
    :param stop_criterion: an integer indicating how many changes in partitions triggers stopping the algorithm
    :param max_iter: an integer, maximum number of iterations of k-means
    :param max_subspace_dim: an integer, maximum dimension of subspaces
    :param initial_segmentation: a list, initial segmentation of variables to clusters
    :param estimate_dimension: a boolean, if TRUE subspaces dimensions are estimated, else value set by default
    :return: segmentation : a list containing the partition of the variables
             pcas : a list of matrices, basis vectors for each cluster (subspace)
    """
    np.random.seed (521)
    num_vars = X.shape[1]
    num_row = X.shape[0]
    pcas = []
    if initial_segmentation is None:
        los = random.sample (range (num_vars), number_clusters)
        pcas = [pd.DataFrame (X[X.columns[los[i]]]) for i in range (len (los))]
        arg = [(X[X.columns[i]], pcas, number_clusters) for i in range (num_vars)]
        segmentation = [_choose_cluster (elem) for elem in arg]
    else:
        segmentation = initial_segmentation
    iteration = 0
    while iteration < max_iter:
        # print "第 %d 次迭代" % iteration
        for i in range (number_clusters):
            index = [j for j, x in enumerate (segmentation) if x == i]
            sub_dim = len (index)
            if sub_dim > max_subspace_dim:
                if estimate_dimension:
                    arg_2 = [(X[X.columns[index]], k) for k in
                             range (np.minimum (np.floor (np.sqrt (sub_dim)), max_subspace_dim))]
                    cut_set = [_pca_new_BIC (elem) for elem in arg_2]
                    cut = np.argmax (cut_set)
                else:
                    cut = max_subspace_dim
                pcas[i] = pd.DataFrame (PCA (n_components=cut).fit_transform (X[X.columns[index]]))
            else:
                dim = np.maximum (1, np.int (np.sqrt (sub_dim)))
                pcas[i] = pd.DataFrame (np.random.randn (num_row, dim))
        if estimate_dimension:
            arg_1 = [(X[X.columns[m]], pcas, number_clusters) for m in range (num_vars)]
            new_segmentation = [_choose_cluster_BIC (elem) for elem in arg_1]
        else:
            arg = [(X[X.columns[n]], pcas, number_clusters) for n in range (num_vars)]
            new_segmentation = [_choose_cluster (elem) for elem in arg]
        if np.count_nonzero (np.array (new_segmentation) - np.array (segmentation)) < stop_criterion:
            break
        segmentation = new_segmentation
        iteration += 1
    cluster_id = list (set (segmentation))
    segmentation_result = pd.DataFrame ({"cluster": segmentation, "variable": X.columns})
    segmentation = [list (segmentation_result[segmentation_result.cluster == k]["variable"]) for k in cluster_id]
    return segmentation, pcas


def middle_loop(arg):
    column, feature_selection_temp, x_train, y_train, x_test, y_test = arg
    auc_value = -float ("inf")
    coeficients = None
    variable_temp = None
    if column not in feature_selection_temp:
        variable_temp = list (np.copy (feature_selection_temp))
        variable_temp.append (column)
        lr = LogisticRegression ()
        model = lr.fit (x_train[variable_temp], y_train)
        coeficients = pd.DataFrame (model.coef_[0], index=variable_temp, columns=["coef"])
        prediction = model.predict_proba (x_test[variable_temp])[:, 1]
        label = y_test
        fpr_lr, tpr_lr, _ = roc_curve (label, prediction)
        auc_value = auc (fpr_lr, tpr_lr)
    return coeficients, auc_value, variable_temp


def feature_selection(data_set, target="label", threshold=6, data_sample_num=7, initial_temperature=100, cool_rate=0.98,
                      limit_temperature=50,
                      iteration_num=100):
    """
    模拟退火算法进行变量选择,依据AUC
    :param data_set: 数据集
    :param target:
    :param initial_temperature:
    :param data_sample_num:
    :param threshold:
    :param cool_rate:
    :param limit_temperature:
    :param iteration_num:
    :return:
    """

    np.random.seed (520)
    variables = cor_feature_selection (dataSet=data_set, data_sample_num=data_sample_num, threshold=threshold,
                                       target=target)
    # variables = list(data_set.drop(target, 1).columns)
    y = data_set[target]
    x_train, x_test, y_train, y_test = train_test_split (data_set.drop (target, 1), y, test_size=0.25)
    feature_selected = []
    auc_record = [0.5]
    iter_num = 1
    while initial_temperature >= limit_temperature and iter_num <= iteration_num:
        # print "第 %d 次迭代" % iter_num
        oktomoveon = 0
        best_index = None
        feature_selection_temp = list (np.copy (feature_selected))
        while oktomoveon == 0 and len (variables) > len (feature_selection_temp) and len (variables) > 0:
            arg = [(column, feature_selection_temp, x_train, y_train, x_test, y_test) for column in variables]
            temp = [middle_loop (elem) for elem in arg]
            temp = np.array (temp)
            coeficients_list = list (temp[:, 0])
            best_auc = list (temp[:, 1])
            best_feature_temp = list (temp[:, 2])
            best_index = np.argmax (best_auc)
            feature_selection_temp = best_feature_temp[best_index]
            negative_var = list (coeficients_list[best_index][coeficients_list[best_index]["coef"] < 0].index)
            if len (negative_var) > 0:
                kickoutwho = chi_square_selected (x_train[feature_selection_temp], y_train, negative_var)
                if kickoutwho is not None:
                    variables.remove (kickoutwho)
                    feature_selection_temp.remove (kickoutwho)
            else:
                oktomoveon = 1
            vif_matrix = x_train[feature_selection_temp].copy ()
            if vif_matrix.shape[1] > 0:
                vif_matrix["inter"] = np.ones (vif_matrix.shape[0])
                vif = [variance_inflation_factor (np.array (vif_matrix), i) for i in range (vif_matrix.shape[1])]
                vif.pop ()
                vif_table = pd.DataFrame (vif, index=feature_selection_temp, columns=["vif"])
                vif_rule_remove = set (vif_table[vif_table["vif"] > 5].index)
                variables = list (set (variables).difference (vif_rule_remove))
                feature_selection_temp = list (set (feature_selection_temp).difference (vif_rule_remove))
        auc_record_upper = auc_record[len (auc_record) - 1]
        if best_index is not None and (auc_record_upper < best_auc[best_index] or np.random.random () < np.exp (
                    (-best_auc[best_index] + auc_record_upper) / initial_temperature)):
            iter_num += 1
            auc_record.append (best_auc[best_index])
            initial_temperature *= cool_rate
            feature_selected = list (np.copy (feature_selection_temp))
        else:
            break
    return feature_selected


def chi_square_selected(X, Y, negvar):
    """

    :param X:
    :param Y:
    :param negvar:
    :return:
    """
    import warnings
    with warnings.catch_warnings ():
        warnings.filterwarnings ("ignore")
        t_values = MyLogisticRegression ().fit_method (X=X, y=Y).t_value
        result_temp = pd.DataFrame (t_values, index=X.columns, columns=["t_value"])
        result_temp_sub = result_temp.loc[negvar]
        result_temp_sub = result_temp_sub.sort_values ("t_value")
    return result_temp_sub.index[0]


def count_positive_value(data_list):
    data_list_1 = pd.Series ([0.0 if np.isnan (elem) else elem for elem in data_list])
    return sum (data_list_1 >= 0)


def cor_feature_selection(dataSet, data_sample_num, threshold, target):
    """
    根据自变量与因变量的相关系数的正负号的差值进行变量选择
    :param dataSet:
    :param data_sample_num:
    :param threshold:
    :param target:
    :return:
    """
    index = np.random.randint (data_sample_num, size=dataSet.shape[0])
    col_name = dataSet.drop (target, 1).columns
    result_set = pd.DataFrame (index=col_name)
    for i in xrange (data_sample_num):
        data_temp = dataSet[index == i]
        correlation, p_value = stats.spearmanr (a=data_temp.drop (target, 1), b=data_temp[target], axis=0)
        result_set["data_set_" + repr (i)] = correlation[:, data_temp.shape[1] - 1][:data_temp.shape[1] - 1]
    countpo = result_set.apply (count_positive_value, axis=1)
    countna = result_set.shape[1] - countpo
    diff = abs (countpo - countna)
    col_list = diff.where (lambda x: x >= threshold).dropna ().index
    return list (col_list)
