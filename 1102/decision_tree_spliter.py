#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:13:13 2017

@author: finup
"""

from sklearn.tree import _tree

def tree_to_code(tree, feature_names, data):
    total = len(data)
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("bad_rate is:", tree_.value[node][0,0] / (tree_.value[node][0,1] + tree_.value[node][0,0]))
            print("subgroup population is", (tree_.value[node][0,1] + tree_.value[node][0,0]) / total)
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)