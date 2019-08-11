import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from IPython.display import display
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("No matplotlib module available.")

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

def eval_ks(y_true, y_pred):
    target_oos = y_pred
#     rf_results = pd.DataFrame({'prediction':target_oos[:, 1],"label":y_true})
    rf_results = pd.DataFrame({'prediction':target_oos,"label":y_true})
    ks_dis = calc_ks(rf_results, 10, prediction="prediction")
#     print(max(ks_dis))
    ks_cont = calc_continus_ks(rf_results, prediction="prediction")
#     print(max(ks_cont))

    return max(ks_dis), max(ks_cont)

def draw_decile_chart(df, bin_var, target, bins=10, chart_name=''):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from math import ceil
    
    df_temp = df.copy()
    chart_df = df_temp.loc[:, list(set([bin_var] + target))]
    chart_df_sorted = chart_df.sort_values(bin_var)
    
    bin_col = []
    bin_size = ceil(len(chart_df_sorted) / bins)
    element_counter = 0
    bin_num = bins
    
    for i in range(len(chart_df_sorted)):
        bin_col.append(bin_num)
        element_counter += 1
        
        if element_counter == bin_size:
            bin_num -= 1
            element_counter = 0
    
    bin_col_name = bin_var + '_bin'
    chart_with_bin_df = chart_df_sorted.copy()
    chart_with_bin_df[bin_col_name] = bin_col

    bin_index = chart_with_bin_df.groupby(bin_col_name).mean().index
    grouped_by_age_bin = chart_with_bin_df.groupby(bin_col_name)
    display(grouped_by_age_bin.mean())
    
    for x in target:
        plt.plot(bin_index, grouped_by_age_bin.mean()[x], linestyle='-')
    
    plt.legend()

    plt.savefig(chart_name + '_decile_chart.png')
    plt.show()

def get_performance_table(y_true, y_pred_prob, y_pred_binary):
    acc = accuracy_score(y_true, y_pred_binary)
    ks = eval_ks(y_true, y_pred_prob[:, 1])[1]
    auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    f1 = f1_score(y_true, y_pred_binary)

    print()
    print("ACC: {}".format(acc))
    print("KS: {}".format(ks))
    print("AUC: {}".format(auc))
    print("F1: {}".format(f1))
    print()

    pkey_list = ['ACC', 'KS', 'AUC', 'F1']
    pkey_val = [acc, ks, auc, f1]

    performance_df = pd.DataFrame(list(zip(pkey_list, pkey_val)))
    performance_df.columns = ['metric', 'score']
    performance_df.set_index(['metric'])
    display(performance_df)

    return performance_df

def get_feature_importance_table(X_train, y_train, random_state=13):
    clf_dt = DecisionTreeClassifier(random_state=13)
    clf_dt.fit(X_train, y_train)

    sorted_importance = sorted(list(zip(X_train.columns, clf_dt.feature_importances_)), key=lambda x: x[1], reverse=True)
    feature_importance_df = pd.DataFrame(sorted_importance)
    feature_importance_df.columns = ['feature', 'rating']
    feature_importance_df.set_index(['feature'])

    feature_importance_df.loc[:, 'rating'] = feature_importance_df.loc[:, 'rating'].apply(lambda x: str(round(100 * x, 2)) + '%')
    display(feature_importance_df)

    return feature_importance_df

def create_and_save_roc_curve_to_png(y_true, y_pred_prob):
    fpr, tpr, threshold = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 9))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_curve.png')

def fico_transform(score, lower_bound=0.001, eps=0.00001):

    score = np.array([x if x >= lower_bound else lower_bound for x in score])
    score = score + eps
    new_score = 632.0 + 62.0 * (np.log(score) - np.log(1 - score + 2 * eps))

    return new_score

def fico_inverse_transform(score, eps=0.00001):
    import numpy as np
    
    prob = ((1 + eps) * np.exp((score - 632) / 62) - eps) / (1 + np.exp((score - 632) / 62))
    prob = [1 if p > 1 else 0 if p < 0.001 else p for p in prob]

    return prob

class BinDecile:
    def __init__(self):
        self.binset = 10
        
    def bin_describe_new(self, data, elt, binset, df_new_bin_name):    
        data_new = data.sort_values(by=elt, ascending=False)
        data_new = data_new.reset_index()
        del data_new['index']
        bins = algos.quantile(data_new.index, np.linspace(0, 1, binset + 1))
        labels_list = [x for x in range(1, binset + 1)]
        tt = pd.cut(data_new.index, bins, include_lowest=True, labels=labels_list)
        data_new[df_new_bin_name] = tt
        return data_new, bins
    
    def bin_apply(self, data_new, label_column, df_new_bin_name):
        grouped_data = data_new.groupby([df_new_bin_name])
        
        return (grouped_data.agg({label_column: ['count', 'mean', 'min', 'max']}))