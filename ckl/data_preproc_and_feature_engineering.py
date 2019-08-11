import numpy as np
import pandas as pd

# def drop_plus_date_rows(df):
#     condition_no_zero = ~(df.plus_date == 0)
#     condition_no_zero_no_eightup = ~(df.plus_date == 0) & ~(df.plus_date > 7)

#     return df.loc[condition_no_zero_no_eightup, :]

def drop_plus_date_rows(df):
    conditions = df.plus_date == 0
    return df.loc[~conditions, :]


def get_data(training_filename, testing_filename):
    print("Loading data...")
    data_train = pd.read_csv(training_filename)
    data_test = pd.read_csv(testing_filename)
    print("Finished loading data.")

    return data_train, data_test


def get_labels(data_train, data_test=None):
    data_train.loc[data_train.plus_date > 7, 'isinvested'] = 0

    if data_test is not None:
        data_test.loc[data_test.plus_date > 7, 'isinvested'] = 0
    
        return data_train.isinvested, data_test.isinvested
    else:
        return data_train.isinvested


def combine_data(data_train, data_test):
    data_all = data_train.append(data_test, ignore_index=True)
    return data_all


def drop_columns(df):
    drop_useless_cols = ['user_id', 'mobile', 'pno']
    drop_might_be_useful_cols = ['channel_id', 'create_date_id', 'create_time', 'areacode', 'devicetype']
    drop_future_cols = ['date_id', 'invest_time', 'active_num', 'lastday_before_invest', 'lianxuday_last', 'rate']
    drop_alter_label = ['plus_date']
    drop_call_cols = ['callstatus', 'call_effec', 'gap_calltoinvest', 'last_calltime',
           'call_times', 'isactive_aftercall']
    drop_other_cols = ['etl_time']
    # drop_other_cols = []

    drop_total = list(set(drop_useless_cols + drop_might_be_useful_cols + drop_future_cols + drop_alter_label + drop_call_cols + drop_other_cols))
#     print(len(drop_useless_cols) + len(drop_might_be_useful_cols) + len(drop_future_cols), len(drop_alter_label), len(drop_call_cols), len(drop_other_cols), len(drop_total))
    df_modified = df.drop(drop_total, axis=1, errors='ignore')
#     print("{} cols dropped.".format(len(cols_to_drop_tuple)))

    # Drop the stay-series, to be added back
    df_modified = df_modified.iloc[:, :10]
    
    return df_modified


def speedy_process(df, cat_colnames, fix_colnames=[]):
    # Process numeric features
    for colname in set(df.columns).difference(set(cat_colnames)).difference(set(fix_colnames)):
        # Create NA indicator
        df[colname + '_na'] = df[colname].isnull().astype(int)
        # Fill NA using median
        df[colname] = df[colname].fillna(df[colname].median())
        # Normalization
        df[colname] = (df[colname] - df[colname].min()) / (df[colname].max() - df[colname].min())
        
    # Process categorical features
    for cat_colname in cat_colnames:
        one_hot_df = pd.get_dummies(df[cat_colname], prefix=cat_colname, drop_first=True, dummy_na=True)
        df = df.join(one_hot_df)
        df.drop([cat_colname], axis=1, inplace=True)
    
    return df


def feature_transform(data_all):
    feature_all = data_all.drop('isinvested', axis=1)
    feature_all = drop_columns(feature_all)

    feature_all = speedy_process(feature_all , cat_colnames=['sex_id', 'client_type_id', 
                                                        'isrecharged', 'rechargestatus', 'isinvited'])

    return feature_all


def drop_stay_cols(df):
    df = df.drop(['staytime_recharge', 'staynum_recharge', 'staytime_p3', 'staynum_p3', 'staytime_p6', 'staynum_p6', 'staytime_p12', 'staynum_p12', 'staytime_ph', 'staynum_ph', 'staytime_p1', 'staynum_p1', 'staytime_buy', 'staynum_buy', 'staytime_asset', 'staynum_asset', 'staytime_redbag', 'staynum_redbag', 'staytime_hudong', 'staynum_hudong', 'staynum_banner', 'staynum_icon1', 'staynum_icon2', 'staynum_icon3'], axis=1)
    
    return df


def drop_columns_x_and_s(df):
    drop_useless_cols = ['user_id', 'mobile', 'pno']
    drop_might_be_useful_cols = ['channel_id', 'create_date_id', 'create_time', 'areacode', 'devicetype']
    drop_future_cols = ['date_id', 'invest_time', 'active_num', 'lastday_before_invest', 'lianxuday_last', 'rate']
    drop_alter_label = ['plus_date']
    drop_call_cols = ['callstatus', 'call_effec', 'gap_calltoinvest', 'last_calltime',
           'call_times', 'isactive_aftercall']
    # drop_other_cols = ['etl_time']
    drop_other_cols = []
    
    drop_total = list(set(drop_useless_cols + drop_might_be_useful_cols + drop_future_cols + drop_alter_label + drop_call_cols + drop_other_cols))

    df_modified = df.drop(drop_total, axis=1)
    
    return df_modified