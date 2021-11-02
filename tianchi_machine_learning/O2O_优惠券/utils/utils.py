from matplotlib.pyplot import fill
import pandas as pd

featurepath = './feature/'

id_col_names = ['user_id', 'coupon_id', 'date_received']
target_col_name = 'label'
id_target_cols = ['user_id', 'coupon_id', 'date_received', 'label']

def get_id_df(df):
    """返回ID列"""

    return df[id_col_names]


def get_target_df(df):
    """返回Target列"""
    return df[target_col_name]


def get_predictors_df(df):
    """返回特征列"""
    predictors = [f for f in df.columns if f not in id_target_cols]
    
    return df[predictors]


def read_featurefile_train(featurename, fill_na=False):
    """按特征名读取训练集"""
    df = pd.read_csv(featurepath + 'train_' + featurename + '.csv',
                     sep=',',
                     encoding="utf-8")
    if fill_na:
        df.fillna(0,inplace=True)
    
    return df


def read_featurefile_test(featurename, fill_na=False):
    """按特征名读取测试集"""
    df = pd.read_csv(featurepath + 'test_' + featurename + '.csv',
                     sep=',',
                     encoding="utf-8")
    if fill_na:
        df.fillna(0,inplace=True)
    
    return df


def read_data(featurename, fill_na=False):
    """按特征名读取数据"""
    traindf = read_featurefile_train(featurename, fill_na)
    testdf = read_featurefile_test(featurename, fill_na)
    
    return traindf, testdf

