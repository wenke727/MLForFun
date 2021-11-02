#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#%%

def load_data(keep_default_na=True):
    atts=['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

    off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv', keep_default_na=keep_default_na)
    off_train.columns = atts + ['date']

    off_test = pd.read_csv("../data/ccf_offline_stage1_test_revised.csv", keep_default_na=keep_default_na)
    off_test.columns = atts

    on_train = pd.read_csv("../data/ccf_online_stage1_train.csv", keep_default_na=keep_default_na)
    on_train.columns = atts + ['date']

    str_atts = ['user_id', 'merchant_id', 'coupon_id']

    for df in [off_train, on_train, off_test]:
        df[str_atts] = df[str_atts].astype(str)

    return off_train, off_test, on_train



def print_date_receive(df, desc='', null_control=True):
    if null_control:
        res = df[df['date_received'] != 'null']['date_received']
    else:
        res = df['date_received']

    print(f'{desc} date_received: {res.min()}, {res.max()}')



off_train, off_test, on_train = load_data(keep_default_na=False)


# %%

# 通过探索可以发现训练数据的用券数据是到6月30日，而领券日期并不是到6月30日，而是到6月15日，这在设计滑窗结构的时候需要注意。
print_date_receive(off_train, 'off_train', True)
print_date_receive(on_train, 'on_train', False)
print_date_receive(off_test, 'off_test', True)


# %%
#查看online offline 训练集的 user_id与测试集的重合度
off_train_user = off_train[['user_id']].copy().drop_duplicates()
off_test_user = off_test[['user_id']].copy().drop_duplicates()
on_train_user = on_train[['user_id']].copy().drop_duplicates()
print('offline 训练集用户ID数量')
print(off_train_user.user_id.count())
print('online 训练集用户ID数量')
print(on_train_user.user_id.count())
print('offline 测试集用户ID数量')
print(off_test_user.user_id.count())
off_train_user['off_train_flag'] = 1
off_merge = off_test_user.merge(off_train_user, on='user_id',
                                how="left").reset_index().fillna(0)
print('offline 训练集用户与测试集用户重复数量')
print(off_merge['off_train_flag'].sum())
print('offline 训练集用户与测试集重复用户在总测试集用户中的占比')
print(off_merge['off_train_flag'].sum() / off_merge['off_train_flag'].count())
on_train_user['on_train_flag'] = 1
on_merge = off_test_user.merge(on_train_user, on='user_id',
                               how="left").reset_index().fillna(0)
print('online 训练集用户与测试集用户重复数量')
print(on_merge['on_train_flag'].sum())
print('online 训练集用户与测试集重复用户在总测试集用户中的占比')
print(on_merge['on_train_flag'].sum() / on_merge['on_train_flag'].count())

# %%
#分隔符
separator = ':'


#计算折扣率，将满减和折扣统一
#因为discount_rate为null的时候一般都是没有使用优惠券，这个时候折扣应该是1
def get_discount_rate(s):
    s = str(s)
    if s == 'null':
        return -1
        #return 1
    s = s.split(separator)
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])


#获取是否满减（full reduction promotion）
def get_if_fd(s):
    s = str(s)
    s = s.split(separator)
    if len(s) == 1:
        return 0
    else:
        return 1


#获取满减的条件
def get_full_value(s):
    s = str(s)
    s = s.split(separator)
    if len(s) == 1:
        return -1
    else:
        return int(s[0])


#获取满减的优惠
def get_reduction_value(s):
    s = str(s)
    s = s.split(separator)
    if len(s) == 1:
        return -1
    else:
        return int(s[1])


#获取月份
def get_month(s):
    if s[0] == 'null':
        return -1
    else:
        return int(s[4:6])


#获取日期
def get_day(s):
    if s[0] == 'null':
        return -1
    else:
        return int(s[6:8])


#获取日期间隔输入内容为Date:Date_received
def get_day_gap(s):
    s = s.split(separator)
    if s[0] == 'null':
        return -1
    if s[1] == 'null':
        return -1
    else:
        return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
                date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days


#获取Label，输入内容为Date:Date_received
def get_label(s):
    s = s.split(separator)
    if s[0] == 'null':
        return 0
    if s[1] == 'null':
        return -1
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
          date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return -1

# %%
def add_feature(df):
    df['if_fd'] = df['discount_rate'].apply(get_if_fd)
    df['full_value'] = df['discount_rate'].apply(get_full_value)
    df['reduction_value'] = df['discount_rate'].apply(get_reduction_value)
    df['discount_rate'] = df['discount_rate'].apply(get_discount_rate)
    df['distance'] = df['distance'].replace('null', -1).astype(int)
    #df['month_received'] = df['date_received'].apply(get_month)
    #df['month'] = df['date'].apply(get_month)
    return df


def add_label(df):
    df['day_gap'] = df['date'].astype(
        'str') + ':' + df['date_received'].astype('str')
    df['label'] = df['day_gap'].apply(get_label)
    df['day_gap'] = df['day_gap'].apply(get_day_gap)
    return df

    
# %%
#拷贝数据，免得调试的时候重读文件
dftrain = off_train.copy()
dftest = off_test.copy()

# %%
dftrain = add_feature(dftrain)
dftrain = add_label(dftrain)
dftest = add_feature(dftest)


# %%
