"""
代码为分层架构，主要分为以下几部分：  
    工具函数层，  
    各个特征群的生成函数层，  
    版本集成层，  
    特征生成层，  

调用关系为：  
    特征生成函数->版本集成函数->特征群的生成函数->工具函数  

这个代码结构的优点是：  
    1，便于灵活调整  
    2，利于版本控制，利于复现成绩  
    3，减少重复逻辑，整体代码量较少  
    4，便于后续编写脚本自动执行多个实验  
"""

# %%
import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 全局参数
from setting import fd_seperator, datapath, featurepath, resultpath, tmppath, scorepath
from setting import id_col_names, id_target_cols, target_col_name, myeval, cvscore


# %%
"""1. 工具函数"""
def get_discount_rate(s):
    # 计算折扣率，将满减和折扣统一
    s = str(s)
    if s == 'null':
        return -1

    s = s.split(fd_seperator)
    if len(s) == 1:
        return float(s[0])
    else:
        return round((1.0 - float(s[1]) / float(s[0])), 3)


def get_if_fd(s):
    # 获取是否满减（full reduction promotion）
    s = str(s)
    s = s.split(fd_seperator)

    if len(s) == 1:
        return 0
    else:
        return 1


def get_full_value(s):
    # 获取满减的条件
    s = str(s)
    s = s.split(fd_seperator)

    if len(s) == 1:
        # return 'null'
        return np.nan
    else:
        return int(s[0])


def get_reduction_value(s):
    # 获取满减的优惠
    s = str(s)
    s = s.split(fd_seperator)
    if len(s) == 1:
        # return 'null'
        return np.nan
    else:
        return int(s[1])


def get_day_gap(s):
    # 获取日期间隔，输入内容为Date_received:Date
    s = s.split(fd_seperator)
    if s[0] == 'null':
        return -1

    if s[1] == 'null':
        return -1
    else:
        return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
                date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days


def get_label(s):
    # 获取Label，输入内容为Date:Date_received
    s = s.split(fd_seperator)
    if s[0] == 'null':
        return 0
    if s[1] == 'null':
        return -1
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
          date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return 0


def add_discount(df):
    # 增加折扣相关特征
    df['if_fd'] = df['discount_rate'].apply(get_if_fd)
    df['full_value'] = df['discount_rate'].apply(get_full_value)
    df['reduction_value'] = df['discount_rate'].apply(get_reduction_value)
    df['discount_rate'] = df['discount_rate'].apply(get_discount_rate)
    df.distance = df.distance.replace('null', np.nan)

    return df


def add_day_gap(df):
    # 计算日期间隔
    df['day_gap'] = df['date'].astype(
        'str') + ':' + df['date_received'].astype('str')
    df['day_gap'] = df['day_gap'].apply(get_day_gap)

    return df


def add_label(df):
    # 获取label
    df['label'] = df['date'].astype(
        'str') + ':' + df['date_received'].astype('str')
    df['label'] = df['label'].apply(get_label)

    return df


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        # return -1
        return np.nan


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        # 将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) -
                    dt.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        # return -1
        return np.nan
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8])) -
                    dt.datetime(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        # return -1
        return np.nan
    else:
        return min(gaps)


def add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names):
    """统计特征处理函数

    Args:
        df: 添加特征的dataframe
        df_group: 特征生成的数据集
        group_cols: group by 的列
        value_col: 被统计的列
        agg_ops:处理方式 包括：count,mean,sum,std,max,min,nunique
        colname: 新特征的名称

    Returns:
        [type]: [description]
    """
    df_group[value_col] = df_group[value_col].astype('float')
    df_agg = pd.DataFrame(df_group.groupby(group_cols)[value_col].agg(agg_ops)).reset_index()
    df_agg.columns = group_cols + col_names
    df = df.merge(df_agg, on=group_cols, how="left")

    return df


def add_agg_feature(df, df_group, group_cols, value_col, agg_ops, keyword):
    """统计特征处理函数
        名称按照keyword+'_'+value_col+'_'+op 自动增加

    Args:
        df ([type]): [description]
        df_group ([type]): [description]
        group_cols ([type]): [description]
        value_col ([type]): [description]
        agg_ops ([type]): [description]
        keyword ([type]): [description]

    Returns:
        [type]: [description]
    """
    col_names = []
    for op in agg_ops:
        col_names.append(keyword + '_' + value_col + '_' + op)
    df = add_agg_feature_names(
        df, df_group, group_cols, value_col, agg_ops, col_names)

    return df


def add_count_new_feature(df, df_group, group_cols, new_feature_name):
    """因为count特征很多，开发了这个专门提取count特征的函数

    Args:
        df ([type]): [description]
        df_group ([type]): [description]
        group_cols ([type]): [description]
        new_feature_name ([type]): [description]

    Returns:
        [type]: [description]
    """
    df_group[new_feature_name] = 1
    df_group = df_group.groupby(group_cols).agg('sum').reset_index()
    df = df.merge(df_group, on=group_cols, how="left")

    return df


# %%
"""2. 特征群生成"""
def get_merchant_feature(feature):
    """[获取商家相关特征]

    Args:
        feature ([type]): [description]

    Returns:
        [type]: [description]
    """
    merchant = feature[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']].copy()
    # 删除重复行数据
    t = merchant[['merchant_id']].copy()
    t.drop_duplicates(inplace=True)

    # 每个商户的交易总次数
    t1 = merchant[merchant.date != 'null'][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(t, t1, 
                                             'merchant_id', 'total_sales')

    # 每个商户销售中，使用了优惠券的交易次数（正样本）
    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(merchant_feature, t2,
                                             'merchant_id', 'sales_use_coupon')

    # 每个商户发放的优惠券总数
    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(merchant_feature, t3,
                                             'merchant_id', 'total_coupon')

    # 在每个线下商户含有优惠券的交易中，统计和用户距离的最大值、最小值、平均值、中位值
    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null') &
                  (merchant.distance != 'null')][['merchant_id', 'distance']].copy()
    t4.distance = t4.distance.astype('int')
    merchant_feature = add_agg_feature(df=merchant_feature, 
                                       df_group=t4, 
                                       group_cols=['merchant_id'],
                                       value_col='distance',
                                       agg_ops=['min', 'max', 'mean', 'median'],
                                       keyword='merchant')

    # 将数据中的NaN用0来替换
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan, 0)
    # 商户发放优惠券的使用率
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_coupon
    # 商户的交易中，使用优惠券的交易占比
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_sales
    # 将数据中的NaN用0来替换
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)

    return merchant_feature


def get_user_feature(feature):
    """获取用户相关特征

    Args:
        feature ([type]): [description]

    Returns:
        [type]: [description]
    """
    # for dataset3
    atts = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    user = feature[atts].copy()

    t = user[['user_id']].copy()
    t.drop_duplicates(inplace=True)

    # 每个用户交易的商户数
    t1 = user[user.date != 'null'][['user_id', 'merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1 = t1[['user_id']]
    user_feature = add_count_new_feature(t, t1, 'user_id', 'count_merchant')

    # 在每个用户线下使用优惠券产生的交易中，统计和商户距离的最大值、最小值、平均值、中位值
    t2 = user[(user.date != 'null') & (user.coupon_id != 'null') & (user.distance != 'null')][['user_id', 'distance']]
    t2.distance = t2.distance.astype('int')
    user_feature = add_agg_feature(user_feature, 
                                   t2, 
                                   ['user_id'], 
                                   'distance',
                                   ['min', 'max', 'mean', 'median'],
                                   'user')

    # 每个用户使用优惠券消费的次数
    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    user_feature = add_count_new_feature(user_feature, t7, 'user_id', 'buy_use_coupon')

    # 每个用户消费的总次数
    t8 = user[user.date != 'null'][['user_id']]
    user_feature = add_count_new_feature(user_feature, t8, 'user_id', 'buy_total')

    # 每个用户收到优惠券的总数
    t9 = user[user.coupon_id != 'null'][['user_id']]
    user_feature = add_count_new_feature(user_feature, t9, 'user_id', 'coupon_received')

    # 用户从收到优惠券到用券消费的时间间隔，统计其最大值、最小值、平均值、中位值
    t10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    t10 = add_day_gap(t10)
    t10 = t10[['user_id', 'day_gap']]
    user_feature = add_agg_feature(user_feature, 
                                   t10, 
                                   ['user_id'], 
                                   'day_gap',
                                   ['min', 'max', 'mean', 'median'], 
                                   'user')
    
    # 将数据中的NaN用0来替换
    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan, 0)

    # 统计用户用券消费在用户总消费次数的占比
    user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.buy_total.astype('float')
    # 统计用户收到消费券的使用率
    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.coupon_received.astype('float')
    # 将数据中的NaN用0来替换
    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    
    return user_feature


def get_user_merchant_feature(feature):
    """提取用户和商户关系特征

    Args:
        feature ([type]): [description]

    Returns:
        [type]: [description]
    """
    t = feature[['user_id', 'merchant_id']].copy()
    t.drop_duplicates(inplace=True)

    # 一个用户在一个商家交易的总次数
    t0 = feature[['user_id', 'merchant_id', 'date']].copy()
    t0 = t0[t0.date != 'null'][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(t, t0, ['user_id', 'merchant_id'], 'user_merchant_buy_total')

    # 一个用户在一个商家一共收到的优惠券数量
    t1 = feature[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t1, ['user_id', 'merchant_id'], 'user_merchant_received')

    # 一个用户在一个商家使用优惠券消费的次数
    t2 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t2, ['user_id', 'merchant_id'], 'user_merchant_buy_use_coupon')

    t3 = feature[['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t3,  ['user_id', 'merchant_id'], 'user_merchant_any')

    # 一个用户在一个商家没有使用优惠券消费的次数
    t4 = feature[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t4, ['user_id', 'merchant_id'], 'user_merchant_buy_common')
    
    # 将数据中的NaN用0来替换
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)
    
    # 一个用户对一个商家发放的优惠券的使用率
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_received.astype('float')
    
    # 一个用户在一个商家总的消费次数中，用优惠券的消费次数占比
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
    
    # FIXME 一个用户到店后消费的可能性统计
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype('float') / user_merchant.user_merchant_any.astype('float')
    # 一个用户在一个商家总的消费次数中，不用优惠券的消费次数占比
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
    
    return user_merchant


def get_leakage_feature(dataset):
    """# TODO 提取穿越特征

    Args:
        dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    t = dataset[['user_id']].copy()
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset[['user_id', 'coupon_id']].copy()
    t1['this_month_user_receive_same_coupn_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t2.date_received = t2.date_received.astype('str')
    # 如果出现相同的用户接收相同的优惠券在接收时间上用‘：’连接上第n次接受优惠券的时间
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    # 将接收时间的一组按着':'分开，这样就可以计算接受了优惠券的数量,apply是合并
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    # 最大接受的日期
    t2['max_date_received'] = t2.date_received.apply(
        lambda s: max([int(d) for d in s.split(':')]))
    # 最小的接收日期
    t2['min_date_received'] = t2.date_received.apply(
        lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset[['user_id', 'coupon_id', 'date_received']]
    # 将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    # 这个优惠券最近接受时间
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype(
        int)
    # 这个优惠券最远接受时间
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(
        int) - t3.min_date_received

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_firstlastone)
    t3 = t3[[
        'user_id', 'coupon_id', 'date_received',
        'this_month_user_receive_same_coupon_lastone',
        'this_month_user_receive_same_coupon_firstone'
    ]]

    # 提取第四个特征,一个用户所接收到的所有优惠券的数量
    t4 = dataset[['user_id', 'date_received']].copy()
    t4['this_day_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    # 提取第五个特征,一个用户不同时间所接收到不同优惠券的数量
    t5 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id',
                     'date_received']).agg('sum').reset_index()

    # 一个用户不同优惠券 的接受时间
    t6 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby([
        'user_id', 'coupon_id'
    ])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[[
        'user_id', 'coupon_id', 'date_received', 'day_gap_before',
        'day_gap_after'
    ]]

    other_feature = pd.merge(t1, t, on='user_id')
    other_feature = pd.merge(other_feature, t3, on=['user_id', 'coupon_id'])
    other_feature = pd.merge(other_feature,
                             t4,
                             on=['user_id', 'date_received'])
    other_feature = pd.merge(other_feature,
                             t5,
                             on=['user_id', 'coupon_id', 'date_received'])
    other_feature = pd.merge(other_feature,
                             t7,
                             on=['user_id', 'coupon_id', 'date_received'])
    
    return other_feature


"""3. 版本集成层"""
def f1(dataset, if_train):
    # 特征1只有最基础的特征
    result = add_discount(dataset)
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


def f2(dataset, feature, if_train):
    # 特征2增加Merchant,user特征
    result = add_discount(dataset)
    
    merchant_feature = get_merchant_feature(feature)
    result = result.merge(merchant_feature, on='merchant_id', how="left")
    
    user_feature = get_user_feature(feature)
    result = result.merge(user_feature, on='user_id', how="left")
    
    user_merchant = get_user_merchant_feature(feature)
    result = result.merge(user_merchant,
                          on=['user_id', 'merchant_id'],
                          how="left")
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    
    return result


def f3(dataset, feature, if_train):
    """
    特征3增加leakage特征
    """
    result = add_discount(dataset)
    
    merchant_feature = get_merchant_feature(feature)
    result = result.merge(merchant_feature, on='merchant_id', how="left")
    
    user_feature = get_user_feature(feature)
    result = result.merge(user_feature, on='user_id', how="left")
    
    user_merchant = get_user_merchant_feature(feature)
    result = result.merge(user_merchant,
                          on=['user_id', 'merchant_id'],
                          how="left")

    leakage_feature = get_leakage_feature(dataset)
    result = result.merge(leakage_feature,
                          on=['user_id', 'coupon_id', 'date_received'],
                          how='left')

    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
        
    return result


def optimize_feature(old_feature, new_feature):
    """
    特征后续处理函数
    可以根据原有特征，进行后处理。生成新的特征。
    """
    from sklearn import preprocessing
    train_data = pd.read_csv(featurepath + 'train_' + old_feature + '.csv',
                             sep=',',
                             encoding="utf-8").fillna(0)
    test_data = pd.read_csv(featurepath + 'test_' + old_feature + '.csv',
                            sep=',',
                            encoding="utf-8").fillna(0)
    id_target_cols = ['user_id', 'coupon_id', 'date_received', 'label']

    # 归一化
    features_columns = [f for f in test_data.columns if f not in id_target_cols]

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])

    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    test_data_scaler = min_max_scaler.transform(test_data[features_columns])

    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns

    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns

    # 可以增加其他处理内容
    train_data_scaler['label'] = train_data['label']
    train_data_scaler.to_csv(featurepath + 'train_s' + new_feature + '.csv',
                             index=False,
                             sep=',')
    test_data_scaler.to_csv(featurepath + 'test_s' + new_feature + '.csv',
                            index=False,
                            sep=',')


# %%
"""4. 特征生成函"""
def normal_feature_generate(feature_function):
    """生成不滑窗的特征
        特征名：训练集：train_版本函数，测试集:test_版本函数

    Args:
        feature_function ([type]): [description]
    """
    off_train = pd.read_csv(datapath + 'ccf_offline_stage1_train.csv',
                            header=0,
                            keep_default_na=False)
    off_train.columns = [
        'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
        'date_received', 'date'
    ]

    off_test = pd.read_csv(datapath + 'ccf_offline_stage1_test_revised.csv',
                           header=0,
                           keep_default_na=False)
    off_test.columns = [
        'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
        'date_received'
    ]

    # 取时间大于'20160501'是为了数据量少点，模型算的快一点，如果时间够的话，可以不加这个限制
    off_train = off_train[(off_train.coupon_id != 'null') & 
                          (off_train.date_received != 'null') &
                          (off_train.date_received >= '20160501')]

    dftrain = feature_function(off_train, True)

    dftest = feature_function(off_test, False)

    dftrain.drop(['date'], axis=1, inplace=True)
    dftrain.drop(['merchant_id'], axis=1, inplace=True)
    dftest.drop(['merchant_id'], axis=1, inplace=True)

    # 输出特征
    print(f'{feature_function.__name__}: 输出特征')
    dftrain.to_csv(featurepath + 'train_' + feature_function.__name__ + '.csv',
                   index=False,
                   sep=',')
    dftest.to_csv(featurepath + 'test_' + feature_function.__name__ + '.csv',
                  index=False,
                  sep=',')


def slide_feature_generate(feature_function):
    """生成滑窗特征
        特征名：训练集：train_s版本函数，测试集:test_s版本函数, s是slide滑窗的意思

    Args:
        feature_function ([type]): [description]
    """
    off_train = pd.read_csv(datapath + 'ccf_offline_stage1_train.csv',
                            header=0,
                            keep_default_na=False)
    off_train.columns = [
        'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
        'date_received', 'date'
    ]

    off_test = pd.read_csv(datapath + 'ccf_offline_stage1_test_revised.csv',
                           header=0,
                           keep_default_na=False)
    off_test.columns = [
        'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
        'date_received'
    ]

    # 交叉训练集一：收到券的日期大于4月14日和小于5月14日
    dataset1 = off_train[(off_train.date_received >= '201604014')
                         & (off_train.date_received <= '20160514')]
    # 交叉训练集一特征：线下数据中领券和用券日期大于1月1日和小于4月13日
    feature1 = off_train[(off_train.date >= '20160101') &
                         (off_train.date <= '20160413') |
                         ((off_train.date == 'null') &
                          (off_train.date_received >= '20160101') &
                          (off_train.date_received <= '20160413'))]

    # 交叉训练集二：收到券的日期大于5月15日和小于6月15日
    dataset2 = off_train[(off_train.date_received >= '20160515')
                         & (off_train.date_received <= '20160615')]
    # 交叉训练集二特征：线下数据中领券和用券日期大于2月1日和小于5月14日
    feature2 = off_train[(off_train.date >= '20160201') &
                         (off_train.date <= '20160514') |
                         ((off_train.date == 'null') &
                          (off_train.date_received >= '20160201') &
                          (off_train.date_received <= '20160514'))]

    # 测试集
    dataset3 = off_test
    #测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
    feature3 = off_train[((off_train.date >= '20160315') &
                          (off_train.date <= '20160630')) |
                         ((off_train.date == 'null') &
                          (off_train.date_received >= '20160315') &
                          (off_train.date_received <= '20160630'))]

    dftrain1 = feature_function(dataset1, feature1, True)
    dftrain2 = feature_function(dataset2, feature2, True)
    dftrain = pd.concat([dftrain1, dftrain2], axis=0)

    dftest = feature_function(dataset3, feature3, False)

    dftrain.drop(['date'], axis=1, inplace=True)
    dftrain.drop(['merchant_id'], axis=1, inplace=True)
    dftest.drop(['merchant_id'], axis=1, inplace=True)

    # 输出特征
    print(f'{feature_function.__name__}, 输出特征')
    dftrain.to_csv(featurepath + 'train_s' + feature_function.__name__ +
                   '.csv',
                   index=False,
                   sep=',')
    dftest.to_csv(featurepath + 'test_s' + feature_function.__name__ + '.csv',
                  index=False,
                  sep=',')


# %%
"""
特征分析
"""

id_col_names = ['user_id', 'coupon_id', 'date_received']
target_col_name = 'label'
id_target_cols = ['user_id', 'coupon_id', 'date_received', 'label']


"""5. 数据读取工具函数"""

def get_id_df(df):
    """返回ID列"""

    return df[id_col_names]


def get_target_df(df):
    # 返回Target列
    return df[target_col_name]


def get_predictors_df(df):
    # 返回特征列
    predictors = [f for f in df.columns if f not in id_target_cols]
    return df[predictors]


def read_featurefile_train(featurename):
    # 按特征名读取训练集
    df = pd.read_csv(featurepath + 'train_' + featurename + '.csv',
                     sep=',',
                     encoding="utf-8")
    # df.fillna(0,inplace=True)
    return df


def read_featurefile_test(featurename):
    # 按特征名读取测试集
    df = pd.read_csv(featurepath + 'test_' + featurename + '.csv',
                     sep=',',
                     encoding="utf-8")
    # df.fillna(0,inplace=True)
    return df


def read_data(featurename):
    # 按特征名读取数据
    traindf = read_featurefile_train(featurename)
    testdf = read_featurefile_test(featurename)
    return traindf, testdf



# %%

# if __name__ == '__main__':
    # pass
    
# 生成特征文件
# 生成的特征中，sf3版本是特征最全的面的特征。
# f1, sf2是简单版，主要是做对比使用。
# 这个代码结构很容易做新版本的特征，大家可以自己进行尝试。后续会在sf3的基础上对特征数据进行分析，优化，最后生成最终的特征。

# f1
normal_feature_generate(f1)
# sf2
slide_feature_generate(f2)
# sf3
slide_feature_generate(f3)


# %%
# 对SF3（最全的一版）版本特征进行分析
traindf, testdf = read_data('sf3')
train_X = get_predictors_df(traindf)
train_y = get_target_df(traindf)
test_X = get_predictors_df(testdf)



# %%
# ## 特征数据总览
traindf.describe()


# %%
testdf.describe()



# %%
# ## 查看数据分布
# 画箱式图
column = train_X.columns.tolist()[:46]  # 列表头
fig = plt.figure(figsize=(20, 40))  # 指定绘图对象宽度和高度
for i in range(45):
    plt.subplot(15, 3, i + 1)  # 15行3列子图
    sns.boxplot(train_X[column[i]], orient="v", width=0.5)  # 箱式图
    plt.ylabel(column[i], fontsize=8)
plt.show()


# %%
# 画箱式图
column = test_X.columns.tolist()[:46]  # 列表头
fig = plt.figure(figsize=(20, 40))  # 指定绘图对象宽度和高度
for i in range(45):
    plt.subplot(15, 3, i + 1)  # 15行3列子图
    sns.boxplot(test_X[column[i]], orient="v", width=0.5)  # 箱式图
    plt.ylabel(column[i], fontsize=8)
plt.show()


# %%
# 对比分布
dist_cols = 4
dist_rows = len(test_X.columns)

plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

for i, col in enumerate(test_X.columns):
    ax = plt.subplot(dist_rows, dist_cols, i + 1)
    ax = sns.kdeplot(train_X[col], color="Red", shade=True)
    ax = sns.kdeplot(test_X[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])

plt.show()


# 通过查看可以发现很多数据训练集和测试集分布不同，不过这和我们在数据探索时就已经发现测试集和训练集中是否满减情况相差很大：
# 训练集满减情况
# 1    0.581241
# 0    0.418759
#
# 测试集满减情况
# 1    0.97742
# 0    0.02258
# 这在if_fd的那张图里面也能看到：
#
# 因为测试集主要是满减，所以我们再对比一下训练集满减数据的分布与测试集满减数据分布的对比。

# %%
train_X_fd1 = train_X[train_X.if_fd == 1].reset_index(drop=True)
test_X_fd1 = test_X[test_X.if_fd == 1].reset_index(drop=True)
dist_cols = 4
dist_rows = len(test_X_fd1.columns)

plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

for i, col in enumerate(test_X_fd1.columns):
    ax = plt.subplot(dist_rows, dist_cols, i + 1)
    ax = sns.kdeplot(train_X_fd1[col], color="Red", shade=True)
    ax = sns.kdeplot(test_X_fd1[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])

plt.show()


# 查看后发现'sales_use_coupon', 'total_coupon', 'merchant_distance_median', 'buy_use_coupon_rate', 'user_merchant_coupon_transfer_rate'特征在训练集和测试集之间相差比较大，之外，其他的训练集和测试集分布相差不是很大。不过因为这是生成的特征，在训练集和测试集之间可能因为不同时间的商家、用户占比不同，所以造成特征分布不同。不能因为分布不同就直接删除。要在后续通过模型来选择

# ## 特征相关性

# %%
plt.figure(figsize=(20, 16))
column = traindf.columns.tolist()
mcorr = traindf[column].corr(method="spearman")
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr,
                mask=mask,
                cmap=cmap,
                square=True,
                annot=True,
                fmt='0.2f')
plt.show()


# ## 相关性分析

# %%
mcorr = mcorr.abs()
numerical_corr = mcorr[mcorr['label'] > 0.1]['label']
print(numerical_corr.sort_values(ascending=False))

#index0 = numerical_corr.sort_values(ascending=False).index
# print(traindf[index0].corr('spearman'))


# 可以发现几个穿越特征都排在了前面，因为它们都是在已经发生的“事实”基础上统计的，所以相关性一半会比正常的特征强。在正常的特征中，客户与某个商家之间的交互特征因为指向性很强，所以也都排在前面。

# 特征工程，到此结束。生成3个版本的特征：f1,sf2,sf3,后续模型的训练将主要以此为主。
# 没有对特征进行归一化等操作，因为对于决策树和随机森林以及XGboost算法而言，特征缩放对于它们没有什么影响，而后续主要是用LGB，XGB等算法。不过我也提供了归一化等优化的方案，大家可以运行生成新的特征，进行尝试。

# # 特征后续处理

# %%

# 生成新版特征sf4
optimize_feature('sf3', 'sf4')


# %%
