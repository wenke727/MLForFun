# In[3]:
import time
import datetime
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics  
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from sklearn import tree  
from sklearn.svm import SVC 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB 

from sklearn.model_selection import train_test_split # 切分数据

import warnings
warnings.filterwarnings("ignore")
 

#%% 
# 全局参数
from setting import fd_seperator, datapath, featurepath, resultpath, tmppath, scorepath
from setting import id_col_names, id_target_cols, target_col_name, myeval, cvscore
# 工具函数
from utils.utils import get_id_df, get_target_df, get_predictors_df, read_featurefile_train, read_featurefile_test, read_data


#%%
""" 特征读取 """
# 模型训练代码

def standize_df(train_data, test_data):
    """将特征归一化"""
    from sklearn import preprocessing
    features_columns = [f for f in test_data.columns if f not in id_target_cols]

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])

    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    test_data_scaler  = min_max_scaler.transform(test_data[features_columns])

    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns

    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns

    train_data_scaler['label'] = train_data['label']
    train_data_scaler[id_col_names] = train_data[id_col_names]
    test_data_scaler[id_col_names] = test_data[id_col_names]
    
    return train_data_scaler, test_data_scaler


def get_sklearn_model(model_name, param=None):
    """部分整合在sklearn的分类算法

    Args:
        model_name ([type]): [description]

    Returns:
        [type]: [description]
    """
    #朴素贝叶斯
    if model_name == 'NB':
        model = MultinomialNB(alpha=0.01)
    #逻辑回归
    elif model_name == 'LR':
        model = LogisticRegression(penalty='l2')
    # KNN
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    #随机森林
    elif model_name == 'RF':
        model = RandomForestClassifier()
    #决策树
    elif model_name == 'DT':
        model = tree.DecisionTreeClassifier()
    #向量机
    elif model_name == 'SVC':
        model = SVC(kernel='rbf')
    #GBDT
    elif model_name == 'GBDT':
        model = GradientBoostingClassifier()
    #XGBoost
    elif model_name == 'XGB':
        model = XGBClassifier()
    #lightGBM
    elif model_name == 'LGB':
        model = LGBMClassifier()
    else:
        print("wrong model name!")
        return
    
    if param is not None:
        model.set_params(**param)
        
    return model


def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=-1,
                        train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    """画学习曲线

    Args:
        estimator ([type]): [description]
        title ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]
        ylim ([type], optional): [description]. Defaults to None.
        cv ([type], optional): [description]. Defaults to None.
        n_jobs (int, optional): [description]. Defaults to -1.
        train_sizes (list, optional): [description]. Defaults to [0.01, 0.02, 0.05, 0.1, 0.2, 0.3].

    Returns:
        [type]: [description]
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=myeval,
        n_jobs=n_jobs,
        train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")
    plt.plot(train_sizes,
             train_scores_mean,
             'o-',
             color="r",
             label="Training score")
    plt.plot(train_sizes,
             test_scores_mean,
             'o-',
             color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt


def plot_curve_single(traindf,
                      classifier,
                      cvnum,
                      train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    """画算法的学习曲线,为加快画图速度，最多选20%数据
       学习曲线是不同训练集大小，模型在训练集和验证集上的得分变化曲线；判断模型的方差或者偏差实够过高，以及增大训练集是否可以减少过拟合

    Args:
        traindf ([type]): [description]
        classifier ([type]): [description]
        cvnum ([type]): [description]
        train_sizes (list, optional): [description]. Defaults to [0.01, 0.02, 0.05, 0.1, 0.2, 0.3].
    """
    X = get_predictors_df(traindf)
    y = get_target_df(traindf)
    estimator = get_sklearn_model(classifier)  #建模
    title = "learning curve of " + classifier + ", cv:" + str(cvnum)
    plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=(0, 1.01),
                        cv=cvnum,
                        train_sizes=train_sizes)


def myauc(test):
    """性能评价函数
        本赛题目标是预测投放的优惠券是否核销。
        针对此任务及一些相关背景知识，使用优惠券核销预测的平均AUC（ROC曲线下面积）作为评价标准。
        即对每个优惠券coupon_id单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。
        coupon平均auc计算
        
    Args:
        test ([type]): [description]

    Returns:
        [type]: [description]
    """
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        coupon_df = i[1]
        #测算AUC必须大于1个类别
        if len(coupon_df['label'].unique()) < 2:
            continue
        
        auc = metrics.roc_auc_score(coupon_df['label'], coupon_df['pred'])
        aucs.append(auc)
    
    return np.average(aucs)


def test_model(traindf, classifier, desc=None, memo=None):
    """按照日期分割

    Args:
        traindf ([type]): [description]
        classifier ([type]): [description]
    """
    train = traindf[traindf.date_received < 20160515].copy()
    test  = traindf[traindf.date_received >= 20160515].copy()

    train_data   = get_predictors_df(train).copy()
    test_data    = get_predictors_df(test).copy()
    train_target = get_target_df(train).copy()
    test_target  = get_target_df(test).copy()

    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    result = clf.predict_proba(test_data)[:, 1]
    test['pred'] = result
    score = metrics.roc_auc_score(test_target, result)
    score_coupon = myauc(test)

    print(f"{classifier} \t总体 AUC: {score:.3f}, Coupon AUC: {score_coupon:.3f}")
    
    if memo is not None:
        t = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
        memo[t] = {
            'classifier': classifier,
            'desc': desc,
            'AUC': score,
            'Coupon_AUC': score_coupon,
        }    


def test_model_split(traindf, classifier):
    """随机划分

    Args:
        traindf ([type]): [description]
        classifier ([type]): [description]
    """
    target = get_target_df(traindf).copy()

    train_all, test_all, train_target, test_target = train_test_split(
        traindf, target, test_size=0.2, random_state=0)

    train_data = get_predictors_df(train_all).copy()
    test_data = get_predictors_df(test_all).copy()

    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    result = clf.predict_proba(test_data)[:, 1]

    test = test_all.copy()
    test['pred'] = result

    score = metrics.roc_auc_score(test_target, result)
    print(classifier + "总体AUC:", score)
    score_coupon = myauc(test)
    print(classifier + " Coupon AUC:", score_coupon)


def classifier_df_simple(train_feat, test_feat, classifier):
    """预测函数

    Args:
        train_feat ([type]): [description]
        test_feat ([type]): [description]
        classifier ([type]): [description]

    Returns:
        [type]: [description]
    """
    model = get_sklearn_model(classifier)
    model.fit(get_predictors_df(train_feat), get_target_df(train_feat))
    predicted = pd.DataFrame(model.predict_proba(get_predictors_df(test_feat))[:, 1])

    return predicted


def output_predicted(predicted, resultfile, test_feat):
    """输出结果函数

    Args:
        predicted ([type]): [description]
        resultfile ([type]): [description]
        test_feat ([type]): [description]

    Returns:
        [type]: [description]
    """
    predicted = round(predicted, 3)
    resultdf = get_id_df(test_feat).copy()
    resultdf['Probability'] = predicted
    
    return resultdf


def check_classifier(train_f1, train_f2, train_f3):
    """对比分析

    Returns:
        [type]: [description]
    """
    train_log = {}

    test_model(train_f1, 'LR', 'f1', train_log)
    test_model(train_f2, 'LR', 'f2', train_log)
    test_model(train_f3, 'LR', 'f3', train_log)

    test_model(train_f1, 'NB', 'f1', train_log)
    test_model(train_f2, 'NB', 'f2', train_log)
    test_model(train_f3, 'NB', 'f3', train_log)

    test_model(train_f1, 'DT', 'f1', train_log)
    test_model(train_f2, 'DT', 'f2', train_log)
    test_model(train_f3, 'DT', 'f3', train_log)

    test_model(train_f1, 'RF', 'f1', train_log)
    test_model(train_f2, 'RF', 'f2', train_log)
    test_model(train_f3, 'RF', 'f3', train_log)

    test_model(train_f1, 'LGB', 'f1', train_log)
    test_model(train_f2, 'LGB', 'f2', train_log)
    test_model(train_f3, 'LGB', 'f3', train_log)

    test_model(train_f1, 'XGB', 'f1', train_log)
    test_model(train_f2, 'XGB', 'f2', train_log)
    test_model(train_f3, 'XGB', 'f3', train_log)

    return pd.DataFrame(train_log).T


#%%
if __name__ == '__main__':
    train_log = {}

    # 用不同的特征训练，对比分析
    train_f1, test_f1 = read_data('f1', True)
    #因为要使用KNN等进行测试，所以需要归一化
    train_f1, test_f1 = standize_df(train_f1, test_f1)

    train_f2, test_f2 = read_data('sf2', True)
    train_f2, test_f2 = standize_df(train_f2, test_f2)

    train_f3, test_f3 = read_data('sf3', True)
    train_f3, test_f3 = standize_df(train_f3, test_f3)

    # 逻辑回归
    test_model(train_f1, 'LR', 'f1', train_log)
    # 朴素贝叶斯
    test_model(train_f1, 'NB', 'f1', train_log)
    # 决策树
    test_model(train_f1, 'DT', 'f1', train_log)
    # 随机森林
    test_model(train_f1, 'RF', 'f1', train_log)
    # LightGBM
    test_model(train_f1, 'LGB', 'f1', train_log)
    # XGBoost
    test_model(train_f1, 'XGB', 'f1', train_log)


    plot_curve_single(train_f1, 'LR', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    plot_curve_single(train_f1, 'NB', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    plot_curve_single(train_f1, 'DT', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    plot_curve_single(train_f1, 'RF', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    plot_curve_single(train_f1, 'LGB', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    plot_curve_single(train_f1, 'XGB', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])


    # 可以发现特征f2比特征f1的结果好很多，这是因为特征2使用滑窗方案，增加了很多统计特征。而特征3比特征又有了很大的提高，这是因为特征3增加了穿越特征。对比LightGBM和LR的成绩可以发现对于本问题，LightGBM有着更好的成绩。    
    df_train_log = check_classifier(train_f1, train_f2, train_f3)


    # 结果输出
    # 通过分析发现特征sf3版本通过lightGBM分析的结果不错。下一步要做的事就是输出结果。
    train_f3.head()
    predicted = classifier_df_simple(train_f3, test_f3, 'LGB')
    predicted.head()

    #生成结果数据
    result = output_predicted(predicted, 'sf3_LGB.csv', test_f3)
    result.head()
    result.to_csv('./result/sf3_lgb.csv', header=False, index=False, sep=',')


