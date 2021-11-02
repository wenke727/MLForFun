#%%
import time
import datetime
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics  
from sklearn.metrics import roc_auc_score
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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, LeavePOut
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


#%%
# 全局参数
from setting import fd_seperator, datapath, featurepath, resultpath, tmppath, scorepath
from setting import id_col_names, id_target_cols, target_col_name, myeval, cvscore
# 工具函数
from utils.utils import get_id_df, get_target_df, get_predictors_df, read_featurefile_train, read_featurefile_test, read_data

from _04_tarin import standize_df, myauc, get_sklearn_model, plot_learning_curve, plot_curve_single


# 预测方式，因为要的结果是购买的几率，所以不能直接用Predict因为这样会直接返回0,1,而要用predict_proba，它会返回每个类别的可能行，取其中为1的列即可

#%%

""" validation method """

def simple_validate(df, classifier, desc='LR'):
    target = get_target_df(df).copy()
    traindf = df.copy()
    train_all, test_all, train_target, test_target = train_test_split(traindf, target, test_size=.2, random_state=0)

    train_data = get_predictors_df(train_all).copy()
    test_data  = get_predictors_df(test_all).copy()

    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    train_pred = clf.predict_proba(train_data)[:, 1]
    test_pred  = clf.predict_proba(test_data)[:,1]

    score_train = roc_auc_score(train_target, train_pred)
    score_test  = roc_auc_score(test_target, test_pred)

    train_all['pred'] = train_pred
    test_all['pred'] = test_pred

    print(f"{classifier} train/test 总体AUC: [{score_train:.3f}, {score_test:.3f}], Coupon AUC [{myauc(train_all):.3f}, {myauc(test_all):.3f}]", )
    
    return


def _cross_validate(df, classifier, desc='LR'):
    target = get_target_df(df).copy()
    train = df.copy()

    kf = KFold(n_splits=5)
    for k, (train_index, test_index) in enumerate(kf.split(train)):
        train_all, test_all = train.iloc[train_index], train.iloc[test_index]
        train_target, test_target = target.iloc[train_index], target.iloc[test_index]

        train_data = get_predictors_df(train_all).copy()
        test_data  = get_predictors_df(test_all).copy()

        clf = get_sklearn_model(classifier)
        clf.fit(train_data, train_target)
        train_pred = clf.predict_proba(train_data)[:, 1]
        test_pred  = clf.predict_proba(test_data)[:,1]

        score_train = roc_auc_score(train_target, train_pred)
        score_test  = roc_auc_score(test_target, test_pred)

        train_all['pred'] = train_pred
        test_all['pred'] = test_pred

        print(f"{classifier} train/test 总体AUC: [{score_train:.3f}, {score_test:.3f}], Coupon AUC [{myauc(train_all):.3f}, {myauc(test_all):.3f}]", )
    
    return


def _leave_p_validate(df, classifier):
    train = train_f3.copy()
    target = get_target_df(train_f3).copy()

    from sklearn.model_selection import LeavePOut
    lpo = LeavePOut(p=200)
    num = 100
    for k, (train_index, test_index) in enumerate(lpo.split(train)):
        train_all, test_all, train_target, test_target = train.iloc[
            train_index], train.iloc[test_index], target[train_index], target[test_index]

        train_data = get_predictors_df(train_all).copy()
        test_data  = get_predictors_df(test_all).copy()

        clf = get_sklearn_model(classifier)
        clf.fit(train_data, train_target)
        train_pred = clf.predict_proba(train_data)[:, 1]
        test_pred  = clf.predict_proba(test_data)[:,1]

        score_train = roc_auc_score(train_target, train_pred)
        score_test  = roc_auc_score(test_target, test_pred)

        train_all['pred'] = train_pred
        test_all['pred'] = test_pred
        
        print(f"{classifier} train/test 总体AUC: [{score_train:.3f}, {score_test:.3f}], Coupon AUC [{myauc(train_all):.3f}, {myauc(test_all):.3f}]", )
        
        if k >= 5:
            break


def _straight_validate(df, classifier):
    train = df.copy()
    target = get_target_df(df).copy()

    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5)

    for k, (train_index, test_index) in enumerate(kf.split(train, target)):
        train_all, test_all, train_target, test_target = train.iloc[
            train_index], train.iloc[test_index], target[train_index], target[test_index]

        train_data = get_predictors_df(train_all).copy()
        test_data  = get_predictors_df(test_all).copy()

        clf = get_sklearn_model(classifier)
        clf.fit(train_data, train_target)
        train_pred = clf.predict_proba(train_data)[:, 1]
        test_pred  = clf.predict_proba(test_data)[:,1]

        score_train = roc_auc_score(train_target, train_pred)
        score_test  = roc_auc_score(test_target, test_pred)

        train_all['pred'] = train_pred
        test_all['pred'] = test_pred
        
        print(f"{classifier} train/test 总体AUC: [{score_train:.3f}, {score_test:.3f}], Coupon AUC [{myauc(train_all):.3f}, {myauc(test_all):.3f}]", )
        
        if k >= 5:
            break


def classifier_df_score(train_feat, classifier, cvnum=5, param=None, memo={}, desc=None):
    """对算法进行分析

    Args:
        train_feat ([type]): [description]
        classifier ([type]): [description]
        cvnum ([type]): [description]
        param ([type], optional): [description]. Defaults to None.
    """
    clf = get_sklearn_model(classifier, param)
    train = train_feat.copy()
    target = get_target_df(train_feat).copy()
    kf = StratifiedKFold(n_splits=cvnum)
    
    scores, score_coupons = [], []
    for k, (train_idx, test_idx) in enumerate(kf.split(train, target)):
        train_all, test_all = train.iloc[train_idx], train.iloc[test_idx]
        train_target, test_target = target.iloc[train_idx], target.iloc[test_idx]

        train_data = get_predictors_df(train_all).copy()
        test_data  = get_predictors_df(test_all).copy()

        clf = get_sklearn_model(classifier)
        clf.fit(train_data, train_target)
        
        # train_pred = clf.predict_proba(train_data)[:, 1]
        # score_train = roc_auc_score(train_target, train_pred)
        # train_all['pred'] = train_pred
        
        test_pred  = clf.predict_proba(test_data)[:,1]
        test_all['pred'] = test_pred
        score_test  = roc_auc_score(test_target, test_pred)
        score_coupon_test = myauc(test_all)
        
        scores.append(score_test)
        score_coupons.append(score_coupon_test)

    print(f"{classifier} 总体/coupon AUC: {np.average(score_test):.3f}, {np.average(score_coupon_test):.3f}", )
    # print(f"\tscore_test:\t {scores}", )
    # print(f"\tscore_coupon_test: \t{score_coupons}", )

    if memo is not None:
        t = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
        memo[t] = {
            'classifier': classifier,
            'AUC': score_test,
            'Coupon_AUC': score_coupon_test,
        }    
        if desc is not None:
            memo[t]['desc'] = desc

    return np.average(score_coupon_test)


""" Tune """
def grid_search_example():
    train = get_predictors_df(train_f3)
    target = get_target_df(train_f3)
    train.head()

    train_data, test_data, train_target, test_target = train_test_split(
        train, target, test_size=0.2, random_state=0)

    model = RandomForestClassifier()
    parameters = {'n_estimators': [20, 50, 100], 'max_depth': [1, 2, 3]}

    clf = GridSearchCV(model, parameters, cv=3, verbose=2, n_jobs=-1)
    clf.fit(train_data, train_target)

    score_test = roc_auc_score(test_target, clf.predict(test_data))

    print("RandomForestClassifier GridSearchCV test AUC:   ", score_test)
    print("最优参数:")
    print(clf.best_params_)
    # sorted(clf.cv_results_.keys())


def grid_search_helper(train_data, train_target, params, param_grid, verbose=2):
    model = LGBMClassifier(**params)
    clf = GridSearchCV(model, param_grid, cv=3, verbose=verbose, n_jobs=-1)
    clf.fit(train_data, train_target)
    score_test = roc_auc_score(test_target, clf.predict_proba(test_data)[:, 1])

    print(f"GridSearchCV AUC Score: {score_test} ,最优参数: {clf.best_params_}")
    
    return 


def grid_plot(train_feat, classifier, cvnum, param_name, param_range, param=None):
    # 对进行网格调参进行可视化
    from sklearn.model_selection import validation_curve
    train_scores, test_scores = validation_curve(
        estimator = get_sklearn_model(classifier, param),
        X = get_predictors_df(train_feat),
        y = get_target_df(train_feat),
        param_name = param_name,
        param_range = param_range,
        cv = cvnum,
        scoring = 'roc_auc',
        n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    
    plt.title("Validation Curve with " + param_name)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range,
                 train_scores_mean,
                 label="Training score",
                 color="r")
    plt.fill_between(param_range,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.2,
                     color="r")
    plt.semilogx(param_range,
                 test_scores_mean,
                 label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.2,
                     color="g")
    plt.legend(loc="best")
    plt.show()
    
    return



#%%
if __name__ == '__main__':
    train_f1, test_f1 = read_data('f1', True)
    train_f2, test_f2 = read_data('sf2', True)
    train_f3, test_f3 = read_data('sf3', True)

    # %%
    """ 验证方式 """
    simple_validate(train_f1, 'LR')
    simple_validate(train_f3, 'LR')

    _cross_validate(train_f3, 'LR')
    _leave_p_validate(train_f3, "LR")
    _straight_validate(train_f3, 'LR')

    # %%
    # 通过对比训练集上不同算法的运算结果可以发现，F1特征集因为特征比较少，有严重的欠拟合，所以所有算法的分数都比较低。
    # F2特征集通过滑窗增加统计特征，它的分数比f1有了飞跃性的提高，其实在现实的业务场景F2+LR已经是一个很常用的解决方案了。之所以在实际作业中更倾向逻辑回归而不是类似LightGBM的算法，是为了减少计算量。当然如果计算资源不是问题的话，LightGBM也是一个好选择
    train_memo = {}

    print('特征f1, 不同模型5折训练Score：')
    classifier_df_score(train_f1, 'NB', 5, desc='f1', memo=train_memo)
    classifier_df_score(train_f1, 'LR', 5, desc='f1', memo=train_memo)
    classifier_df_score(train_f1, 'RF', 5, desc='f1', memo=train_memo)
    classifier_df_score(train_f1, 'LGB', 5, desc='f1', memo=train_memo)

    print('特征f2, 不同模型5折训练Score：')
    classifier_df_score(train_f2, 'NB', 5, desc='f2', memo=train_memo)
    classifier_df_score(train_f2, 'LR', 5, desc='f2', memo=train_memo)
    classifier_df_score(train_f2, 'RF', 5, desc='f2', memo=train_memo)
    classifier_df_score(train_f2, 'LGB', 5, desc='f2', memo=train_memo)

    print('特征f3, 不同模型5折训练Score：')
    classifier_df_score(train_f3, 'NB', 5, desc='f3', memo=train_memo)
    classifier_df_score(train_f3, 'LR', 5, desc='f3', memo=train_memo)
    classifier_df_score(train_f3, 'RF', 5, desc='f3', memo=train_memo)
    classifier_df_score(train_f3, 'LGB', 5, desc='f3', memo=train_memo)

    # %%
    """绘制学习曲线进行可视化分析"""
    # 我们还可以通过绘制学习曲线，对训练的过程进行比较深入的了解。

    for clf in ['LR', "NB", "DT", "RF", "LGB"]:
        plot_curve_single(train_f1, clf, 3, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])

    # %%
    """模型超参数空间及调参"""

    #f3特征
    traindf = train_f3.copy()

    #按日期分割，为了加快速度，只用了一般数据进行网格调参，正式的时候应该全用
    train = traindf[traindf.date_received < 20160515]
    test = traindf[traindf.date_received >= 20160515]

    train_data = get_predictors_df(train).copy()
    train_target = get_target_df(train).copy()
    test_data = get_predictors_df(test).copy()
    test_target = get_target_df(test).copy()

    traindf.head()

    # LightGBM 调参次序：  
    # 第一步：学习率和迭代次数  
    # 第二步：确定max_depth和num_leaves  
    # 第三步：确定min_data_in_leaf和max_bin in  
    # 第四步：确定feature_fraction、bagging_fraction、bagging_freq  
    # 第五步：确定lambda_l1和lambda_l2  
    # 第六步：确定 min_split_gain   
    # 第七步：降低学习率，增加迭代次数，验证模型  


    # step 1: n_estimators
    params = {
        'boosting_type':'gbdt',
        'objective':'binary',
        'metrics':'auc',
        'learning_rate':.1,
        'max_depth':5,
        'bagging_fraction':.8,
        'feature_fraction':.8
    }
    param_grid = {'n_estimators': [100, 150, 175, 200, 225, 250]}
    grid_search_helper(train_data, train_target, params, param_grid)

    # %%
    params['n_estimators'] = 200
    param_grid = {
        # 树的最大深度，默认值为-1，表示不做限制，合理的设置可以防止过拟合
        'max_depth': range(4, 8, 1),
        # 指定叶子的个数，默认值为31，此参数的数值应该小于 2 m a x _ d e p t h 2^{max\_depth}2 max_depth
        'num_leaves': range(10, 150, 10)
    }
    grid_search_helper(train_data, train_target, params, param_grid)

    #step3：确定min_data_in_leaf和max_bin in
    params['max_depth'], params['num_leaves'] = 6, 40
    param_grid = {
        'max_bin': range(100, 500, 100),
        # min_child_samples，它的值取决于训练数据的样本个树和num_leaves. 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合
        'min_data_in_leaf': range(100, 150, 50)
    }
    grid_search_helper(train_data, train_target, params, param_grid)

    #第四步：确定feature_fraction、bagging_fraction、bagging_freq
    params['max_bin'], params['min_data_in_leaf'] = 400, 120
    param_grid = {
        # 构建弱学习器时，对特征随机采样的比例，默认值为1
        'feature_fraction': [.6, .7, .8, .9, 1],
        # 默认值1，指定采样出 subsample * n_samples 个样本用于训练弱学习器, 不放回抽样
        'bagging_fraction': [.6, .7, .8, .9, 1],
        # 默认值0，表示禁用样本采样。如果设置为整数 z ，则每迭代 k 次执行一次采样
        'bagging_freq': range(0, 10, 2)
    }
    grid_search_helper(train_data, train_target, params, param_grid)

    #第五步：确定lambda_l1和lambda_l2
    params['feature_fraction'], params['bagging_fraction'], params['bagging_freq'] = .6, .9, 4
    param_grid = {
        # L1正则化权重项，增加此值将使模型更加保守。
        'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        # L2正则化权重项，增加此值将使模型更加保守。
        'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }
    grid_search_helper(train_data, train_target, params, param_grid)

    #第六步：确定 min_split_gain
    params['lambda_l1'], params['lambda_l2'] = 1e-5, 1e-5
    param_grid = {
        # 指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。
        'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    grid_search_helper(train_data, train_target, params, param_grid)

    # 第七步：降低学习率，增加迭代次数，验证模型
    params['min_split_gain'] = 0
    param_grid = {
        'learning_rate': [.1, .05, .01, .005]
    }
    grid_search_helper(train_data, train_target, params, param_grid)

    print(f'最优参数: {params}')

    # %%

    train = train_f3.copy()
    score_default = classifier_df_score(train, 'LGB', 5)
    score_fine_tune = classifier_df_score(train, 'LGB', 5, params)

    print(f"Fine tune: {score_default:.3f} -> {score_fine_tune:.3f}")

    # %%
    """绘制验证曲线"""
    #对逻辑回归的max_iter情况进行查看
    train_feat = train_f3.copy()
    grid_plot(train_feat, "LGB", 3, 'n_estimators', [10,20,40,80,200,400,800], param=params)
    # grid_plot(train_feat, 'LR', 3,  'max_iter', [1, 2, 5, 10, 20, 40, 50], param=None)


    # %%
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': 0.6,
        'num_leaves': 50,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8
    }
    train_feat = train_f3.copy()
    #grid_plot(train_feat,classifier,3,[10,20,40,80,200,400,800],'n_estimators',param=params)
    grid_plot(train_feat, 'LGB', 3, 'n_estimators', [10, 20, 40], param=params)
    
    
    # %%
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 10
    }
    train_feat = train_f3.copy()
    #grid_plot(train_feat,classifier,3,[10,20,40,80,200,400,800],'n_estimators',param=params)
    grid_plot(train_feat,
            'LGB',
            3, 
            'colsample_bytree',
            [0.1, 0.2, 0.5, 0.7, 0.8],
            param=params)

    # %%
