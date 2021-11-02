#%%
from _04_tarin import *

pred_result_col = 'pred'


#预测方式：按照购买概率进行预测
def proba_predict(model, df):
    pred = model.predict_proba(df)
    return pred[:, 1]


#预测，
def classifier_pred(traindf, classifier, param=None):
    model = get_sklearn_model(classifier, param)
    if classifier in ['LGB']:
        model.fit(get_predictors_df(traindf),
                  get_target_df(traindf),
                  eval_metric=myeval)
    if classifier in ['XGB']:
        model.fit(get_predictors_df(traindf),
                  get_target_df(traindf),
                  eval_metric='auc')
    else:
        model.fit(get_predictors_df(traindf), get_target_df(traindf))
    return model


#不分折进行预测
def fit_once(train_feat, test_feat, classifier, param=None):
    model = classifier_pred(train_feat, classifier, param)
    predicted = pd.DataFrame(proba_predict(model,
                                           get_predictors_df(test_feat)))
    return predicted, get_target_df(train_feat)


#分折进行预测
def fit_cv(train_feat, test_feat, classifier, cvnum, param=None):
    print('开始CV ' + str(cvnum) + '折训练...')
    train_preds = np.zeros(train_feat.shape[0])
    test_preds = np.zeros((test_feat.shape[0], cvnum))
    i = 0
    kf = StratifiedKFold(n_splits=cvnum, shuffle=True, random_state=520)
    for train_index, test_index in kf.split(get_predictors_df(train_feat),
                                            get_target_df(train_feat)):
        print('第{}次训练...'.format(i + 1))
        train_feat1 = train_feat.iloc[train_index]
        train_feat2 = train_feat.iloc[test_index]
        model = classifier_pred(train_feat1, classifier, param)

        train_preds[test_index] += proba_predict(
            model, get_predictors_df(train_feat2))
        test_preds[:, i] = proba_predict(model, get_predictors_df(test_feat))
        i = i + 1


#    print('CV训练用时{}秒'.format(time.time() - t0))
    test_y = test_preds.mean(axis=1)
    #test_y_1 = pd.Series(test_y).apply(lambda x : 1 if x>0.5 else 0)
    #submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
    return pd.DataFrame(test_y), pd.DataFrame(train_preds)


def classifier_df(train_feat, test_feat, classifier, cvnum, param=None):
    if cvnum <= 1:
        predicted, train_preds = fit_once(train_feat, test_feat, classifier,
                                          param)
    else:
        predicted, train_preds = fit_cv(train_feat, test_feat, classifier,
                                        cvnum, param)
    print('output')
    #predicted=predicted.round(3)
    return predicted, train_preds


#输出结果
def output_predicted(predicted, resultfile, test_feat):
    predicted = round(predicted, 3)
    resultdf = get_id_df(test_feat).copy()
    resultdf['Probability'] = predicted
    resultdf.to_csv(resultfile, header=False, index=False, sep=',')


#预测函数
def classifier_df_simple(train_feat, test_feat, classifier, param=None):
    model = get_sklearn_model(classifier, param)
    model.fit(get_predictors_df(train_feat), get_target_df(train_feat))
    predicted = pd.DataFrame(
        model.predict_proba(get_predictors_df(test_feat))[:, 1])
    return predicted



"""最终输出函数"""
def classifier_single(featurename, classifier, cvnum, param=None):
    """只运行一种算法

    Args:
        featurename ([type]): [description]
        classifier ([type]): [description]
        cvnum ([type]): [description]
        param ([type], optional): [description]. Defaults to None.
    """
    traindf, testdf = read_data(featurename, True)
    predicted, train_preds = classifier_df(traindf, testdf, classifier, cvnum, param)

    if cvnum > 1:
        traindf[pred_result_col] = train_preds
        score = myauc(traindf)
        print('线下成绩：    {}'.format(score))
        resultfile = resultpath + featurename + '_' + str(cvnum) + '_' + classifier + '_' + format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '_' + str(round(score, 3)) + '.csv'
    else:
        resultfile = resultpath + featurename + '_' + str(cvnum) + '_' + classifier + '_' + format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv'
        
    output_predicted(predicted, resultfile, testdf)


# 平均融合classifier_multi的一种融合方式，可以写多个类似的,作为classifier_multi的参数
def multi_mean(train_multi, test_multi, pred_names):
    i = 0
    for pred in pred_names:
        i = i + 1
        if i == 1:
            train_multi[pred_result_col] = train_multi[pred]
            test_multi[pred_result_col] = test_multi[pred]
        else:
            train_multi[pred_result_col] = train_multi[
                pred_result_col] + train_multi[pred]
            test_multi[pred_result_col] = train_multi[
                pred_result_col] + test_multi[pred]
    
    train_multi[pred_result_col] = train_multi[pred_result_col] / i
    test_multi[pred_result_col] = test_multi[pred_result_col] / i
    
    return train_multi, test_multi


#运行多种算法
#sum_func为对多种算法结果的整合函数，要求最终的输出列为'pred'
def classifier_multi(featurename, classifiers, cvnum, sum_func, param=None):
    traindf, testdf = read_data(featurename, True)
    train_multi = traindf.copy()
    test_multi = testdf.copy()

    notes = ''
    pred_names = []

    for classifier in classifiers:
        print('开始' + classifier + '训练')
        notes = notes + '-' + classifier
        pred_names.append(classifier + '_pred')
        predicted, train_preds = classifier_df(traindf, testdf, classifier, cvnum, param)
        train_multi[classifier + '_pred'] = train_preds
        test_multi[classifier + '_pred'] = predicted

    train_result, test_result = sum_func(train_multi, test_multi, pred_names)

    #score = metrics.roc_auc_score(get_target_df(train_result),train_result['pred'])
    #print('线下得分：    {}'.format(score))

    score = myauc(train_result)
    print('线下成绩：    {}'.format(score))

    if cvnum > 1:
        resultfile = resultpath + featurename + '_' + str(
            cvnum) + '_' + notes + '_' + format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '_' + str(
                    round(score, 3)) + '.csv'
    else:
        resultfile = resultpath + featurename + '_' + str(
            cvnum) + '_' + notes + '_' + format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv'

    output_predicted(test_result['pred'], resultfile, test_result)


#按满减情况分别预测
def classifier_single_sep_fd(featurename, classifier, cvnum, param=None):
    trainalldf, testalldf = read_data(featurename, True)
    test_result = pd.DataFrame()
    train_result = pd.DataFrame()
    #按满减情况分类
    for fd in range(0, 2):
        traindf = trainalldf[trainalldf.if_fd == fd].copy()
        testdf = testalldf[testalldf.if_fd == fd].copy()

        predicted, train_preds = classifier_df(traindf, testdf, classifier,
                                               cvnum, param)
        predicted = round(predicted, 3)

        if fd == 0:
            test_result = get_id_df(testdf).copy().reset_index(drop=True)
            test_result['pred'] = predicted

            train_result = traindf.copy().reset_index(drop=True)
            train_result['pred'] = train_preds

        else:
            dft1 = get_id_df(testdf).copy().reset_index(drop=True)
            dft1['pred'] = predicted
            test_result = pd.concat([test_result, dft1],
                                    axis=0).reset_index(drop=True)

            dfv1 = traindf.copy().reset_index(drop=True)
            dfv1['pred'] = train_preds
            train_result = pd.concat([train_result, dfv1],
                                     axis=0).reset_index(drop=True)

    if cvnum > 1:
        #score = metrics.roc_auc_score(get_target_df(train_result),train_result['pred'])
        score = round(myauc(train_result), 3)
        print('线下得分：    {}'.format(score))
        resultfile = resultpath + featurename + '_sepfd_' + str(
            cvnum) + '_' + classifier + '_' + str(score) + '.csv'
    else:
        resultfile = resultpath + featurename + '_sepfd_' + str(
            cvnum) + '_' + classifier + '.csv'
    test_result.to_csv(resultfile, header=False, index=False, sep=',')
    
#%%
classifier_single('sf2', 'LGB', 5)

# %%
classifier_single('sf3', 'LGB', 5)


# %%
# 采用f3版本特征，LightGBM，5折，优化后参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'eval_metric': 'auc',
    'n_estimators': 200,
    'max_depth': 5,
    'num_leaves': 40,
    'max_bin': 400,
    'min_data_in_leaf': 120,
    'learning_rate': 0.05,
    'lambda_l1': 1e-05,
    'lambda_l2': 1e-05,
    'min_split_gain': 0.0,
    'bagging_freq': 4,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.6,
    'seed': 1024,
    'n_thread': 12
}
classifier_single('sf3', 'LGB', 5, params)

# %%
# 采用f3版本特征，LightGBM+XGBoost融合，5折，默认参数
classifier_multi('sf3', ['XGB', 'LGB'], 5, multi_mean)

# %%
# 采用f3版本特征，LightGBM，5折，默认参数，根据是否为满减分别训练
classifier_single_sep_fd('sf3', 'LGB', 5)

# %%
# 采用f3版本特征，LightGBM，5折，优化后参数，根据是否为满减分别训练
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'eval_metric': 'auc',
    'n_estimators': 200,
    'max_depth': 5,
    'num_leaves': 40,
    'max_bin': 400,
    'min_data_in_leaf': 120,
    'learning_rate': 0.05,
    'lambda_l1': 1e-05,
    'lambda_l2': 1e-05,
    'min_split_gain': 0.0,
    'bagging_freq': 4,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.6,
    'seed': 1024,
    # 'n_thread': 12
}

classifier_single_sep_fd('sf3', 'LGB', 5, params)

# %%
# 绘制学习曲线
traindf, testdf = read_data('f1', True)
plot_curve_single(traindf, 'LGB', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])

#%%
# 参数调优
grid_plot_single('sf3', 'LGB', 3, [0.1, 0.2, 0.5, 0.7, 0.8],'colsample_bytree')

