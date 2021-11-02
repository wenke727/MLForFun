fd_seperator = ':'

# folder
datapath     = '../data/'
featurepath  = './feature/'
resultpath   = './result/'
tmppath      = './tmp/'
scorepath    = './score/'


# 全局参数
id_col_names    = ['user_id', 'coupon_id', 'date_received']
id_target_cols  = ['user_id', 'coupon_id', 'date_received', 'label']
target_col_name = 'label'
myeval          = 'roc_auc'
cvscore         = 0