# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from keras_model import KerasModel
import random

random.seed(666)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys

sys.path.append('/home/michael/lightgbm/py/pyLightGBM')
from pylightgbm.models import GBMClassifier

path_to_exec = '/home/michael/lightgbm/LightGBM/lightgbm'


def loss_func(y_true, y_pred):
    return log_loss(y_true, y_pred)


all_train = pd.read_csv('x_train.csv', sep=';')
all_target = pd.read_csv('y_train.csv', sep=';', names=['TARGET'])
all_train['TARGET'] = all_target['TARGET']
submiss = pd.read_csv('x_test.csv', sep=';')

cols_to_drop = ['ID', 'TARGET']
cols = list(set(all_train.columns) - set(cols_to_drop))
base_cols = cols

# определи группы одинаковых строк
all_train['row_id'] = all_train[base_cols].apply(lambda row: '_'.join([str(i) for i in row]), axis=1)
submiss['row_id'] = submiss[base_cols].apply(lambda row: '_'.join([str(i) for i in row]), axis=1)

gb = all_train.groupby(['row_id'], as_index=False).size()
gb.name = 'size'
gb = gb.reset_index()
sizdata = gb[gb['size'] > 50].sort_values('size', ascending=False)

similar_data = all_train[all_train['row_id'].isin(sizdata['row_id'].values)]


# генерируем признаки
def transform_data(data):
    for i1, col1 in enumerate(base_cols):
        data[col1 + '_log'] = np.log(data[col1] + 1.1)

        for i2, col2 in enumerate(base_cols):
            data['%s_%s_1' % (col1, col2)] = data[col1] - data[col2]
            data['%s_%s_2' % (col1, col2)] = data[col1] + data[col2]
            data['%s_%s_3' % (col1, col2)] = data[col1] / (data[col2] + 0.1)
            data['%s_%s_4' % (col1, col2)] = data[col1] * data[col2]

            data['%s_%s_11' % (col1, col2)] = data[col1] - np.log(data[col2] + 1)
            data['%s_%s_22' % (col1, col2)] = data[col1] + np.log(data[col2] + 1)
            data['%s_%s_33' % (col1, col2)] = data[col1] / (np.log(data[col2] + 1) + 0.1)
            data['%s_%s_44' % (col1, col2)] = data[col1] * np.log(data[col2] + 1)

    return data


all_train = transform_data(all_train)
submiss = transform_data(submiss)

# список колонок вынес в отдельный файл
from cols_description import *

# выборки для разных моделей
X_train = all_train[cols].values
X_train_nn = all_train[cols_k2].values
X_train_xgb2 = all_train[xgb2_cols].values

scaler_reg = MinMaxScaler((-1, 1))
scaler_reg.fit(np.vstack((all_train[reg_cols], submiss[reg_cols])))
X_train_reg = scaler_reg.transform(all_train[reg_cols])
submiss_reg = scaler_reg.transform(submiss[reg_cols])

y_train = all_train['TARGET'].values



# ------------------------------------------------------------------
params = {
    'silent': 1,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.01,
    'subsample': 0.4,
    'min_child_weight': 7,
    'n': 580,
}

dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)

bst1 = xgb.train(params, dtrain, params['n'])
# ------------------------------------------------------------------
params = {
    'exec_path': path_to_exec,
    'num_iterations': 108,
    'learning_rate': 0.079,
    'num_leaves': 13,
    'metric': 'binary_error',
    'min_sum_hessian_in_leaf': 1,
    'bagging_fraction': 0.642,
    'bagging_freq': 1,
    'verbose': 0
}

bst2 = GBMClassifier(boosting_type='gbdt', **params)
bst2.fit(X_train, y_train)
# ------------------------------------------------------------------
params_est = {
    'n_estimators': 300,
    'loss': 'exponential',
    'learning_rate': 0.08,
    'subsample': 0.6910000000000001,
    'min_samples_leaf': 340,
    'max_features': 53,
    'random_state': 1
}
bst3 = GradientBoostingClassifier(**params_est)
bst3.fit(X_train, y_train)
# ------------------------------------------------------------------
from keras.callbacks import Callback as keras_clb
random.seed(666)
np.random.seed(666)

class LearningRateClb(keras_clb):
    def on_epoch_end(self, epoch, logs={}):
        if epoch ==300:
            self.model.optimizer.lr.set_value(0.01)


bst4 = KerasModel(cols_k2,600)
bst4.fit_process(X_train_nn, y_train)
bst4.fit(X_train_nn, y_train,
         callbacks=[LearningRateClb()]
)
# ------------------------------------------------------------------
bst5 = LogisticRegression()
bst5.fit(X_train_reg, y_train)
# ------------------------------------------------------------------
params = {
    'silent': 1,
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.01,
    'subsample': 0.65,
    'colsample_bytree': 0.3,
    'min_child_weight': 5,
    'n': 1140,
}


dtrain = xgb.DMatrix(X_train_xgb2, label=y_train, missing=np.NaN)

bst6 = xgb.train(params, dtrain, params['n'])
# ------------------------------------------------------------------
params_est = {
    'n_estimators': 200,
    'loss': 'deviance',
    'learning_rate': 0.04,
    'subsample': 0.50,
    'min_samples_leaf': 60,
    'max_features': 4,
    'random_state': 1
}
bst7 = GradientBoostingClassifier(**params_est)
bst7.fit(X_train_xgb2, y_train)

sub_data = submiss[cols].values
submiss_val = xgb.DMatrix(sub_data, missing=np.NaN)
t1 = np.asarray([[i] for i in bst1.predict(submiss_val)])
t2 = np.asarray([[i] for i in  bst2.predict_proba(submiss[cols].values)[:,1]])
t3 = np.asarray([[i] for i in  bst3.predict_proba(submiss[cols].values)[:,1]])
t4 = np.asarray(bst4.predict_proba(submiss[cols_k2].values))
t5 = np.asarray([[i] for i in  bst5.predict_proba(submiss_reg)[:,1]])
t6 = np.asarray([[i] for i in  bst6.predict(xgb.DMatrix(submiss[xgb2_cols].values,missing=np.NaN))])
t7 = np.asarray([[i] for i in  bst7.predict_proba(submiss[xgb2_cols].values)[:,1]])

tst_data = np.hstack((
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
))



itog = np.mean(tst_data , axis=1)

submiss['TARGET'] = itog
# для часто повторяющихся элементов возьмем среднее
for r in range(len(sizdata.index)):
    ind1 = (similar_data['row_id'] == sizdata.ix[sizdata.index[r], 'row_id'])
    ind2 = (submiss['row_id'] == sizdata.ix[sizdata.index[r], 'row_id'])
    submiss.ix[ind2, 'TARGET'] = similar_data.ix[ind1, 'TARGET'].mean()

sub_df = pd.DataFrame(data=submiss['TARGET'])
sub_df.to_csv('itog.csv', index=False, header=False)
