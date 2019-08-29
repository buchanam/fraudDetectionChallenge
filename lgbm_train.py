import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

raw_train = pd.read_csv('data/train_transaction.csv', delimiter=',', encoding="utf-8-sig")

df_train, df_test = train_test_split(raw_train, test_size=0.2)

#print(df_train['isFraud'])

print('Done loading data...')

y_train = df_train['isFraud']
y_test = df_test['isFraud']
        
x_train = df_train.drop(columns='isFraud')
x_test = df_test.drop(columns='isFraud')

x_train = pd.get_dummies(x_train, columns =['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'])

x_test = pd.get_dummies(x_test, columns=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'])

print(y_test)

lgb_train = lgb.Dataset(x_d, y_train)
lgb_eval = lgb.Dataset(x_t, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting Training')

#gbm = lgb.train(params,
 #               lgb_train,
  #              num_boost_round=20,
   #             valid_sets=lgb_eval,
    #            early_stopping_rounds=5)

lgmodel = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    learning_rate=0.01,
    colsample_bytree=0.9,
    subsample=0.8,
    random_state=1,
    n_estimators=100,
    num_leaves=31,
    silent=False
    )

lgmodel.fit(x_train, y_train)

y_pred = lgmodel.predict(x_test)

accuracy=accuracy_score(y_pred, y_test)

print('The accuracy is: ', accuracy)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
