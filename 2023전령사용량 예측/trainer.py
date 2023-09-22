import pandas as pd
import numpy as np
import tqdm

from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sktime.forecasting.model_selection import temporal_train_test_split


def SMAPE(true, pred):
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) * 100

def train_validation(train):
    smape = 0

    for i in tqdm.tqdm(range(100)):
        ## 건물번호 별 발전량
        y = train.loc[train.building_number == i + 1, 'power_consumption']
        x = train.loc[train.building_number == i + 1,].drop(['power_consumption', 'building_number'], axis=1)
        y_train, y_valid, x_train, x_valid = temporal_train_test_split(y=y, X=x, test_size=168)

        # model = LGBMRegressor(boosting_type='gbdt',  # ['gbdt', 'dart', 'goss']
        #                       objective='regression',
        #                       n_estimators=10000,
        #                       max_depth=8,
        #                       learning_rate=0.03,
        #                       colsample_bytree=0.9,
        #                       subsample=0.7,
        #                       num_leaves=256,
        #                       reg_alpha=0.01,
        #                       reg_lambda=0.01,
        #                       n_jobs=-1, random_state=42)

        model = LGBMRegressor()
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                  callbacks=[early_stopping(stopping_rounds=300, verbose=False)])
        ## 주황색이 실제 전력소비량, 초록색이 예측값입니다.
        pred = model.predict(x_valid)
        pred = pd.Series(pred)
        pred.index = np.arange(y_valid.index[0], y_valid.index[-1] + 1)
        # plot_series(y_train, y_valid, pd.Series(pred), markers=[',' , ',', ','])
        # plt.show()
        print('{}번 건물 SMAPE : {}'.format(i+1, SMAPE(y_valid, pred)))
        smape += SMAPE(y_valid, pred)
    return smape

def train_inference(train, test):
    preds = np.array([])
    for i in tqdm.tqdm(range(100)):

        pred_df = pd.DataFrame()   # 시드별 예측값을 담을 data frame

        for seed in [0 ,1 ,2 ,3 ,4 ,5]: # 각 시드별 예측
            y_train = train.loc[train.building_number == i + 1, 'power_consumption']
            x_train = train.loc[train.building_number == i + 1,].drop(['power_consumption', 'building_number'], axis=1)
            x_test = test.loc[test.building_number == i + 1].drop(['building_number'], axis=1)
            x_test = x_test[x_train.columns]

            # model = LGBMRegressor(seed = seed,learning_rate = rgbm_param_df.iloc[i,0], n_estimators = rgbm_param_df.iloc[i,1], max_depth = rgbm_param_df.iloc[i,2],
            #                       num_leaves = rgbm_param_df.iloc[i,3], colsample_bytree = rgbm_param_df.iloc[i,4],subsample= rgbm_param_df.iloc[i,5],
            #                       subsample_freq= rgbm_param_df.iloc[i,6],min_child_samples = rgbm_param_df.iloc[i,7])
            # model = LGBMRegressor(seed = seed)
            # model.fit(x_train, y_train)
            model = LGBMRegressor()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            pred_df.loc[:, seed] = y_pred  # 각 시드별 예측 담기

        pred = pred_df.median(axis=1)  # (i+1)번째 건물의 예측 =  (i+1)번째 건물의 각 시드별 예측 평균값
        preds = np.append(preds, pred)

    return preds

