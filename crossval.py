import pandas as pd
from mlforecast import MLForecast
from xgboost import XGBRegressor
from utilsforecast.plotting import plot_series
from matplotlib import pyplot as plt
from window_ops.rolling import rolling_mean, rolling_std, rolling_min, rolling_max
from mlforecast.target_transforms import Differences
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.seasonal import STL

def crossval(df1,seasonal_period):
    trainbef=df1[:5340]
    testbef=df1[5340:]
    trainbef.shape,testbef.shape
    stl = STL(df1['y'],period=seasonal_period)
    result=stl.fit()
    result.plot()
    trend=result.trend
    trend.head()
    seasonal=result.seasonal
    a=pd.DataFrame(trend)
    a.shape
    a['ds']=df1['ds']
    a['unique_id']=df1['unique_id']
    a['y']=a['trend']
    a=a.dropna().reset_index(drop=True)
    a.head()
    a.shape
    train=a[:5340]
    test=a[5340:]
    train.shape,test.shape
    model=XGBRegressor(random_state=990,learning_rate=0.01,n_estimators=500,max_depth=10,reg_lambda=0.2)
    ftsc=MLForecast(
        models={
            'xgb':model,
            # 'prophet':ProphetWrapper(),
        },
        freq='H',
        lags=[168,30],
        lag_transforms={
            30:[(rolling_mean,30),(rolling_std,30),(rolling_min,30),(rolling_max,30)],
            168:[(rolling_mean,168),(rolling_std,168),(rolling_min,168),(rolling_max,168)],
        #     350:[(rolling_mean,350),(rolling_std,350),(rolling_min,350),(rolling_max,350)],
        #     720:[(rolling_mean,720),(rolling_std,720),(rolling_min,720),(rolling_max,720)]
        },
        target_transforms=[Differences([350])],
        date_features=['hour','day','weekday','month','year'],
    )
    ftsc.fit(train,id_col='unique_id',time_col='ds',target_col='y')
    # min_child_weight=
    # ,gamma=
    # ,subsample=
    # objective="reg:squarederror",
    preds=ftsc.predict(test.shape[0])
    preds.head()
    seasonalx = np.tile(seasonal[-seasonal_period:], int(np.ceil( 1336 / seasonal_period)))[:1336]
    seasonalx = pd.Series(seasonalx, index=test.index)
    seasonalx=pd.DataFrame(seasonalx)
    seasonalx['ds']=pd.to_datetime(test['ds'])
    seasonalx['z']=seasonalx[0]
    seasonalx['unique_id']='h1'
    seasonalx.head()
    seasonalx=pd.DataFrame(seasonalx)
    seasonalx['y']=seasonalx['z']
    seasonalx=seasonalx.reset_index()
    seasonalx.head()
    seasonalx.shape,preds.shape
    plot_series(train,preds,max_insample_length=1000)
    plot_series(train,test,max_insample_length=1000)
    mse_test = mean_squared_error(test['y'], preds['xgb'])
    mse_test
    preds['seas+xgb']=preds['xgb']*(seasonalx['y'])
    preds['seas+xgb']=(preds['seas+xgb']/df1['y'].mean())+df1['y'].mean()
    p=preds.copy()
    p.head()
    p.drop(columns=['xgb'],inplace=True)
    plot_series(trainbef,p,max_insample_length=1000)
    plot_series(trainbef,testbef,max_insample_length=1000)