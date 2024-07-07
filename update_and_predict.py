import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from mlforecast import MLForecast
from xgboost import XGBRegressor
from utilsforecast.plotting import plot_series
from matplotlib import pyplot as plt
from window_ops.rolling import rolling_mean, rolling_std, rolling_min, rolling_max
from mlforecast.target_transforms import Differences
from sklearn.metrics import mean_squared_error, accuracy_score ,root_mean_squared_error
import numpy as np
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.seasonal import STL
from lightgbm import LGBMRegressor
from scipy.signal import periodogram

def update_and_predict(train,test,ftsc):
    frequencies, spectrum = periodogram(train['y'])
    max_idx = np.argmax(spectrum)
    second_max_idx = np.argsort(spectrum)[-2]  # Get the second largest index

    # Get the maximum and second maximum values
    max_value = spectrum[max_idx]
    second_max_value = spectrum[second_max_idx]

    # Get the corresponding frequencies
    max_freq = frequencies[max_idx]
    second_max_freq = frequencies[second_max_idx]

    1/max_freq, 1/second_max_freq

    # plt.figure(figsize=(14, 7))
    # plt.plot(frequencies, spectrum)
    # plt.title('Periodogram')
    # plt.xlabel('Frequency')
    # plt.ylabel('Power')
    # plt.show()

    seasonal_period=int(1/second_max_freq)
    stl = STL(train['y'],period=seasonal_period)
    result=stl.fit()
    trend=result.trend
    seasonal=result.seasonal
    a=pd.DataFrame(trend)
    a['ds']=train['ds']
    a['unique_id']=train['unique_id']
    a['y']=a['trend']
    a=a.dropna().reset_index(drop=True)
    # ftsc.ts.update(a)
    # ftsc.fit(a)


    predictions = ftsc.predict(test.shape[0])

    seasonalx = np.tile(seasonal[-seasonal_period:], int(np.ceil( test.shape[0] / seasonal_period)))[:test.shape[0]]
    seasonalx = pd.Series(seasonalx, index=test.index)
    seasonalx=pd.DataFrame(seasonalx)
    seasonalx['ds']=pd.to_datetime(test['ds'])
    seasonalx['y']=seasonalx[0]
    seasonalx['unique_id']='h1'
    seasonalx=seasonalx.reset_index()

    predictions['seas+xgb']=predictions['XGBRegressor']*(seasonalx['y'])
    predictions['seas+xgb']=(predictions['seas+xgb']/a['y'].mean())+a['y'].mean()
    predictions['seas+lgb']=predictions['LGBMRegressor']*(seasonalx['y'])
    predictions['seas+lgb']=(predictions['seas+lgb']/a['y'].mean())+a['y'].mean()

    rmse_xgb,rmse_lgb=root_mean_squared_error(test['y'],predictions['seas+xgb']),root_mean_squared_error(test['y'],predictions['seas+lgb'])
    return ftsc,rmse_xgb,rmse_lgb,predictions 