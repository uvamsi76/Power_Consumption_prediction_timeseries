import warnings

# Ignore all warnings
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


df=pd.read_csv('MLTempDataset.csv')
df.drop(columns=['Unnamed: 0',],inplace=True)#'Datetime1'
df1=df.copy()
df1['unique_id']='h1'
# df1['unique_id']=
df1['ds']=pd.to_datetime(df['Datetime'])
df1['y']=df['DAYTON_MW']
df1.drop(columns=['Datetime','DAYTON_MW','Datetime1'],inplace=True)
df1.head()

chunks=[]
r=int(df1.shape[0]/5)
for i in range(4):
    a=df1[i*r:(i+1)*r]
    chunks.append(a)
chunks.append(df1[4*r:])


initial_train=chunks[0]

frequencies, spectrum = periodogram(initial_train['y'])
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
seasonal_period

model=XGBRegressor(random_state=990,learning_rate=0.01,n_estimators=500,max_depth=10,reg_lambda=0.2)
model2=LGBMRegressor(boosting_type='gbdt',num_leaves=31,learning_rate=0.1,n_estimators=100,random_state=990,verbose=-1)
ftsc=MLForecast(
    models=[model,model2],
    freq='H',
    lags=[seasonal_period],
    lag_transforms={
        # 30:[(rolling_mean,30),(rolling_std,30),(rolling_min,30),(rolling_max,30)],
        # 168:[(rolling_mean,168),(rolling_std,168),(rolling_min,168),(rolling_max,168)],
        seasonal_period:[(rolling_mean,seasonal_period),(rolling_std,seasonal_period),(rolling_min,seasonal_period),(rolling_max,seasonal_period)],
    #     350:[(rolling_mean,350),(rolling_std,350),(rolling_min,350),(rolling_max,350)],
    #     720:[(rolling_mean,720),(rolling_std,720),(rolling_min,720),(rolling_max,720)]
    },
    # target_transforms=[Differences([350])],
    date_features=['hour','day','weekday','month','year'],
)


ftsc.fit(initial_train)

def update_and_predict(train,test,ftsc):
    stl = STL(train['y'],period=seasonal_period)
    result=stl.fit()
    trend=result.trend
    seasonal=result.seasonal
    a=pd.DataFrame(trend)
    a['ds']=df1['ds']
    a['unique_id']=df1['unique_id']
    a['y']=a['trend']
    a=a.dropna().reset_index(drop=True)
    ftsc.ts.update(a)
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