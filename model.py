import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesForecasting:
    """Forecast time series values by chosen model
    Init Parameters
    ----------
    df_y : pandas dataframe containing two columns: [ds, y]
        historical time series input data
    act_st : datetime
        start date of input data
    fcst_st : datetime format
        forecast date
    fcst_pr : int
        forecast period
    fcst_freq : {"d", "m", "q", "y"}, optional
        forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
    df_x : pandas dataframe containing three columns: [id, ds, x], default=None
        time series of external features
        if not provided, some models return blank result
    df_lag : pandas dataframe containing two columns: [id, lag], default=None
        length of lagging from external features
    col_ds: str, default='ds'
        name of col ds (datestamp)
    col_y: str, default='y'
        name of col y
    col_idx: str, default='id'
        name of col external id in df_x and lag
    col_x: str, default='x'
        name of col x
    col_lag: str, default='lag'
        name of col lag
    """

    freq_dict = {'d': 'D', 'm': 'MS', 'q': 'QS', 'y': 'YS'}
    freq_period = {'d': 365, 'm': 12, 'q': 4, 'y': 1}

    def __init__(self, df_y, act_st, fcst_st, fcst_pr, fcst_freq='m', df_x=None, df_lag=None, col_ds='ds', col_y='y', col_idx='id', col_x='x', col_lag='lag'):
        self.act_st = datetime.datetime.combine(act_st, datetime.datetime.min.time())
        self.fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        self.df_y = df_y.rename(columns={col_ds: 'ds', col_y: 'y'})
        self.fcst_pr = fcst_pr
        self.fcst_freq = fcst_freq
        self.fcst_freq_txt = self.freq_dict[fcst_freq]
        self.fcst_freq_pr = self.freq_period[fcst_freq]
        self.dt = pd.date_range(start=self.fcst_st, periods=self.fcst_pr, freq=self.fcst_freq_txt)
        if self.act_st < self.fcst_st:
            self.df_y = self.df_y[(self.df_y['ds']>=self.act_st) & (self.df_y['ds']<self.fcst_st)]
            self.df_d = self.filldaily(self.df_y, self.act_st, self.fcst_st + datetime.timedelta(days=-1))
            self.df_act = self.df_d.resample(self.fcst_freq_txt, on='ds').agg({'y':'sum'}).reset_index()
            self.df_x = df_x
            self.df_lag = df_lag
            if df_x is not None:
                self.df_x = df_x.rename(columns={col_idx: 'id', col_ds: 'ds', col_x: 'x'})
                self.df_x = self.df_x[self.df_x['ds']<self.fcst_st]
                self.df_x = self.df_x.groupby('id').resample(self.fcst_freq_txt, on='ds').sum().reset_index()
                df_lag = df_lag.rename(columns={col_idx: 'id', col_lag: 'lag'})
                self.x_lag = df_lag.set_index('id')['lag'].to_dict()

    @staticmethod
    def filldaily(df, start, end, col_ds='ds', col_y='y'):
        """Fill time series dataframe for all dates"""
        if start > end:
            return pd.DataFrame(columns = [col_ds, col_y])
        df = df.append(pd.DataFrame(data={col_ds: [start, end], col_y: [0, 0]})).reset_index(drop=True)
        df = df.resample('D', on=col_ds).agg({col_y:'sum'}).reset_index()
        df = df[[col_ds, col_y]].sort_values(by=col_ds, ascending=True).reset_index(drop=True)
        return df

    @staticmethod
    def correctzero(df, col_y='y'):
        """Edit dataframe <0 to 0"""
        df[col_y] = df[col_y].apply(lambda x: 0 if x<0 else x)
        return df

    @classmethod
    def valtogr(cls, df, fcst_freq='m', mth_shift=12, col_ds='ds', col_y='y'):
        """Transform nominal values to growth"""
        df = df.copy()
        df['ds_shift'] = df[col_ds].apply(lambda x: x + relativedelta(months=-mth_shift))
        df = pd.merge(df, df[[col_ds, col_y]].rename(columns={col_ds: 'ds_shift', col_y: 'y_shift'}), how='left', on='ds_shift')
        df['gr'] = (df[col_y] - df['y_shift']) / df['y_shift']
        df.loc[cls.freq_period[fcst_freq]:, 'gr'] = df.loc[cls.freq_period[fcst_freq]:, 'gr'].replace([np.inf, -np.inf, None], [1, -1, 1])
        return list(df['gr'])

    @classmethod
    def grtoval(cls, df, df_act, mth_shift=12, col_ds='ds', col_y='y', col_yact='y'):
        """Transform nominal growth to values"""
        df = df.copy()
        df_act = df_act.copy()
        # map actual data
        df_act['y_shift'] = df_act[col_yact]
        df_act['ds_shift'] = df_act[col_ds].apply(lambda x: x + relativedelta(months=+mth_shift))
        dict_y = pd.Series(df_act['y_shift'].values, index=df_act['ds_shift']).to_dict()
        df['y_shift'] = df[col_ds].map(dict_y)
        # map predict data
        while df['y_shift'].isnull().any():
            df_pd = df.copy()
            df_pd['y_shift'] = (1 + df_pd[col_y]) * df_pd['y_shift']
            df_pd['ds_shift'] = df_pd[col_ds].apply(lambda x: x + relativedelta(months=+mth_shift))
            for i, r in df_pd.dropna().iterrows():
                dict_y[r['ds_shift']] = r['y_shift']
            df['y_shift'] = df[col_ds].map(dict_y)
        return list((1 + df[col_y]) * df['y_shift'])

    @staticmethod
    def deltafreq(x, freq='m'):
        """Add date by forecasting frequency"""
        if freq == 'd':
            return relativedelta(days=x)
        elif freq == 'm':
            return relativedelta(months=x)
        elif freq == 'q':
            return relativedelta(months=x * 3)
        elif freq == 'y':
            return relativedelta(years=x)

    # create feature
    def extractfeat(self, df, col, rnn_delay=3):
        """Extract features"""
        df = df.copy()
        df_act = self.df_act.copy()
        df_append = df[(df['ds'] < df_act['ds'].min()) | (df['ds'] > df_act['ds'].max())]
        df_act = df_act.append(df_append, ignore_index = True)
        df_act = self.filldaily(df_act, df_act['ds'].min(), df_act['ds'].max())
        df_act = df_act.resample(self.fcst_freq_txt, on='ds').agg({'y':'sum'}).reset_index()
        # date features
        df_act['day'] = pd.DatetimeIndex(df_act['ds']).day
        df_act['dayofyear'] = pd.DatetimeIndex(df_act['ds']).dayofyear
        df_act['weekofyear'] = pd.DatetimeIndex(df_act['ds']).weekofyear
        df_act['weekday'] = pd.DatetimeIndex(df_act['ds']).weekday
        df_act['month'] = pd.DatetimeIndex(df_act['ds']).month
        df_act['quarter'] = pd.DatetimeIndex(df_act['ds']).quarter
        df_act['year'] = pd.DatetimeIndex(df_act['ds']).year
        df_act = pd.get_dummies(df_act, columns = ['day'], drop_first = False)
        df_act = pd.get_dummies(df_act, columns = ['dayofyear'], drop_first = False)
        df_act = pd.get_dummies(df_act, columns = ['weekofyear'], drop_first = False)
        df_act = pd.get_dummies(df_act, columns = ['weekday'], drop_first = False)
        df_act = pd.get_dummies(df_act, columns = ['month'], drop_first = False)
        df_act = pd.get_dummies(df_act, columns = ['quarter'], drop_first = False)
        df_act = pd.get_dummies(df_act, columns = ['year'], drop_first = False)
        # last features
        df_act['last_period'] = df_act['y'].shift(1)
        df_act['last_year'] = df_act['y'].shift(self.fcst_freq_pr)
        df_act['last_momentum'] = (df_act['y'].shift(self.fcst_freq_pr) - df_act['y'].shift(self.fcst_freq_pr+1)) / df_act['y'].shift(self.fcst_freq_pr+1)
        df_act.loc[12:, 'last_momentum'] = df_act.loc[12:, 'last_momentum'].replace([np.inf, -np.inf, None], [1, -1, 1])
        df_act['gr'] = self.valtogr(df_act, self.fcst_freq, 12)
        df_act['lastgr_period'] = df_act['gr'].shift(1)
        df_act['lastgr_year'] = df_act['gr'].shift(self.fcst_freq_pr)
        # decomposed features
        df_act = df_act.set_index('ds')
        decomposition = seasonal_decompose(df_act['y'], model = 'additive', period = 4)
        df_act['dec_trend'] = decomposition.trend.shift(2)
        df_act['dec_seasonal'] = decomposition.seasonal
        df_act['dec_residual'] = decomposition.resid.shift(2)
        df_act = df_act.reset_index()
        # external feature
        if self.df_x is not None and len(self.x_lag) > 0:
            rnn_lag = self.df_x.groupby(['id'], as_index=False).agg({"ds":"max"})
            rnn_lag = rnn_lag.set_index('id')['ds'].to_dict()
            rnn_lag = {k: max(len(pd.date_range(start=v, end=self.fcst_st, freq=self.fcst_freq_txt))-1, rnn_delay) for k, v in rnn_lag.items()}
            for i in self.x_lag:
                # external features with external lag
                df_x = self.df_x[self.df_x['id']==i].copy()
                df_x['ds'] = df_x['ds'].apply(lambda x: x + self.deltafreq(self.x_lag[i], self.fcst_freq))
                x_col = 'x_{}'.format(i)
                df_x[x_col] = df_x['x']
                # external features with rnn lag
                df_rnn = self.df_x[self.df_x['id']==i].copy()
                df_rnn['ds'] = df_rnn['ds'].apply(lambda x: x + self.deltafreq(rnn_lag[i], self.fcst_freq))
                rnn_col = 'xrnn_{}'.format(i)
                df_rnn[rnn_col] = df_rnn['x']
                # merge with actual
                df_act = pd.merge(df_act, df_x[['ds', x_col]], how='left', on='ds')
                df_act = pd.merge(df_act, df_rnn[['ds', rnn_col]], how='left', on='ds')
        df_act = df_act[['ds', 'y'] + [x for x in df_act.columns if x.startswith(tuple(col))]]
        df = df_act[(df_act['ds'] >= df['ds'].min()) & (df_act['ds'] <= df['ds'].max())]
        df = df.sort_values(by='ds', ascending=True).reset_index(drop=True)
        return df

    def forecast(self, i, **kwargs):
        """Forecast function
        Parameters
        ----------
        i : {options}
            specify model to run
            options:
                expo01 - Single Exponential Smoothing (Simple Smoothing)
                expo02 - Double Exponential Smoothing (Holt’s Method)
                expo03 - Triple Exponential Smoothing (Holt-Winters’ Method)
                naive01 - Naive model
                snaive01 - Seasonal Naive model
                sma01 - Simple Moving Average (short n)
                sma02 - Simple Moving Average (middle n)
                sma03 - Simple Moving Average (long n)
                wma01 - Weighted Moving Average (short n)
                wma02 - Weighted Moving Average (middle n)
                wma03 - Weighted Moving Average (long n)
                ema01 - Exponential Moving Average (short n)
                ema02 - Exponential Moving Average (middle n)
                ema03 - Exponential Moving Average (long n)
                arima01 - ARIMA model with fixed parameter forecast nominal
                arima02 - ARIMA model with fixed parameter forecast growth
                arimax01 - ARIMA model with fixed parameter and external features forecast nominal
                arimax02 - ARIMA model with fixed parameter and external features forecast nominal
                autoarima01 - ARIMA model with optimal parameter forecast nominal
                autoarima01 - ARIMA model with optimal parameter and external features forecast growth
                autoarimax01 - ARIMA model with optimal parameter and external features forecast nominal
                autoarimax02 - ARIMA model with optimal parameter and external features forecast growth
                prophet01 - Prophet by Facebook forecast nominal
                linear01 - Linear Regression forecast nominal
                linear02 - Linear Regression forecast growth
                linearx01 - Linear Regression with external features forecast nominal
                linearx02 - Linear Regression with external features forecast growth
                randomforest01 - Random Forest forecast nominal
                randomforest02 - Random Forest forecast growth
                randomforestx01 - Random Forest with external features forecast nominal
                randomforestx01 - Random Forest with external features forecast growth
                xgboost01 - XGBoost forecast nominal
                xgboost02 - XGBoost forecast growth
                xgboostx01 - XGBoost with external features forecast nominal
                xgboostx01 - XGBoost with external features forecast growth    
                lstm01 - Long Short-Term Memory forecast nominal
                lstm02 - Long Short-Term Memory forecast growth
                lstmr01 - Long Short-Term Memory with rolling forecast forecast nominal
                lstmr02 - Long Short-Term Memory with rolling forecast forecast growth
                lstmx01 - Long Short-Term Memory with external features forecast nominal
                lstmx02 - Long Short-Term Memory with external features forecast growth                
        Returns
        -------
        result : dataframe (ds, y)
        """
        if self.act_st >= self.fcst_st:
            return pd.DataFrame(columns = ['ds', 'y'])
        fn = getattr(TimeSeriesForecasting, i)
        return fn(self, **kwargs)

    """ MODELS """
    # Exponential Smoothing model
    def expo(self, model):
        r = model.forecast(self.fcst_pr)
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        return r

    # Single Exponential Smoothing (Simple Smoothing)
    def expo01(self):
        # too few data points, return none
        if len(self.df_act) <= 1:
            return pd.DataFrame(columns = ['ds', 'y'])
        x = list(self.df_act['y'])
        m = SimpleExpSmoothing(x).fit(optimized=True)
        r = self.expo(m)
        return r

    # Double Exponential Smoothing (Holt’s Method)
    def expo02(self):
        # too few data points, return none
        if len(self.df_act) <= 1:
            return pd.DataFrame(columns = ['ds', 'y'])
        param = {'trend': 'add'}
        x = list(self.df_act['y'])
        m = ExponentialSmoothing(x, trend=param['trend']).fit(optimized=True)
        r = self.expo(m)
        return r

    # Triple Exponential Smoothing (Holt-Winters’ Method)
    def expo03(self):
        # too few data points, return none
        if len(self.df_act) < self.fcst_freq_pr * 2 or self.fcst_freq_pr <= 1:
            return pd.DataFrame(columns = ['ds', 'y'])
        param = {'trend': 'add', 'seasonal': 'add'}
        x = list(self.df_act['y'])
        m = ExponentialSmoothing(x, trend=param['trend'], seasonal=param['seasonal'], seasonal_periods=self.fcst_freq_pr).fit(optimized=True)
        r = self.expo(m)
        return r

    # Naive model
    def naive01(self):
        r = self.df_act.copy()
        r = r.sort_values(by='ds', ascending=True).reset_index(drop=True)
        r = [r['y'].iloc[-1]] * self.fcst_pr
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        return r

    # Seasonal Naive model
    def snaive01(self):
        # too few data points, return none
        if len(self.df_act) < self.fcst_freq_pr:
            return pd.DataFrame(columns = ['ds', 'y'])
        r = self.df_act.copy()
        r = r.sort_values(by='ds', ascending=True).reset_index(drop=True)
        n = self.fcst_freq_pr
        r = r['y'].iloc[-n:].tolist() * int(np.ceil(self.fcst_pr / n))
        r = r[:self.fcst_pr]
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        return r

    # Moving Average model
    def ma(self, param):
        r = self.df_act.copy()
        r = r.sort_values(by='ds', ascending=True).reset_index(drop=True)
        w = np.arange(1, param['n']+1)
        r['sma'] = r['y'].rolling(window=param['n'], min_periods=1).mean()
        r['wma'] = r['y'].rolling(window=param['n'], min_periods=1).apply(lambda x: np.dot(x, w[-len(x):])/w[-len(x):].sum(), raw=True)
        r['ema'] = r['y'].ewm(span=param['n'], min_periods=1).mean()
        r = [r[param['method']].iloc[-1]] * self.fcst_pr
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        return r

    # Simple Moving Average
    def sma01(self):
        n = {'d': 7, 'm': 3, 'q': 2, 'y': 2}
        param = {'method': 'sma', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    def sma02(self):
        n = {'d': 30, 'm': 6, 'q': 4, 'y': 5}
        param = {'method': 'sma', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    def sma03(self):
        n = {'d': 90, 'm': 12, 'q': 8, 'y': 10}
        param = {'method': 'sma', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    # Weighted Moving Average
    def wma01(self):
        n = {'d': 7, 'm': 3, 'q': 2, 'y': 2}
        param = {'method': 'wma', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    def wma02(self):
        n = {'d': 30, 'm': 6, 'q': 4, 'y': 5}
        param = {'method': 'wma', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    def wma03(self):
        n = {'d': 90, 'm': 12, 'q': 8, 'y': 10}
        param = {'method': 'wma', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    # Exponential Moving Average
    def ema01(self):
        n = {'d': 7, 'm': 3, 'q': 2, 'y': 2}
        param = {'method': 'ema', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    def ema02(self):
        n = {'d': 30, 'm': 6, 'q': 4, 'y': 5}
        param = {'method': 'ema', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    def ema03(self):
        n = {'d': 90, 'm': 12, 'q': 8, 'y': 10}
        param = {'method': 'ema', 'n': n[self.fcst_freq]}
        r = self.ma(param)
        return r

    # ARIMA
    def arima(self, gr, param):
        # input data
        df = self.df_act.copy()
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.fillna(0).reset_index(drop=True)
        # too few data points, return none
        if len(df) < 3:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare tranining data
        x = df['y'].values
        # fit model
        m = SARIMAX(x, order=(param['p'], param['d'], param['q']), initialization='approximate_diffuse')
        m = m.fit(disp = False)
        # forecast
        r = m.predict(start=df.index[-1] + 1, end=df.index[-1] + self.fcst_pr)
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_act) if gr else r['y']
        return r

    def arima01(self):
        param = {'p': 4, 'd': 1, 'q': 4}
        r = self.arima(gr=False, param=param)
        return r

    def arima02(self):
        param = {'p': 4, 'd': 0, 'q': 4}
        r = self.arima(gr=True, param=param)
        return r

    # ARIMAX 
    def arimax(self, gr, feat, param):
        # if no external features, no forecast result
        if self.df_x is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        # input monthly data
        df = self.df_act.copy()
        df = self.extractfeat(self.df_act, col=feat)
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        # clean data - drop null from growth calculation, fill 0 when no external data
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.fillna(0).reset_index(drop=True)
        # too few data points, return none
        if len(df) < 3:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data
        x = df['y'].values
        ex = df.iloc[:, 2:].values
        # fit model1 with external
        m1 = SARIMAX(x, exog=ex, order=(param['p'], param['d'], param['q']), initialization='approximate_diffuse')
        m1 = m1.fit(disp = False)
        # prepare external data
        df_pred = pd.DataFrame(columns = ['ds', 'y'])
        for i in self.dt:
            df_pred = df_pred.append({'ds' : i} , ignore_index=True)
            df_pred = self.extractfeat(df_pred, col=feat)
            if np.isnan(list(df_pred.iloc[-1, 2:].values)).any():
                df_pred = df_pred.iloc[:-1, :]
                break
        # forecast model1
        ex_pred = df_pred.iloc[:, 2:].values
        r1 = m1.predict(start=df.index[-1] + 1, end=df.index[-1] + ex_pred.shape[0], exog=ex_pred)
        # model2 (used when there is no external features in future prediction)
        if len(r1) < self.fcst_pr:
            # fit model2 without external
            m2 = SARIMAX(x, order=(param['p'], param['d'], param['q']), initialization='approximate_diffuse')
            m2 = m2.fit(disp = False)
            # forecast model2
            r2 = m2.predict(start=df.index[-1] + ex_pred.shape[0] + 1, end=df.index[-1] + self.fcst_pr)
        else:
            r2 = []
        # summarize result
        r = list(r1) + list(r2)
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_act) if gr else r['y']
        return r

    def arimax01(self):
        gr = False
        feat = ['x_']
        param = {'p': 4, 'd': 1, 'q': 4}
        r = self.arimax(gr, feat, param)
        return r

    def arimax02(self):
        gr = True
        feat = ['x_']
        param = {'p': 4, 'd': 1, 'q': 4}
        r = self.arimax(gr, feat, param)
        return r

    # Auto ARIMA
    def autoarima(self, gr, param):
        # input monthly data
        df = self.df_act.copy()
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.fillna(0).reset_index(drop=True)
        # too few data points, return none
        if len(df) < 5:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data
        x = df['y'].values
        # fit model
        try:
            m = pm.arima.AutoARIMA(start_p=param['start_p'], max_p=param['max_p'],
                                start_q=param['start_q'], max_q=param['max_q'], 
                                d=param['d'], 
                                m = param['m'], seasonal=param['seasonal'], 
                                trace=False, error_action='ignore', suppress_warnings=True, 
                                stepwise=param['stepwise'])
            m = m.fit(x)
        except Exception:
            m = pm.arima.AutoARIMA()
            m = m.fit(x)
        # forecast
        r = m.predict(n_periods=self.fcst_pr)
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_act) if gr else r['y']
        return r

    def autoarima01(self):
        gr = False
        max_pq = {'d': 10, 'm': 12, 'q': 4, 'y': 3}
        param = {'start_p': 1, 'max_p': max_pq[self.fcst_freq], 'start_q': 1, 'max_q': max_pq[self.fcst_freq], 'd': None, 
                 'm': self.fcst_freq_pr, 'seasonal': True, 'stepwise': True}
        r = self.autoarima(gr, param)
        return r

    def autoarima02(self):
        gr = True
        max_pq = {'d': 10, 'm': 12, 'q': 4, 'y': 3}
        param = {'start_p': 1, 'max_p': max_pq[self.fcst_freq], 'start_q': 1, 'max_q': max_pq[self.fcst_freq], 'd': None, 
                 'm': self.fcst_freq_pr, 'seasonal': True, 'stepwise': True}
        r = self.autoarima(gr, param) 
        return r

    # Auto ARIMAX
    def autoarimax(self, gr, feat, param):
        # if no external features, no forecast result
        if self.df_x is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        # input monthly data
        df = self.df_act.copy()
        df = self.extractfeat(self.df_act, col=feat)
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        # clean data - drop null from growth calculation, fill 0 when no external data
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.fillna(0).reset_index(drop=True)
        # too few data points, return none
        if len(df) < 3:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data
        x = df['y'].values
        ex = df.iloc[:, 2:].values
        # fit model1 with external
        try:
            m1 = pm.arima.AutoARIMA(start_p=param['start_p'], max_p=param['max_p'],
                                    start_q=param['start_q'], max_q=param['max_q'], 
                                    d=param['d'], 
                                    m = param['m'], seasonal=param['seasonal'], 
                                    trace=False, error_action='ignore', suppress_warnings=True, 
                                    stepwise=param['stepwise'])
            m1 = m1.fit(x, exogenous=ex)
        except Exception:
            m1 = pm.arima.AutoARIMA()
            m1 = m1.fit(x, exogenous=ex)
        # prepare external data
        df_pred = pd.DataFrame(columns = ['ds', 'y'])
        for i in self.dt:
            df_pred = df_pred.append({'ds' : i} , ignore_index=True)
            df_pred = self.extractfeat(df_pred, col=feat)
            if np.isnan(list(df_pred.iloc[-1, 2:].values)).any():
                df_pred = df_pred.iloc[:-1, :]
                break
        # forecast model1
        ex_pred = df_pred.iloc[:, 2:].values
        r1 = m1.predict(n_periods=ex_pred.shape[0], exogenous=ex_pred)
        # model2 (used when there is no external features in future prediction)
        if len(r1) < self.fcst_pr:
            # fit model2 without external
            try:
                m2 = pm.arima.AutoARIMA(start_p=param['start_p'], max_p=param['max_p'],
                                        start_q=param['start_q'], max_q=param['max_q'], 
                                        d=param['d'], 
                                        m = param['m'], seasonal=param['seasonal'], 
                                        trace=False, error_action='ignore', suppress_warnings=True, 
                                        stepwise=param['stepwise'])
                m2 = m2.fit(x)
            except Exception:
                m2 = pm.arima.AutoARIMA()
                m2 = m2.fit(x)
            # forecast model2
            r2 = m2.predict(n_periods=self.fcst_pr)
            r2 = r2[-(len(r2) - len(r1)):]
        else:
            r2 = []
        # summarize result
        r = list(r1) + list(r2)
        r = pd.DataFrame(zip(self.dt, r), columns =['ds', 'y'])
        r['y'] = self.grtoval(r, self.df_act) if gr else r['y']
        return r

    def autoarimax01(self):
        gr = False
        feat = ['x_']
        max_pq = {'d': 10, 'm': 12, 'q': 4, 'y': 3}
        param = {'start_p': 1, 'max_p': max_pq[self.fcst_freq], 'start_q': 1, 'max_q': max_pq[self.fcst_freq], 'd': None, 
                 'm': self.fcst_freq_pr, 'seasonal': True, 'stepwise': True}
        r = self.autoarimax(gr, feat, param)
        return r

    def autoarimax02(self):
        gr = True
        feat = ['x_']
        max_pq = {'d': 10, 'm': 12, 'q': 4, 'y': 3}
        param = {'start_p': 1, 'max_p': max_pq[self.fcst_freq], 'start_q': 1, 'max_q': max_pq[self.fcst_freq], 'd': None, 
                 'm': self.fcst_freq_pr, 'seasonal': True, 'stepwise': True}
        r = self.autoarimax(gr, feat, param)
        return r

    # Prophet by Facebook 
    def prophet01(self):
        days = {'d': 1, 'm': 31, 'q': 93, 'y': 366}
        n = self.fcst_pr * days[self.fcst_freq]
        m = Prophet()
        m.fit(self.df_d)
        f = m.make_future_dataframe(periods=n)
        r = m.predict(f)
        r = r.resample(self.fcst_freq_txt, on='ds').agg({'yhat':'sum'}).rename(columns={'yhat': 'y'}).reset_index()
        r = r[(r['ds'] >= self.fcst_st) & (r['ds'] < self.fcst_st + self.deltafreq(self.fcst_pr, self.fcst_freq))]
        return r

    # Machine Learning model without external features
    def ml(self, m, feat, gr=False):
        # too few data points for decomposing, return none
        if len(self.df_act) < 8:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data
        df = self.extractfeat(self.df_act, col=feat)
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        # too few data points, return none
        if len(df) <= 1:
            return pd.DataFrame(columns = ['ds', 'y'])
        sc = StandardScaler()
        X_trn = df.iloc[:, 2:]
        X_trn = sc.fit_transform(X_trn)
        y_trn = df.iloc[:, 1].values
        # fit model
        m.fit(X_trn, y_trn)
        # forecast each month
        r = pd.DataFrame(columns = ['ds', 'y', 'y_pred'])
        r = r.astype(dtype={'ds': str, 'y': float, 'y_pred': float})
        for i in self.dt:
            r = r.append({'ds' : i, 'y': 0, 'y_pred': 0} , ignore_index=True)
            df_pred = self.extractfeat(r, col=feat)
            x = df_pred.iloc[-1, 2:].values
            # predict m
            x_pred = sc.transform(x.reshape(1, -1))
            y_pred = m.predict(x_pred)
            r.iloc[-1, 2] = y_pred
            r['y'] = self.grtoval(r, self.df_act, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        r = r[['ds', 'y']]
        return r

    # Machine Learning model with external features
    def mlx(self, m, feat, gr=False):
        # if no external features, no forecast result
        if self.df_x is None or len(self.df_act) < 8:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare data for model1
        df = self.extractfeat(self.df_act, col=feat)
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        df = df.dropna().reset_index(drop=True)
        # too few data points, return none
        if len(df) <= 1:
            return pd.DataFrame(columns = ['ds', 'y'])
        sc = StandardScaler()
        X_trn = df.iloc[:, 2:]
        X_trn = sc.fit_transform(X_trn)
        y_trn = df.iloc[:, 1].values
        # fit model1
        m.fit(X_trn, y_trn)
        # forecast each month
        r = pd.DataFrame(columns = ['ds', 'y', 'y_pred'])
        r = r.astype(dtype={'ds': str, 'y': float, 'y_pred': float})
        for i in self.dt:
            r = r.append({'ds' : i, 'y': 0, 'y_pred': 0} , ignore_index=True)
            df_pred = self.extractfeat(r, col=feat)
            x = df_pred.iloc[-1, 2:].values
            # if no external features for prediction, break and do model2
            if np.isnan(list(x)).any():
                r = r.iloc[:-1, :]
                break
            x_pred = sc.transform(x.reshape(1, -1))
            y_pred = m.predict(x_pred)
            r.iloc[-1, 2] = y_pred
            r['y'] = self.grtoval(r, self.df_act, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        # model2 (used when there is no external features in future prediction)
        if len(r) < self.fcst_pr:
            # prepare data for model2
            feat = [x for x in feat if not x.startswith('x_')]
            df = self.extractfeat(self.df_act, col=feat)
            df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
            df = df.dropna().reset_index(drop=True)
            sc = StandardScaler()
            X_trn = df.iloc[:, 2:]
            X_trn = sc.fit_transform(X_trn)
            y_trn = df.iloc[:, 1].values
            # fit model2
            m.fit(X_trn, y_trn)
            # forecast the rest months
            for i in self.dt[len(r):]:
                r = r.append({'ds' : i, 'y': 0, 'y_pred': 0} , ignore_index=True)
                df_pred = self.extractfeat(r, col=feat)
                x = df_pred.iloc[-1, 2:].values
                x_pred = sc.transform(x.reshape(1, -1))
                y_pred = m.predict(x_pred)
                r.iloc[-1, 2] = y_pred
                r['y'] = self.grtoval(r, self.df_act, col_y='y_pred', col_yact='y') if gr else r['y_pred']
        # summarize result
        r = r[['ds', 'y']]
        return r

    # Linear Regression model without external features
    def linear(self, feat, param, gr):
        model = SGDRegressor(
            penalty=param['penalty'], max_iter=param['max_iter'], 
            early_stopping=True, random_state=1
        )
        r = self.ml(model, feat, gr)
        return r
 
    # Linear Regression without external: forecast y
    def linear01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_'],
            'q': ['quarter_', 'last_', 'dec_'],
            'y': ['last_period', 'dec_']
        }
        param = {'penalty': 'l1', 'max_iter': 1000}
        gr = False
        r = self.linear(feat[self.fcst_freq], param, gr)
        return r

    # Linear Regression without external: forecast growth
    def linear02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_'],
            'q': ['quarter_', 'lastgr_', 'dec_'],
            'y': ['lastgr_period', 'dec_']
        }
        param = {'penalty': 'l1', 'max_iter': 1000}
        gr = True
        r = self.linear(feat[self.fcst_freq], param, gr)
        return r

    # Linear Regression model with external features
    def linearx(self, feat, param, gr):
        model = SGDRegressor(
            penalty=param['penalty'], max_iter=param['max_iter'], 
            early_stopping=True, random_state=1
        )
        r = self.mlx(model, feat, gr)
        return r

    # Linear Regression with external: forecast y
    def linearx01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_', 'x_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_', 'x_'],
            'q': ['quarter_', 'last_', 'dec_', 'x_'],
            'y': ['last_period', 'dec_', 'x_']
        }
        param = {'penalty': 'l1', 'max_iter': 1000}
        gr = False
        r = self.linearx(feat[self.fcst_freq], param, gr)
        return r

    # Linear Regression with external: forecast growth
    def linearx02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_', 'x_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_', 'x_'],
            'q': ['quarter_', 'lastgr_', 'dec_', 'x_'],
            'y': ['lastgr_period', 'dec_', 'x_']
        }
        param = {'penalty': 'l1', 'max_iter': 1000}
        gr = True
        r = self.linearx(feat[self.fcst_freq], param, gr)
        return r

    # Random Forest model without external features
    def randomforest(self, feat, param, gr):
        model = RandomForestRegressor(
            n_estimators = param['n_estimators'], min_samples_split = param['min_samples_split'], 
            max_depth = param['max_depth'], max_features = param['max_features'], random_state=1
        )
        r = self.ml(model, feat, gr)
        return r
 
    # Random Forest without external: forecast y
    def randomforest01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_'],
            'q': ['quarter_', 'last_', 'dec_'],
            'y': ['last_period', 'dec_']
        }
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        gr = False
        r = self.randomforest(feat[self.fcst_freq], param, gr)
        return r

    # Random Forest without external: forecast growth
    def randomforest02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_'],
            'q': ['quarter_', 'lastgr_', 'dec_'],
            'y': ['lastgr_period', 'dec_']
        }
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        gr = True
        r = self.randomforest(feat[self.fcst_freq], param, gr)
        return r

    # Random Forest model with external features
    def randomforestx(self, feat, param, gr):
        model = RandomForestRegressor(
            n_estimators = param['n_estimators'], min_samples_split = param['min_samples_split'], 
            max_depth = param['max_depth'], max_features = param['max_features'], random_state=1
        )
        r = self.mlx(model, feat, gr)
        return r

    # Random Forest with external: forecast y
    def randomforestx01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_', 'x_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_', 'x_'],
            'q': ['quarter_', 'last_', 'dec_', 'x_'],
            'y': ['last_period', 'dec_', 'x_']
        }
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        gr = False
        r = self.randomforestx(feat[self.fcst_freq], param, gr)
        return r

    # Random Forest with external: forecast growth
    def randomforestx02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_', 'x_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_', 'x_'],
            'q': ['quarter_', 'lastgr_', 'dec_', 'x_'],
            'y': ['lastgr_period', 'dec_', 'x_']
        }
        param = {'n_estimators': 1000, 'min_samples_split': 2, 'max_depth': None, 'max_features': 'auto'}
        gr = True
        r = self.randomforestx(feat[self.fcst_freq], param, gr)
        return r

    # XGBoost model without external features
    def xgboost(self, feat, param, gr):
        model = XGBRegressor(
            learning_rate = param['learning_rate'], n_estimators = param['n_estimators'], 
            max_dept = param['max_dept'], min_child_weight = param['min_child_weight']
        )
        r = self.ml(model, feat, gr)
        return r

    # XGBoost without external: forecast y
    def xgboost01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_'],
            'q': ['quarter_', 'last_', 'dec_'],
            'y': ['last_period', 'dec_']
        }
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        gr = False
        r = self.xgboost(feat[self.fcst_freq], param, gr)
        return r

    # XGBoost without external: forecast growth
    def xgboost02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_'],
            'q': ['quarter_', 'lastgr_', 'dec_'],
            'y': ['lastgr_period', 'dec_']
        }
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        gr = True
        r = self.xgboost(feat[self.fcst_freq], param, gr)
        return r

    # XGBoost model with external features
    def xgboostx(self, feat, param, gr):
        model = XGBRegressor(
            learning_rate = param['learning_rate'], n_estimators = param['n_estimators'], 
            max_dept = param['max_dept'], min_child_weight = param['min_child_weight']
        )
        r = self.mlx(model, feat, gr)
        return r

    # XGBoost with external: forecast y
    def xgboostx01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_', 'x_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_', 'x_'],
            'q': ['quarter_', 'last_', 'dec_', 'x_'],
            'y': ['last_period', 'dec_', 'x_']
        }
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        gr = False
        r = self.xgboostx(feat[self.fcst_freq], param, gr)
        return r

    # XGBoost with external: forecast growth
    def xgboostx02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_', 'x_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_', 'x_'],
            'q': ['quarter_', 'lastgr_', 'dec_', 'x_'],
            'y': ['lastgr_period', 'dec_', 'x_']
        }
        param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_dept': 5, 'min_child_weight': 1}
        gr = True
        r = self.xgboostx(feat[self.fcst_freq], param, gr)
        return r

    # LSTM (Long Short-Term Memory) with or without external
    def lstm(self, feat, param, gr=False, rolling=False):
        # set parameter
        forward = 1 if rolling else self.fcst_pr
        look_back = param['look_back']
        n_val = param['n_val']
        # prepare data
        df = self.extractfeat(self.df_act, col=feat, rnn_delay=3)
        df['y'] = self.valtogr(df, self.fcst_freq) if gr else df['y']
        df = df.iloc[len([x for x in df['y'] if pd.isnull(x)]):, :]
        df = df.iloc[len([x for x in df['last_year'] if pd.isnull(x)]):, :] if 'last_year' in df.columns else df
        df = df.iloc[len([x for x in df['last_momentum'] if pd.isnull(x)]):, :] if 'last_momentum' in df.columns else df
        df = df.fillna(0).sort_values(by='ds', ascending=True).reset_index(drop=True)
        # check if data is sufficient to run train and validate
        len_val = n_val + look_back + forward - 1
        if len(df) <= len_val:
            return pd.DataFrame(columns = ['ds', 'y'])
        # prepare training data and validate data
        trn = df.iloc[:-n_val, :]
        val = df.iloc[-len_val:, :]
        # scaler
        sc = MinMaxScaler()
        trn = trn.iloc[:, 1:]
        trn = sc.fit_transform(trn)
        val = val.iloc[:, 1:]
        val = sc.transform(val)
        # transform to rnn format
        X_trn, y_trn = [], []
        for i in range(len(trn) - look_back - forward + 1):
            X_trn.append(trn[i:(i + look_back), :])
            y_trn.append(trn[(i + look_back):(i + look_back + forward), 0])
        X_trn, y_trn = np.array(X_trn), np.array(y_trn)
        X_val, y_val = [], []
        for i in range(len(val) - look_back - forward + 1):
            X_val.append(val[i:(i + look_back), :])
            y_val.append(val[(i + look_back):(i + look_back + forward), 0])
        X_val, y_val = np.array(X_val), np.array(y_val)
        # lstm model
        m = Sequential()
        m.add(LSTM(param['node1'], return_sequences = True, input_shape = (look_back, X_trn.shape[2])))
        m.add(LSTM(param['node2']))
        m.add(Dense(forward, activation = param['activation']))
        m.compile(loss = param['loss'], optimizer = param['optimizer'])
        # set callbacks
        callbacks = [EarlyStopping(monitor = 'val_loss', patience = param['patience'], mode = 'min', restore_best_weights = True)]
        # fit model
        m.fit(X_trn, y_trn, epochs = param['epochs'], batch_size = param['batch_size'], validation_data = (X_val, y_val), callbacks = callbacks, verbose=0)
        if rolling:
            # if rolling, forecast each month by rolling data
            r = self.df_act[['ds', 'y']]
            r['y_pred'] = self.valtogr(r, self.fcst_freq) if gr else r['y']
            r = r.iloc[len([x for x in r['y_pred'] if pd.isnull(x)]):, :]
            for i in self.dt:
                df_pred = self.extractfeat(r, col=feat, rnn_delay=3)
                df_pred['y'] = self.valtogr(df_pred, self.fcst_freq) if gr else df_pred['y']
                x_pred = df_pred.iloc[-look_back:, :]
                x_pred = x_pred.iloc[:, 1:]
                x_pred = sc.transform(x_pred)
                x_pred = x_pred.reshape(1, look_back, -1)
                y_pred = m.predict(x_pred)
                y_pred = y_pred.reshape(forward, 1)
                y_pred = np.concatenate((y_pred, np.zeros([forward, X_trn.shape[2]-1])), axis=1)
                y_pred = sc.inverse_transform(y_pred)
                y_pred = list(y_pred[:, 0])
                r = r.append({'ds' : i, 'y_pred': y_pred[0]} , ignore_index=True)
                r['y'] = self.grtoval(r, self.df_act, col_y='y_pred', col_yact='y') if gr else r['y_pred']
            r = r.iloc[-self.fcst_pr:, :][['ds', 'y']].reset_index(drop=True)
        else:
            # prepare data for predict
            df_pred = df.iloc[-look_back:, :]
            x_pred = df_pred.iloc[:, 1:]
            x_pred = sc.transform(x_pred)
            x_pred = x_pred.reshape(1, look_back, -1)
            # batch predict and transform data
            y_pred = m.predict(x_pred)
            y_pred = y_pred.reshape(forward, 1)
            y_pred = np.concatenate((y_pred, np.zeros([forward, X_trn.shape[2]-1])), axis=1)
            y_pred = sc.inverse_transform(y_pred)
            y_pred = list(y_pred[:, 0])
            r = pd.DataFrame(zip(self.dt, y_pred), columns =['ds', 'y'])
            r['y'] = self.grtoval(r, self.df_act) if gr else r['y']
        del m
        clear_session()
        return r

    # LSTM without external: forecast y
    def lstm01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_'],
            'q': ['quarter_', 'last_', 'dec_'],
            'y': ['last_period', 'dec_']
        }
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': self.fcst_freq_pr*2, 'n_val': self.fcst_freq_pr}
        gr = False
        rolling = False
        r = self.lstm(feat[self.fcst_freq], param, gr, rolling)
        return r

    # LSTM without external: forecast growth
    def lstm02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_'],
            'q': ['quarter_', 'lastgr_', 'dec_'],
            'y': ['lastgr_period', 'dec_']
        }
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': self.fcst_freq_pr*2, 'n_val': self.fcst_freq_pr}
        gr = True
        rolling = False
        r = self.lstm(feat[self.fcst_freq], param, gr, rolling)
        return r

    # LSTM without external and rolling forecast: forecast y
    def lstmr01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_'],
            'q': ['quarter_', 'last_', 'dec_'],
            'y': ['last_period', 'dec_']
        }
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': self.fcst_freq_pr*2, 'n_val': self.fcst_freq_pr}
        gr = False
        rolling = True
        r = self.lstm(feat[self.fcst_freq], param, gr, rolling)
        return r

    # LSTM without external and rolling forecast: forecast growth
    def lstmr02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_'],
            'q': ['quarter_', 'lastgr_', 'dec_'],
            'y': ['lastgr_period', 'dec_']
        }
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': self.fcst_freq_pr*2, 'n_val': self.fcst_freq_pr}
        gr = True
        rolling = True
        r = self.lstm(feat[self.fcst_freq], param, gr, rolling)
        return r

    # LSTM with external: forecast y
    def lstmx01(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'last_', 'dec_', 'xrnn_'], 
            'm': ['month_', 'quarter_', 'last_', 'dec_', 'x_'],
            'q': ['quarter_', 'last_', 'dec_', 'x_'],
            'y': ['last_period', 'dec_', 'x_']
        }
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': self.fcst_freq_pr*2, 'n_val': self.fcst_freq_pr}
        gr = False
        rolling = False
        if self.df_x is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        r = self.lstm(feat[self.fcst_freq], param, gr, rolling)
        return r

    # LSTM with external: forecast growth
    def lstmx02(self):
        feat = {
            'd': ['day_', 'dayofyear_', 'weekofyear_', 'weekday_', 'month_', 'quarter_', 'lastgr_', 'dec_', 'xrnn_'], 
            'm': ['month_', 'quarter_', 'lastgr_', 'dec_', 'x_'],
            'q': ['quarter_', 'lastgr_', 'dec_', 'x_'],
            'y': ['lastgr_period', 'dec_', 'x_']
        }
        param = {'node1': 800, 'node2': 400, 
                 'activation': 'linear', 'optimizer': 'adam', 
                 'loss': 'mean_absolute_error', 'epochs': 10, 
                 'batch_size': 1, 'patience': 10,
                 'look_back': self.fcst_freq_pr*2, 'n_val': self.fcst_freq_pr}
        gr = True
        rolling = False
        if self.df_x is None:
            return pd.DataFrame(columns = ['ds', 'y'])
        r = self.lstm(feat[self.fcst_freq], param, gr, rolling)
        return r


class EnsembleModel:
    """Ensemble top forecasting model based on forecasting log
    Init Parameters
    ----------
    df_y : pandas dataframe containing two columns: [id, ds, y]
        historical time series input data by id
    df_fcstlog : pandas dataframe containing six columns: [id, ds, dsr, period, model, forecast]
        historical rolling forecasting result
    fcst_freq : {"d", "m", "q", "y"}, optional
        forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
    """
    def __init__(self, df_y, df_fcstlog, fcst_freq='m'):
        self.df_y = df_y
        self.df_fcstlog = df_fcstlog
        self.fcst_freq = fcst_freq
        df_act = pd.DataFrame()
        for i in df_y['id'].unique():
            df_i = df_y[df_y['id']==i].copy()
            df_i = TimeSeriesForecasting.filldaily(df_i, df_i['ds'].min(), df_i['ds'].max())
            df_i = df_i.resample(TimeSeriesForecasting.freq_dict[fcst_freq], on='ds').agg({'y':'sum'}).reset_index()
            df_i['id'] = i
            df_act = df_act.append(df_i[['id', 'ds', 'y']], ignore_index=True)
        self.df_act = df_act

    @staticmethod
    def cal_error(act, pred, error_type='mae'):
        if error_type == 'mae':
            return abs(act - pred)
        elif error_type == 'mape':
            if act == 0 and pred == 0:
                return 0
            elif act == 0 and pred != 0:
                return 1
            else:
                try:
                    return np.abs((act - pred) / act)
                except Exception:
                    return None
        else:
            return None

    def rank(self, test_back, error_ens, error_dsp, rank_by='mae'):
        """Rank model by their error
        Parameters
        ----------
        test_back : int
            number of periods to test back
        error_ens : {"mean", "median"}, optional
            method to ensemble error
        error_dsp : {"mae", "mape"}, optional
            display error by
        rank_by : {"mae", "mape"}, optional
            rank top model by
        """
        dsr_list = list(pd.Timestamp(x) + TimeSeriesForecasting.deltafreq(1, self.fcst_freq) for x in self.df_fcstlog['dsr'].unique())
        df_rank = pd.DataFrame()
        for i in dsr_list:
            # limit ds / dsr
            df_act = self.df_act[self.df_act['ds'] < i]
            df_fcstlog = self.df_fcstlog[(self.df_fcstlog['ds'] >= i - TimeSeriesForecasting.deltafreq(test_back, self.fcst_freq))  & (self.df_fcstlog['dsr'] < i)].copy()
            # calculate mae / mape
            df_i = pd.merge(df_fcstlog, df_act.rename(columns={'y': 'actual'}), on=['id', 'ds'], how='left')
            df_i['mae'] = df_i.apply(lambda x: self.cal_error(x['actual'], x['forecast'], 'mae'), axis=1)
            df_i['mape'] = df_i.apply(lambda x: self.cal_error(x['actual'], x['forecast'], 'mape'), axis=1)
            # calculate test back and na period
            df_testback = df_i[df_i['actual'].notnull()].groupby(['id', 'period', 'model'], as_index=False).agg({'ds':'count'}).rename(columns={'ds': 'test_back'})
            df_fillna = df_i.groupby(['period'], as_index=False)['actual'].sum(min_count=1)
            # rank error
            df_i = df_i.groupby(['id', 'period', 'model'], as_index=False).agg({'mae': error_ens, 'mape': error_ens})
            df_i['rank'] = df_i.groupby(['id', 'period'])[rank_by].rank(method='dense', ascending=True)
            df_i['error'] = df_i[error_dsp]
            # merge test back count
            df_i = pd.merge(df_i, df_testback, on=['id', 'period', 'model'], how='left')
            # fill rank = 1 for all periods that do not have forecast log
            df_i.loc[df_i['period'].isin(df_fillna[df_fillna['actual'].isnull()]['period']), 'rank'] = 1
            # sort column
            df_i['dsr'] = i
            df_i = df_i[['id', 'dsr', 'period', 'rank', 'model', 'error', 'test_back']]
            df_rank = df_rank.append(df_i, ignore_index=True)
        self.df_rank = df_rank

    def ensemble(self, df_fcst, top_model=3, fcst_ens='mean', error_ens='mean'):
        """Ensemble latest forecasting result
        Parameters
        ----------
        df_fcst : pandas dataframe containing six columns: [id, ds, dsr, period, model, forecast]
            latest forecasting result of all models
        top_model : int
            number of top model to ensemble
        fcst_ens : {"mean", "median"}, optional
            method to ensemble forecast
        error_ens : {"mae", "mape"}, optional
            method to ensemble error
        """
        df_ens = pd.merge(df_fcst, self.df_rank, on=['id', 'dsr', 'period', 'model'], how='left')
        df_ens['rank'] = df_ens.groupby(['id', 'period'])['rank'].rank(method='dense', ascending=True)
        df_ens = df_ens.sort_values(by=['id', 'period', 'rank'], ascending=True).reset_index(drop=True)
        df_ens = df_ens[df_ens['rank'] <= top_model].copy()
        df_ens = df_ens.groupby(['id', 'ds', 'dsr', 'period'], as_index=False).agg({
            'forecast': fcst_ens, 
            'error': error_ens, 
            'model': list, 
            'test_back':'mean'}
            ).rename(columns={'model': 'top_model'})
        df_ens = df_ens.sort_values(by=['id', 'dsr', 'ds'], ascending=True).reset_index(drop=True)
        return df_ens

    def ensemble_fcstlog(self, top_model=3, fcst_ens='mean', error_ens='mean'):
        """Ensemble all forecasting result from forecasting log
        Parameters
        ----------
        top_model : int
            number of top model to ensemble
        fcst_ens : {"mean", "median"}, optional
            method to ensemble forecast
        error_ens : {"mae", "mape"}, optional
            method to ensemble error
        """
        df_ens = pd.DataFrame()
        for i in self.df_fcstlog['dsr'].unique():
            df_i = self.df_fcstlog[self.df_fcstlog['dsr'] == i]
            df_ens = df_ens.append(self.ensemble(df_i, top_model, fcst_ens, error_ens), ignore_index=True)
        df_ens = df_ens.sort_values(by=['id', 'dsr', 'ds'], ascending=True).reset_index(drop=True)
        return df_ens


class FeatSelection:
    """Find best features and their lag from x-series which have the most correlation to the y-series
    Init Parameters
    ----------
    fcst_freq : {"d", "m", "q", "y"}, optional
        series frequency of x and y (d-daily, m-monthly, q-quarterly, y-yearly)
    growth : boolean, default=True
        calculate based on growth or not
    """

    freq_dict = {'d': 'D', 'm': 'MS', 'q': 'QS', 'y': 'YS'}
    freq_period = {'d': 365, 'm': 12, 'q': 4, 'y': 1}

    def __init__(self, freq='m', growth=True):
        self.freq = freq
        self.growth = growth

    @classmethod
    def clean_date(cls, df, freq, col_val='x', col_ds='ds'):
        """Removes all the data before the latest missing date of a series"""
        df = df.groupby(col_ds, as_index=False).agg({col_val:'sum'})
        missing_dates = pd.date_range(start=min(df[col_ds]), end=max(df[col_ds]), freq=cls.freq_dict[freq]).difference(df[col_ds])
        if len(missing_dates) > 0:
            df = df[df[col_ds]>max(missing_dates)]
        return df.sort_values(by=col_ds, ascending=True).reset_index(drop=True)

    @classmethod
    def cal_gr(cls, df, freq, col_val='x', col_ds='ds'):
        """Calculates percentage growth"""
        df.sort_values(by=col_ds, ascending=True).reset_index(drop=True)
        df[col_val] = df[col_val].pct_change(periods=cls.freq_period[freq])
        df = df.iloc[cls.freq_period[freq]:, :].reset_index(drop=True)
        df[col_val] = df[col_val].replace([np.inf, -np.inf, None], [1, -1, 1])
        return df

    @classmethod
    def merge_lag(cls, df, df_x, x, lag):
        """Merge dataframe to the features with specific lag"""
        df = pd.merge(df, df_x.rename(columns={'x': x}), how='right', on='ds').sort_values('ds')
        df[x] = df[x].shift(lag)
        return df.dropna(axis=0, how='any')

    def set_x(self, df_x, min_data_points=72):
        """Set x series and clean them
        ----------
        Parameter:
            df_x : pandas dataframe containing three columns: ['id','ds','x']
                The x-series to test with y-series
            min_data_points : int
                Minimum accepted data points for x-series after cleaning process
        """
        self.df_x = {}
        self.total_x = len(df_x['id'].unique())
        for x in df_x['id'].unique():
            df = df_x[df_x['id']==x].copy()
            df = self.clean_date(df, freq=self.freq, col_val='x')
            df = self.cal_gr(df, freq=self.freq, col_val='x') if self.growth else df
            if len(df) >= min_data_points:
                self.df_x[x] = df
        self.total_test_x = len(self.df_x)

    def set_y(self, df_y):
        """Set y series and clean it
        ----------
        Parameter:
            df_y : pandas dataframe containing two columns: ['ds','y']
                The y-series to test with x-series
        """
        self.df_y = self.clean_date(df_y, freq=self.freq, col_val='y')
        self.df_y = self.cal_gr(self.df_y, freq=self.freq, col_val='y') if self.growth else self.df_y

    def find_lag(self, min_lag=2, max_lag=23):
        """Calculate best lag for each x-series"""
        self.best_lag = {}
        for x in self.df_x:
            df = pd.merge(self.df_y, self.df_x[x], how='right', on='ds').sort_values('ds')
            best_corr = -1.
            for lag in range(min_lag, max_lag+1):
                corr = np.fabs(df['y'].corr(df['x'].shift(lag)))
                if corr > best_corr:
                    self.best_lag[x] = lag
                    best_corr = corr

    def fit(self, i, **kwargs):
        """Fit function
        Parameters
        ----------
        i : {options}
            Specify model to run
            options:
                aicc01 - Select features by forwarding method and AICc scoring
                aic01 - Select features by forwarding method and AIC scoring
                r2adj01 - Select features by forwarding method and R2-adj scoring
        Returns
        -------
        result : pandas dataframe containing two columns: ['id', 'lag']
        """
        fn = getattr(FeatSelection, i)
        return fn(self, **kwargs)

    """ MODELS """
    def cal_score(self, yx_mat, scoring_method='AICc'):
        """
        Parameters:
        ----------
        yx_mat : array
            A time series matrix with a size of (n, m+1)
            The first column contains y time series
            The rest are x's
        method : {"AICc", "AIC", "R2-adj"}, optional, default="AICc"
            A method to calculate score
        Return:
        ----------
        result : An objective function value
        """
        y = yx_mat[:,0]
        x = yx_mat[:,1:]
        n = x.shape[0]
        p = x.shape[1]
        reg = LinearRegression().fit(x, y)
        if scoring_method == 'R2-adj':
            r2 = reg.score(x, y)
            r2_adj = 1 - (1-r2)*(n-p)/(n-p-1)
            return r2_adj
        elif scoring_method in ('AICc', 'AIC'):
            y_pred = reg.predict(x)
            error = y - y_pred
            sd = np.sqrt(np.mean(error**2))
            var = sd**2
            # Both are correct, I like to calculate my own...
            log_l = n*np.log(1/np.sqrt(2*np.pi*var)) - np.sum(error**2)/(2*var)
            #log_l = sum(norm.logpdf(y, loc=y_pred, scale=sd))
            if n-p-1 == 0:
                corrected = 99999
            else:
                corrected = 0 if scoring_method == 'AIC' else (2*p*(p+1)/(n-p-1))
            AICc = -2*log_l + 2*p + corrected
            return -AICc

    # Forwarding selection model to find the best features
    def opt_forward(self, max_features=10, scoring_method='AICc'):
        self.best_features = {}
        best_score = -999999.
        df = self.df_y
        n_iter = min(len(self.best_lag), max_features)
        for i in range(n_iter):
            best_update = False
            for x in self.best_lag:
                if x in self.best_features.keys(): continue

                df_temp = self.merge_lag(df, self.df_x[x], x, self.best_lag[x])
                yx_mat = df_temp.drop(columns=['ds']).values
                score = self.cal_score(yx_mat, scoring_method)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_update = True

            if not best_update: break

            self.best_features[best_x] = self.best_lag[best_x]
            df = self.merge_lag(df, self.df_x[best_x], best_x, self.best_lag[best_x])

    # Optimize score to find the best features
    def opt_score(self, selection_method, scoring_method='AICc', max_features=10):
        if selection_method=='forward':
            self.opt_forward(max_features, scoring_method)
            return pd.DataFrame(self.best_features.items(), columns=['id', 'lag'])
        else:
            return pd.DataFrame(columns = ['id', 'lag'])

    # Optimize score by forward selection and AICc score
    def aicc01(self, max_features=10):
        selection_method = 'forward'
        scoring_method = 'AICc'
        r = self.opt_score(selection_method, scoring_method, max_features)
        return r

    # Optimize score by forward selection and AIC score
    def aic01(self, max_features=10):
        selection_method = 'forward'
        scoring_method = 'AIC'
        r = self.opt_score(selection_method, scoring_method, max_features)
        return r

    # Optimize score by forward selection and adjusted R-squared score
    def r2adj01(self, max_features=10):
        selection_method = 'forward'
        scoring_method = 'R2-adj'
        r = self.opt_score(selection_method, scoring_method, max_features)
        return r
