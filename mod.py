# -*- coding: utf-8 -*-
import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing
import warnings
import traceback

from pytz import timezone
import numpy as np
import pandas as pd

from model import TimeSeriesForecasting, FeatSelection
from utils import FilePath, Logging, chunker, mape


warnings.filterwarnings("ignore")


class Validation:
    """Validate forecast model by rolling forecast
    Init Parameters
    ----------
    platform : {'local', 'gcp'}
        platform to store input/output
    tz : str (e.g. Asia/Bangkok)
        timezone for logging
    logtag : str
        logging tag
    cloud_auth : str
        authentication file path
    """
    def __init__(self, platform, logtag, tz, cloud_auth=None):
        self.fp = FilePath(platform, cloud_auth)
        self.lg = Logging(platform, "validate", logtag, cloud_auth)
        self.lg.logtxt("[START VALIDATION]")
        self.tz = tz

    def loaddata(self, act_path, ext_path=None, extlag_path=None):
        """Load data for validation process
        Parameters
        ----------
        act_path : str
            historical data path
        ext_path : str
            external features path
        extlag_path : str
            external lag path
        """
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        # load sales data
        df_act = pd.read_csv(self.fp.loadfile(act_path), parse_dates=['ds'], date_parser=dateparse)
        self.df_act = df_act[['id', 'ds', 'y']]
        # load external features
        if ext_path is not None:
            df_ext = pd.read_csv(self.fp.loadfile(ext_path), parse_dates=['ds'], date_parser=dateparse)
            df_extlag = pd.read_csv(self.fp.loadfile(extlag_path))
            self.df_ext = df_ext[['id', 'ds', 'x']]
            self.df_extlag = df_extlag[['id_y', 'id_x', 'lag']]
            self.lg.logtxt("load data: {} | {} | {}".format(act_path, ext_path, extlag_path))
        else:
            self.df_ext = None
            self.df_extlag = None
            self.lg.logtxt("load data: {}".format(act_path))
            
    def validate_byitem(self, id_y, act_st, test_date, test_model, fcst_pr, fcst_freq, pr_st, batch_no):
        """Validate data by item for parallel computing"""
        df_y = self.df_act[self.df_act['id']==id_y][['ds', 'y']].copy()
        if self.df_ext is not None:
            df_x = self.df_ext[['id', 'ds', 'x']].copy()
            df_lag = self.df_extlag[self.df_extlag['id_y']==id_y].rename(columns={'id_x': 'id'})[['id', 'lag']].copy()
        else:
            df_x = None
            df_lag = None
        df_r = pd.DataFrame()
        for d in test_date:
            model = TimeSeriesForecasting(df_y=df_y, act_st=act_st, fcst_st=d, fcst_pr=fcst_pr, fcst_freq=fcst_freq, df_x=df_x, df_lag=df_lag)
            for m in test_model:
                runitem = {"batch": batch_no, "id": id_y, "testdate": d, "model": m}
                try:
                    st_time = datetime.datetime.now()
                    r = model.forecast(m)
                    r = r.rename(columns={'y': 'forecast'})
                    r['time'] = (datetime.datetime.now() - st_time).total_seconds()
                    r['id'] = id_y
                    r['dsr'] = d
                    r['period'] = np.arange(pr_st, len(r)+pr_st)
                    r['model'] = m
                    r = r[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'time']]
                    df_r = df_r.append(r, ignore_index = True)
                except Exception:
                    error_item = "batch: {} | id: {} | testdate: {} | model:{}".format(
                        runitem.get('batch'), runitem.get('id'), runitem.get('testdate').strftime("%Y-%m-%d"), runitem.get('model'))
                    error_txt = "ERROR: {} ({})".format(error_item, str(traceback.format_exc()))
                    self.lg.logtxt(error_txt, error=True)
        return df_r

    def validate(self, output_dir, act_st, test_st, test_pr, test_model, fcst_pr, fcst_freq, pr_st, chunk_sz, cpu):
        """Validate forecast model and write result by batch
        Parameters
        ----------
        output_dir : str
            output directory
        act_st : datetime
            actual start date
        test_st : datetime
            test start date
        test_pr : int
            number of rolling period to test (months)
        test_model : list
            list of model to test
        fcst_pr : int
            number of periods to forecast for each rolling
        fcst_freq : {"d", "m", "q", "y"}, optional
            forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
        pr_st : int
            starting period for each forecast (default 0/1)
        chunk_sz : int
            number of item to validate for each chunk
        cpu : int
            number of running processors
        """
        # make output directory
        output_dir = "{}validate_{}/".format(output_dir, datetime.datetime.now(timezone(self.tz)).strftime("%Y%m%d-%H%M%S"))
        self.output_dir = output_dir
        self.fp.mkdir(output_dir)
        self.lg.logtxt("create output directory: {}".format(output_dir))
        self.fp.writecsv(self.df_act, "{}input_actual.csv".format(output_dir))
        # write external features
        if self.df_ext is not None:
            self.fp.writecsv(self.df_ext, "{}input_external.csv".format(output_dir))
            self.fp.writecsv(self.df_extlag, "{}input_externallag.csv".format(output_dir))
            self.lg.logtxt("write input file: {}input_actual.csv | {}input_external.csv | {}input_externallag.csv".format(output_dir,output_dir,output_dir))
        else:
            self.lg.logtxt("write input file: {}input_actual.csv".format(output_dir))
        # set parameter
        items = self.df_act['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        test_date = [x.to_pydatetime() + datetime.timedelta(days=+test_st.day-1) for x in pd.date_range(start=test_st, periods=test_pr, freq=TimeSeriesForecasting.freq_dict[fcst_freq])]
        self.lg.logtxt("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # loop by chunk
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            if cpu_count==1:
                for r in [self.validate_byitem(x, act_st, test_date, test_model, fcst_pr, fcst_freq, pr_st, i) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.validate_byitem, [[x, act_st, test_date, test_model, fcst_pr, fcst_freq, pr_st, i] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
            # write csv file
            output_path = "{}output_validate_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_fcst, output_path)
            self.lg.logtxt("write output file ({}/{}): {}".format(i, n_chunk, output_path))
        self.lg.logtxt("[END VALIDATION]")
        self.lg.writelog("{}logfile.log".format(output_dir))


class Forecasting:
    """Forecast and perform model selection based on historical forecast
    Init Parameters
    ----------
    platform : {'local', 'gcp'}
        platform to store input/output
    logtag : str
        logging tag
    tz : str (e.g. Asia/Bangkok)
        timezone for logging
    cloud_auth : str
        authentication file path
    """
    def __init__(self, platform, logtag, tz, cloud_auth=None):
        self.fp = FilePath(platform, cloud_auth)
        self.lg = Logging(platform, "forecast", logtag, cloud_auth)
        self.lg.logtxt("[START FORECASTING]")
        self.tz = tz
    
    def loaddata(self, act_path, fcstlog_path, ext_path=None, extlag_path=None):
        """Load data for validation process
        Parameters
        ----------
        act_path : str
            historical data path
        fcstlog_path : str
            forecast log path
        ext_path : str
            external features path
        extlag_path : str
            external lag path
        """
        # load actual and forecast data
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        df_act = pd.read_csv(self.fp.loadfile(act_path), parse_dates=['ds'], date_parser=dateparse)
        df_fcstlog = pd.read_csv(self.fp.loadfile(fcstlog_path), parse_dates=['ds', 'dsr'], date_parser=dateparse)
        self.df_act = df_act[['id', 'ds', 'y']]
        self.df_fcstlog = df_fcstlog[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'time']]
        # load external features
        if ext_path is not None:
            df_ext = pd.read_csv(self.fp.loadfile(ext_path), parse_dates=['ds'], date_parser=dateparse)
            df_extlag = pd.read_csv(self.fp.loadfile(extlag_path))
            self.df_ext = df_ext[['id', 'ds', 'x']]
            self.df_extlag = df_extlag[['id_y', 'id_x', 'lag']]
            self.lg.logtxt("load data: {} | {} | {} | {}".format(act_path, fcstlog_path, ext_path, extlag_path))
        else:
            self.df_ext = None
            self.df_extlag = None
            self.lg.logtxt("load data: {} | {}".format(act_path, fcstlog_path))

    def forecast_byitem(self, id_y, act_st, fcst_st, fcst_pr, fcst_freq, model_list, pr_st, batch_no):
        """Forecast data by item for parallel computing"""
        df_y = self.df_act[self.df_act['id']==id_y].copy()
        if self.df_ext is not None:
            df_x = self.df_ext[['id', 'ds', 'x']].copy()
            df_lag = self.df_extlag[self.df_extlag['id_y']==id_y].rename(columns={'id_x': 'id'})[['id', 'lag']].copy()
        else:
            df_x = None
            df_lag = None
        model = TimeSeriesForecasting(df_y=df_y, act_st=act_st, fcst_st=fcst_st, fcst_pr=fcst_pr, fcst_freq=fcst_freq, df_x=df_x, df_lag=df_lag)
        df_r = pd.DataFrame()
        for m in model_list:
            try:
                runitem = {"batch": batch_no, "id": id_y, "model": m}
                st_time = datetime.datetime.now()
                r = model.forecast(m)
                r = r.rename(columns={'y': 'forecast'})
                r['time'] = (datetime.datetime.now() - st_time).total_seconds()
                r['id'] = id_y
                r['dsr'] = fcst_st
                r['model'] = m
                r['period'] = np.arange(pr_st, len(r)+pr_st)
                r = r[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'time']]
                df_r = df_r.append(r, ignore_index = True)
            except Exception:
                error_item = "batch: {} | id: {} | model:{}".format(runitem.get('batch'), runitem.get('id'), runitem.get('model'))
                error_txt = "ERROR: {} ({})".format(error_item, str(traceback.format_exc()))
                self.lg.logtxt(error_txt, error=True)
        return df_r

    def rank_model(self, fcst_model, act_st, fcst_st, fcst_freq, test_st, rank_by='mae', error_by='mape'):
        """Rank model based on historical forecast"""
        df_act = pd.DataFrame()
        for i in self.df_act['id'].unique():
            df_i = self.df_act[self.df_act['id']==i].copy()
            df_i = TimeSeriesForecasting.filldaily(df_i, act_st, fcst_st + datetime.timedelta(days=-1))
            df_i = df_i.resample(TimeSeriesForecasting.freq_dict[fcst_freq], on='ds').agg({'y':'sum'}).reset_index()
            df_i['id'] = i
            df_act = df_act.append(df_i[['id', 'ds', 'y']], ignore_index=True)
        df_act = df_act[df_act['ds']<fcst_st]
        df_rank = self.df_fcstlog[(self.df_fcstlog['ds']>=test_st) & (self.df_fcstlog['dsr']<fcst_st)].copy()
        df_rank = df_rank.groupby(['id', 'dsr', 'period', 'model']).resample(TimeSeriesForecasting.freq_dict[fcst_freq], on='ds').sum()[['forecast', 'time']].reset_index()
        # select only in config file
        df_rank['val'] = df_rank['period'].map(fcst_model)
        df_rank = df_rank[df_rank['val'].notnull()].copy()
        df_rank['val'] = df_rank.apply(lambda x: True if x['model'] in x['val'] else False, axis=1)
        df_rank = df_rank[df_rank['val']==True].copy()
        # calculate error comparing with actual
        df_rank = pd.merge(df_rank, df_act.rename(columns={'y': 'actual'}), on=['id', 'ds'], how='left')
        df_rank['mae'] = df_rank.apply(lambda x: abs(x['actual'] - x['forecast']), axis=1)
        df_rank['mape'] = df_rank.apply(lambda x: mape(x['actual'], x['forecast']), axis=1)
        df_testback = df_rank[df_rank['actual'].notnull()].groupby(['id', 'period', 'model'], as_index=False).agg({'ds':'count'}).rename(columns={'ds': 'test_back'})
        df_fillna = df_rank.groupby(['period'], as_index=False)['actual'].sum(min_count=1)
        # ranking error
        df_rank = df_rank.groupby(['id', 'period', 'model'], as_index=False).agg({'mae':'mean', 'mape':'mean'})
        df_rank['rank'] = df_rank.groupby(['id', 'period'])[rank_by].rank(method='dense', ascending=True)
        df_rank['error'] = df_rank[error_by]
        # count test back period
        df_rank = pd.merge(df_rank, df_testback, on=['id', 'period', 'model'], how='left')
        # fill rank=1 for periods that have no forecast log
        df_rank.loc[df_rank['period'].isin(df_fillna[df_fillna['actual'].isnull()]['period']), 'rank'] = 1
        return df_rank

    def ensemble_model(self, df_fcst, df_rank, top_model, method):
        # combine forecast
        df_ens = pd.merge(df_fcst, df_rank, on=['id', 'period', 'model'], how='left')
        df_ens = df_ens[df_ens['rank'] <= top_model].copy()
        df_ens = df_ens.groupby(['id', 'ds', 'dsr', 'period'], as_index=False).agg({'forecast': method, 'error': method, 'model': list, 'test_back':'mean'}).rename(columns={'model': 'top_model'})
        df_ens = df_ens.sort_values(by=['id', 'dsr', 'ds'], ascending=True).reset_index(drop=True)
        return df_ens

    def forecast(self, output_dir, act_st, fcst_st, fcst_model, fcst_freq, test_bck, top_model=3, ens_method='mean', chunk_sz=1, cpu=1):
        """Forecast and write result by batch
        Parameters
        ----------
        output_dir : str
            output directory
        act_st : datetime
            actual start date
        fcst_st : datetime
            forecast date
        fcst_model : dict('period', [list of models])
            forecast model options for each periods
        fcst_freq : {"d", "m", "q", "y"}, optional
            forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
        test_bck : int
            number of months to test back
        chunk_sz : int
            number of item to validate for each chunk
        cpu : int
            number of running processors
        """
        # make output directory
        output_dir = "{}forecast_{}/".format(output_dir, datetime.datetime.now(timezone(self.tz)).strftime("%Y%m%d-%H%M%S"))
        self.output_dir = output_dir
        self.fp.mkdir(output_dir)
        self.lg.logtxt("create output directory: {}".format(output_dir))
        self.fp.writecsv(self.df_act, "{}input_actual.csv".format(output_dir))
        self.fp.writecsv(self.df_fcstlog, "{}input_forecast.csv".format(output_dir))
        # write external features
        if self.df_ext is not None:
            self.fp.writecsv(self.df_ext, "{}input_external.csv".format(output_dir))
            self.fp.writecsv(self.df_extlag, "{}input_externallag.csv".format(output_dir))
            self.lg.logtxt("write input file: {}input_actual.csv | {}input_forecast.csv | {}input_external.csv | {}input_externallag.csv".format(output_dir,output_dir,output_dir,output_dir))
        else:
            self.lg.logtxt("write input file: {}input_actual.csv | {}input_forecast.csv".format(output_dir, output_dir))
        self.runitem = {}
        # set parameter
        items = self.df_act['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        act_st = datetime.datetime.combine(act_st, datetime.datetime.min.time())
        fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        test_st = fcst_st - TimeSeriesForecasting.deltafreq(test_bck, fcst_freq)
        fcst_pr = len(fcst_model.keys())
        pr_st = min(fcst_model.keys())
        model_list = list(set(b for a in fcst_model.values() for b in a))
        self.lg.logtxt("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # rank the models
        df_rank = self.rank_model(fcst_model, act_st, fcst_st, fcst_freq, test_st)
        # forecast
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            if cpu_count==1:
                for r in [self.forecast_byitem(x, act_st, fcst_st, fcst_pr, fcst_freq, model_list, pr_st, i) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.forecast_byitem, [[x, act_st, fcst_st, fcst_pr, fcst_freq, model_list, pr_st, i] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
            # ensemble forecast results
            df_ens = self.ensemble_model(df_fcst, df_rank, top_model, method=ens_method)
            # write forecast result
            fcst_path = "{}output_forecast_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_ens, fcst_path)
            # write forecast log result
            fcstlog_path = "{}output_forecastlog_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_fcst, fcstlog_path)
            self.lg.logtxt("write output file ({}/{}): {} | {}".format(i, n_chunk, fcst_path, fcstlog_path))
        self.lg.logtxt("[END FORECAST]")
        self.lg.writelog("{}logfile.log".format(output_dir))


class BatchFeatSelection:
    def __init__(self, platform, logtag, tz, cloud_auth=None):
        self.fp = FilePath(platform, cloud_auth)
        self.lg = Logging(platform, "feature-selection", logtag, cloud_auth)
        self.lg.logtxt("[START FEATURE SELECTION]")
        self.tz = tz
    
    def loaddata(self, x_path, y_path):
        """Load data for validation process
        Parameters
        ----------
        x_path : str
            x-series path
        y_path : str
            y-series path
        """
        # load x-series and y-series data
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        df_x = pd.read_csv(self.fp.loadfile(x_path), parse_dates=['ds'], date_parser=dateparse)
        df_y = pd.read_csv(self.fp.loadfile(y_path), parse_dates=['ds'], date_parser=dateparse)
        self.df_x = df_x[['id', 'ds', 'x']]
        self.df_y = df_y[['id', 'ds', 'y']]
        self.lg.logtxt("load data: {} | {}".format(x_path, y_path))

    def select_byitem(self, id_y, model, model_name, min_lag, max_lag, max_features, batch_no):
        """Select features by item for parallel computing"""
        df_y = self.df_y[self.df_y['id']==id_y][['ds', 'y']].copy()
        runitem = {"batch": batch_no, "id": id_y}
        try:
            model.set_y(df_y)
            model.find_lag(min_lag, max_lag)
            df_r = model.fit(model_name, max_features=max_features)
            df_r = df_r.rename(columns={'id': 'id_x'})
            df_r['id_y'] = id_y
            df_r = df_r[['id_y', 'id_x', 'lag']]
            return df_r
        except Exception:
            error_item = "batch: {} | id: {}".format(runitem.get('batch'), runitem.get('id'))
            error_txt = "ERROR: {} ({})".format(error_item, str(traceback.format_exc()))
            self.lg.logtxt(error_txt, error=True)

    def select(self, output_dir, model_name='aicc01', freq='m', growth=True, min_data_points=72, min_lag=2, max_lag=23, max_features=10, chunk_sz=1, cpu=1):
        """Forecast and write result by batch
        Parameters
        ----------
        output_dir : str
            output directory
        """
        # make output directory
        output_dir = "{}featselection_{}/".format(output_dir, datetime.datetime.now(timezone(self.tz)).strftime("%Y%m%d-%H%M%S"))
        self.output_dir = output_dir
        self.fp.mkdir(output_dir)
        self.lg.logtxt("create output directory: {}".format(output_dir))
        self.fp.writecsv(self.df_x, "{}input_x.csv".format(output_dir))
        self.fp.writecsv(self.df_y, "{}input_y.csv".format(output_dir))
        self.lg.logtxt("write input file: {}input_x.csv | {}input_y.csv".format(output_dir, output_dir))
        self.runitem = {}
        # initial model
        f = FeatSelection(freq, growth)
        f.set_x(self.df_x, min_data_points)
        self.lg.logtxt("total x-series: {}/{}".format(f.total_test_x, f.total_x))
        # separate chunk
        items = self.df_y['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        self.lg.logtxt("total y-series: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # select based on y-series
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_select = pd.DataFrame()
            if cpu_count==1:
                for r in [self.select_byitem(y, f, model_name, min_lag, max_lag, max_features, i) for y in c]:
                    df_select = df_select.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.select_byitem, [[y, f, model_name, min_lag, max_lag, max_features, i] for y in c]):
                    df_select = df_select.append(r, ignore_index = True)
                pool.close()
                pool.join()
            # write selection result
            feat_path = "{}output_feature_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_select, feat_path)
            self.lg.logtxt("write output file ({}/{}): {}".format(i, n_chunk, feat_path))
        self.lg.logtxt("[END SELECTION]")
        self.lg.writelog("{}logfile.log".format(output_dir))
