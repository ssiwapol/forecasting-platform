# -*- coding: utf-8 -*-
import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing
import warnings

from pytz import timezone
import numpy as np
import pandas as pd

from model import TimeSeriesForecasting
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
        col_id, col_ds, col_y = 'id', 'ds', 'y'
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        # load sales data
        df = pd.read_csv(self.fp.loadfile(act_path), parse_dates=['ds'], date_parser=dateparse)
        self.df = df.rename(columns={col_id: 'id', col_ds: 'ds', col_y: 'y'})[['id', 'ds', 'y']]
        # load external features
        if ext_path is not None:
            col_yid, col_extid, col_lag = 'y_id', 'ext_id', 'lag'
            ext = pd.read_csv(self.fp.loadfile(ext_path), parse_dates=['ds'], date_parser=dateparse)
            ext_lag = pd.read_csv(self.fp.loadfile(extlag_path), date_parser=dateparse)
            self.ext = ext.rename(columns={col_id: 'id', col_ds: 'ds', col_y: 'y'})[['id', 'ds', 'y']]
            self.ext_lag = ext_lag.rename(columns={col_yid: 'y_id', col_extid: 'ext_id', col_lag: 'lag'})[['y_id', 'ext_id', 'lag']]
            self.lg.logtxt("load data: {} | {} | {}".format(act_path, ext_path, extlag_path))
        else:
            self.ext = None
            self.ext_lag = None
            self.lg.logtxt("load data: {}".format(act_path))
            
    def validate_byitem(self, x, act_st, test_date, test_model, fcst_pr, pr_st, batch_no):
        """Validate data by item for parallel computing"""
        df = self.df[self.df['id']==x][['ds', 'y']].copy()
        if self.ext is not None:
            ext = self.ext[['id', 'ds', 'y']].copy()
            ext_lag = self.ext_lag[self.ext_lag['y_id']==x].rename(columns={'ext_id': 'id'})[['id', 'lag']].copy()
        else:
            ext = None
            ext_lag = None
        df_r = pd.DataFrame()
        for d in test_date:
            model = TimeSeriesForecasting(df=df, act_st=act_st, fcst_st=d, fcst_pr=fcst_pr, ext=ext, ext_lag=ext_lag)
            for m in test_model:
                runitem = {"batch": batch_no, "id": x, "testdate": d, "model": m}
                try:
                    st_time = datetime.datetime.now()
                    r = model.forecast(m)
                    r = r.rename(columns={'y': 'forecast'})
                    r['time'] = (datetime.datetime.now() - st_time).total_seconds()
                    r['id'] = x
                    r['dsr'] = d
                    r['period'] = np.arange(pr_st, len(r)+pr_st)
                    r['model'] = m
                    r = r[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'time']]
                    df_r = df_r.append(r, ignore_index = True)
                except Exception as e:
                    error_item = "batch: {} | id: {} | testdate: {} | model:{}".format(
                        runitem.get('batch'), runitem.get('id'), runitem.get('testdate').strftime("%Y-%m-%d"), runitem.get('model'))
                    error_txt = "ERROR: {} ({})".format(str(e), error_item)
                    self.lg.logtxt(error_txt, error=True)
        return df_r

    def validate(self, output_dir, act_st, test_st, test_pr, test_model, fcst_pr, pr_st, chunk_sz, cpu):
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
        self.fp.writecsv(self.df, "{}input_actual.csv".format(output_dir))
        # write external features
        if self.ext is not None:
            self.fp.writecsv(self.ext, "{}input_external.csv".format(output_dir))
            self.fp.writecsv(self.ext_lag, "{}input_externallag.csv".format(output_dir))
            self.lg.logtxt("write input file: {}input_actual.csv | {}input_external.csv | {}input_externallag.csv".format(output_dir,output_dir,output_dir))
        else:
            self.lg.logtxt("write input file: {}input_actual.csv".format(output_dir))
        # set parameter
        items = self.df['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        test_date = [x.to_pydatetime() + datetime.timedelta(days=+test_st.day-1) for x in pd.date_range(start=test_st, periods=test_pr, freq='MS')]
        self.lg.logtxt("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # loop by chunk
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            if cpu_count==1:
                for r in [self.validate_byitem(x, act_st, test_date, test_model, fcst_pr, pr_st, i) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.validate_byitem, [[x, act_st, test_date, test_model, fcst_pr, pr_st, i] for x in c]):
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
    
    def loaddata(self, act_path, fcst_path, ext_path=None, extlag_path=None):
        """Load data for validation process
        Parameters
        ----------
        act_path : str
            historical data path
        fcst_path : str
            forecast log path
        ext_path : str
            external features path
        extlag_path : str
            external lag path
        """
        # load actual and forecast data
        col_id, col_ds, col_y, col_mth, col_model, col_fcst = 'id', 'ds', 'y', 'mth', 'model', 'forecast'
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        df_act = pd.read_csv(self.fp.loadfile(act_path), parse_dates=['ds'], date_parser=dateparse)
        df_fcstlog = pd.read_csv(self.fp.loadfile(fcst_path), parse_dates=['ds'], date_parser=dateparse)
        self.df_act = df_act.rename(columns={col_id:'id', col_ds:'ds', col_y:'y'})
        self.df_fcstlog = df_fcstlog.rename(columns={col_id:'id', col_ds:'ds', col_mth:'mth', col_model:'model', col_fcst:'forecast'})
        # load external features
        if ext_path is not None:
            col_yid, col_extid, col_lag = 'y_id', 'ext_id', 'lag'
            ext = pd.read_csv(self.fp.loadfile(ext_path), parse_dates=['ds'], date_parser=dateparse)
            ext_lag = pd.read_csv(self.fp.loadfile(extlag_path), date_parser=dateparse)
            self.ext = ext.rename(columns={col_id: 'id', col_ds: 'ds', col_y: 'y'})[['id', 'ds', 'y']]
            self.ext_lag = ext_lag.rename(columns={col_yid: 'y_id', col_extid: 'ext_id', col_lag: 'lag'})[['y_id', 'ext_id', 'lag']]
            self.lg.logtxt("load data: {} | {} | {} | {}".format(act_path, fcst_path, ext_path, extlag_path))
        else:
            self.ext = None
            self.ext_lag = None
            self.lg.logtxt("load data: {} | {}".format(act_path, fcst_path))

    def forecast_byitem(self, x, act_st, fcst_st, fcst_pr, model_list, pr_st, batch_no):
        """Forecast data by item for parallel computing"""
        df = self.df_act[self.df_act['id']==x].copy()
        if self.ext is not None:
            ext = self.ext[['id', 'ds', 'y']].copy()
            ext_lag = self.ext_lag[self.ext_lag['y_id']==x].rename(columns={'ext_id': 'id'})[['id', 'lag']].copy()
        else:
            ext = None
            ext_lag = None
        model = TimeSeriesForecasting(df=df, act_st=act_st, fcst_st=fcst_st, fcst_pr=fcst_pr, ext=ext, ext_lag=ext_lag)
        df_r = pd.DataFrame()
        for m in model_list:
            try:
                runitem = {"batch": batch_no, "id": x, "model": m}
                st_time = datetime.datetime.now()
                r = model.forecast(m)
                r = r.rename(columns={'y': 'forecast'})
                r['time'] = (datetime.datetime.now() - st_time).total_seconds()
                r['id'] = x
                r['dsr'] = fcst_st
                r['model'] = m
                r['period'] = np.arange(pr_st, len(r)+pr_st)
                r = r[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'time']]
                df_r = df_r.append(r, ignore_index = True)
            except Exception as e:
                error_item = "batch: {} | id: {} | model:{}".format(runitem.get('batch'), runitem.get('id'), runitem.get('model'))
                error_txt = "ERROR: {} ({})".format(str(e), error_item)
                self.lg.logtxt(error_txt, error=True)
        return df_r

    def rankmodel_byitem(self, x, fcst_model, act_st, fcst_st, test_type, test_st):
        """Rank model based on historical forecast"""
        df_act = self.df_act[self.df_act['id']==x].copy()
        df_fcstlog = self.df_fcstlog[self.df_fcstlog['id']==x].copy()
        df_act = df_act[(df_act['ds']>=act_st) & (df_act['ds']<fcst_st)].copy()
        df_fcstlog = df_fcstlog[(df_fcstlog['ds']>=test_st) & (df_fcstlog['ds']<fcst_st)].copy()
        df_rank = df_fcstlog.copy()
        # select only in config file
        df_rank['val'] = df_rank['period'].map(fcst_model)
        df_rank = df_rank[df_rank['val'].notnull()].copy()
        df_rank['val'] = df_rank.apply(lambda x: True if x['model'] in x['val'] else False, axis=1)
        df_rank = df_rank[df_rank['val']==True].copy()
        # calculate error comparing with actual
        act = df_act if test_type == 'daily' else TimeSeriesForecasting.daytomth(df_act)
        df_rank = pd.merge(df_rank, act, on=['ds'], how='left')
        df_rank = df_rank.rename(columns={'y': 'actual'})
        df_rank['error'] = df_rank.apply(lambda x: mape(x['actual'], x['forecast']), axis=1)
        df_rank = df_rank[df_rank['error'].notnull()]
        df_rank = df_rank.groupby(['id', 'period', 'model'], as_index=False).agg({'error':'mean'})
        # ranking error
        df_rank['rank'] = df_rank.groupby('period')['error'].rank(method='first', ascending=True)
        return df_rank
        
    def forecast(self, output_dir, act_st, fcst_st, fcst_model, test_type, test_bck, pr_st, chunk_sz, cpu):
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
        test_type : {'monthly', 'daily}
            type of testing back error by month or day
        test_bck : int
            number of months to test back
        pr_st : int
            starting period for each forecast (default 0/1)
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
        if self.ext is not None:
            self.fp.writecsv(self.ext, "{}input_external.csv".format(output_dir))
            self.fp.writecsv(self.ext_lag, "{}input_externallag.csv".format(output_dir))
            self.lg.logtxt("write input file: {}input_actual.csv | {}input_forecast.csv | {}input_external.csv | {}input_externallag.csv".format(output_dir,output_dir,output_dir,output_dir))
        else:
            self.lg.logtxt("write input file: {}input_actual.csv | {}input_forecast.csv".format(output_dir, output_dir))
        self.runitem = {}
        # set parameter
        items = self.df_act['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        act_st = datetime.datetime.combine(act_st, datetime.datetime.min.time())
        fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        test_st = fcst_st + relativedelta(months=-test_bck)
        fcst_pr = len(fcst_model.keys())
        model_list = list(set(b for a in fcst_model.values() for b in a))
        self.lg.logtxt("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # forecast
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logtxt("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            df_rank = pd.DataFrame()
            if cpu_count==1:
                for r in [self.forecast_byitem(x, act_st, fcst_st, fcst_pr, model_list, pr_st, i) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
                for r in [self.rankmodel_byitem(x, fcst_model, act_st, fcst_st, test_type, test_st) for x in c]:
                    df_rank = df_rank.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.forecast_byitem, [[x, act_st, fcst_st, fcst_pr, model_list, pr_st, i] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.rankmodel_byitem, [[x, fcst_model, act_st, fcst_st, test_type, test_st] for x in c]):
                    df_rank = df_rank.append(r, ignore_index = True)
            # find best model
            df_sl = pd.merge(df_fcst, df_rank, on=['id', 'period', 'model'], how='left')
            df_sl = df_sl[df_sl['rank']==1].copy()
            df_sl = df_sl[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'error', 'time']]
            df_sl = df_sl.sort_values(by=['id', 'ds'], ascending=True).reset_index(drop=True)
            # write forecast result
            fcst_path = "{}output_forecast_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_fcst, fcst_path)
            # write selection result
            selection_path = "{}output_selection_{:04d}-{:04d}.csv".format(output_dir, i, n_chunk)
            self.fp.writecsv(df_sl, selection_path)
            self.lg.logtxt("write output file ({}/{}): {} | {}".format(i, n_chunk, fcst_path, selection_path))
        self.lg.logtxt("[END FORECAST]")
        self.lg.writelog("{}logfile.log".format(output_dir))
