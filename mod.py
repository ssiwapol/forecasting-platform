# -*- coding: utf-8 -*-
import os
import datetime
import multiprocessing
import warnings
import traceback

import numpy as np
import pandas as pd

from model import TimeSeriesForecasting, EnsembleModel, FeatSelection
from utils import Logging, chunker, zip_output


warnings.filterwarnings("ignore")


class ForecastInit:
    """Rolling forecast and evaluate result
    Init Parameters
    ----------
    output_dir : str
        directory to write output
    log_path : str
        log path to write log
    log_tz : str (e.g. Asia/Bangkok)
        timezone for logging
    log_stream : True/False
        stream log or not
    """
    def __init__(self, output_dir, log_path, log_tz, log_stream=True):
        # init variable
        self.output_dir = output_dir
        self.dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        self.lg = Logging(log_path, log_tz, stream=log_stream)
        self.lg.logger.info("[START INITIAL FORECAST]")

    def loaddata(self, y_path, x_path=None, lag_path=None):
        """Load data for validation process
        Parameters
        ----------
        y_path : str
            historical data path
        x_path : str
            external features path
        lag_path : str
            external lag path
        """
        # load input_y data
        df_y = pd.read_csv(y_path, dtype={'id': str}, parse_dates=['ds'], date_parser=self.dateparse, encoding='utf-8')
        self.df_y = df_y[['id', 'ds', 'y']]
        # load input_x, input_lag data
        if x_path is not None:
            df_x = pd.read_csv(x_path, dtype={'id': str}, parse_dates=['ds'], date_parser=self.dateparse, encoding='utf-8')
            df_lag = pd.read_csv(lag_path, dtype={'id_y': str, 'id_x': str}, encoding='utf-8')
            self.df_x = df_x[['id', 'ds', 'x']]
            self.df_lag = df_lag[['id_y', 'id_x', 'lag']]
            self.lg.logger.info("load data: {} | {} | {}".format(os.path.basename(y_path), os.path.basename(x_path), os.path.basename(lag_path)))
        else:
            self.df_x = None
            self.df_lag = None
            self.lg.logger.info("load data: {}".format(os.path.basename(y_path)))

    def run(self, test_st, test_end, test_model, fcst_pr, fcst_freq, pr_st, chunk_sz, cpu):
        """Run forecasting and write result by batch
        Parameters
        ----------
        test_st : datetime
            test start date
        test_end : int
            test end date
        test_model : list
            list of model to test
        fcst_pr : int
            number of periods to forecast for each rolling
        fcst_freq : {"d", "m", "q", "y"}, optional
            forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
        pr_st : int
            starting period for each forecast (default 0/1)
        chunk_sz : int
            number of item to run for each chunk
        cpu : int
            number of running processors
        """
        self.lg.logger.info("start forecasting")
        # set parameter
        self.fcst_freq = fcst_freq
        items = self.df_y['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        test_date = [x.to_pydatetime() + datetime.timedelta(days=+test_st.day-1) for x in pd.date_range(start=test_st, end=test_end, freq=TimeSeriesForecasting.freq_dict[fcst_freq])]
        self.lg.logger.info("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # forecast by chunk
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logger.info("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            if cpu_count==1:
                for r in [self.forecast_byitem(x, test_date, test_model, fcst_pr, fcst_freq, pr_st, i) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.forecast_byitem, [[x, test_date, test_model, fcst_pr, fcst_freq, pr_st, i] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
            # write csv file
            output_path = os.path.join(self.output_dir, "output_forecastlog_{:04d}-{:04d}.csv".format(i, n_chunk))
            df_fcst.to_csv(output_path, encoding='utf-8', index=False)
            self.lg.logger.info("write output file ({}/{}): {}".format(i, n_chunk, os.path.basename(output_path)))
        # combine forecast to single file and write to csv
        files = [f.path for f in os.scandir(self.output_dir) if f.is_file() and os.path.basename(f).startswith("output_forecastlog_")]
        df_fcstlog = [pd.read_csv(f, dtype={'id': str}, parse_dates=['ds', 'dsr'], date_parser=self.dateparse, encoding='utf-8') for f in files]
        self.df_fcstlog = pd.concat(df_fcstlog, ignore_index=True)
        fcstlog_path = os.path.join(self.output_dir, "output_forecastlog.csv")
        self.df_fcstlog.to_csv(fcstlog_path, encoding='utf-8', index=False)
        self.lg.logger.info("write output file: {}".format(os.path.basename(fcstlog_path)))
        self.lg.logger.info("end forecasting")

    def forecast_byitem(self, id_y, test_date, test_model, fcst_pr, fcst_freq, pr_st, batch_no):
        """Forecast by item for parallel computing"""
        df_y = self.df_y[self.df_y['id']==id_y][['ds', 'y']].copy()
        if self.df_x is not None:
            df_x = self.df_x[['id', 'ds', 'x']].copy()
            df_lag = self.df_lag[self.df_lag['id_y']==id_y].rename(columns={'id_x': 'id'})[['id', 'lag']].copy()
        else:
            df_x = None
            df_lag = None
        df_r = pd.DataFrame()
        for d in test_date:
            model = TimeSeriesForecasting(df_y=df_y, act_st=df_y['ds'].min(), fcst_st=d, fcst_pr=fcst_pr, fcst_freq=fcst_freq, df_x=df_x, df_lag=df_lag)
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
                    error_txt = "{} ({})".format(error_item, str(traceback.format_exc()))
                    self.lg.logger.error(error_txt)
        return df_r

    def evaluate(self, top_model=3, test_back=6, fcst_ens='mean', error_ens='mean', error_dsp='mape'):
        """Evaluate forecasting result
        Parameters
        ----------
        top_model : int
            number of top models to ensemble
        test_back : int
            number of periods to test back
        fcst_ens : {"mean", "median"}, optional
            method to ensemble forecast
        error_ens : {"mae", "mape"}, optional
            method to ensemble error
        error_dsp : {"mae", "mape"}, optional
            display error by
        """
        # ensemble model for all dsr in forecasting log
        self.lg.logger.info("start forecast evaluation")
        em = EnsembleModel(self.df_y, self.df_fcstlog, self.fcst_freq)
        em.rank(test_back, error_ens, error_dsp)
        df_ens = em.ensemble_fcstlog(top_model, fcst_ens, error_ens)
        # write ensemble result
        fcst_path = os.path.join(self.output_dir, "output_forecast.csv")
        self.df_fcst = df_ens.copy()
        self.df_fcst.to_csv(fcst_path, encoding='utf-8', index=False)
        self.lg.logger.info("write output file: {}".format(os.path.basename(fcst_path)))
        # evaluation result from forecasting log
        df_eval1 = pd.merge(em.df_fcstlog, em.df_act.rename(columns={'y': 'actual'}), on=['id', 'ds'], how='left')
        df_eval1['mae'] = df_eval1.apply(lambda x: EnsembleModel.cal_error(x['actual'], x['forecast'], 'mae'), axis=1)
        df_eval1['mape'] = df_eval1.apply(lambda x: EnsembleModel.cal_error(x['actual'], x['forecast'], 'mape'), axis=1)
        df_eval1['y'] = df_eval1['forecast']
        df_eval1['y_type'] = 'forecast'
        # evaluation result from ensemble model
        df_eval2 = pd.merge(df_ens, em.df_act.rename(columns={'y': 'actual'}), on=['id', 'ds'], how='left')
        df_eval2['mae'] = df_eval2.apply(lambda x: em.cal_error(x['actual'], x['forecast'], 'mae'), axis=1)
        df_eval2['mape'] = df_eval2.apply(lambda x: em.cal_error(x['actual'], x['forecast'], 'mape'), axis=1)
        df_eval2['model'] = "top{}".format(top_model)
        df_eval2['y'] = df_eval2['forecast']
        df_eval2['y_type'] = 'forecast'
        # evaluation result from actual data
        df_eval3 = em.df_act.copy()
        df_eval3['y_type'] = 'actual'
        # finalize evaluation result
        df_eval = pd.concat([df_eval1, df_eval2, df_eval3])
        df_eval = df_eval[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'actual', 'mae', 'mape', 'top_model', 'test_back', 'y', 'y_type']]
        # write evaluation result
        eval_path = os.path.join(self.output_dir, "output_evaluate.csv")
        self.df_eval = df_eval.copy()
        self.df_eval.to_csv(eval_path, encoding='utf-8', index=False)
        self.lg.logger.info("write output file: {}".format(os.path.basename(eval_path)))
        self.lg.logger.info("end forecast evaluation")
        self.lg.logger.info("[END INITIAL FORECAST]")
        # zip output
        zip_path = os.path.join(self.output_dir, "{}.zip".format(os.path.basename(self.output_dir)))
        zip_output(self.output_dir, zip_path)
        self.lg.logger.info("write output file: {}".format(os.path.basename(zip_path)))


class ForecastProd:
    """Forecast next period by top model selection
    Init Parameters
    ----------
    output_dir : str
        directory to write output
    log_path : str
        log path to write log
    log_tz : str (e.g. Asia/Bangkok)
        timezone for logging
    log_stream : True/False
        stream log or not
    """
    def __init__(self, output_dir, log_path, log_tz, log_stream=True):
        # init variable
        self.output_dir = output_dir
        self.dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        self.lg = Logging(log_path, log_tz, stream=log_stream)
        self.lg.logger.info("[START PRODUCTION FORECAST]")

    def loaddata(self, y_path, fcstlog_path, x_path=None, lag_path=None):
        """Load data for validation process
        Parameters
        ----------
        y_path : str
            historical data path
        fcstlog_path : str
            historical forecasting result
        x_path : str
            external features path
        lag_path : str
            external lag path
        """
        # load input_y data
        df_y = pd.read_csv(y_path, dtype={'id': str}, parse_dates=['ds'], date_parser=self.dateparse, encoding='utf-8')
        self.df_y = df_y[['id', 'ds', 'y']]
        # load input_fcstlog data
        df_fsctlog = pd.read_csv(fcstlog_path, dtype={'id': str}, parse_dates=['ds', 'dsr'], date_parser=self.dateparse, encoding='utf-8')
        self.df_fsctlog = df_fsctlog[['id', 'ds', 'dsr', 'period', 'model', 'forecast', 'time']]
        # load input_x, input_lag data
        if x_path is not None:
            df_x = pd.read_csv(x_path, dtype={'id': str}, parse_dates=['ds'], date_parser=self.dateparse, encoding='utf-8')
            df_lag = pd.read_csv(lag_path, dtype={'id_y': str, 'id_x': str}, encoding='utf-8')
            self.df_x = df_x[['id', 'ds', 'x']]
            self.df_lag = df_lag[['id_y', 'id_x', 'lag']]
            self.lg.logger.info("load data: {} | {} | {}".format(os.path.basename(y_path), os.path.basename(x_path), os.path.basename(lag_path)))
        else:
            self.df_x = None
            self.df_lag = None
            self.lg.logger.info("load data: {}".format(os.path.basename(y_path)))

    def run(self, fcst_st, fcst_pr, fcst_freq, pr_st, top_model, test_back, fcst_ens, error_ens, error_dsp, chunk_sz, cpu):
        """Run forecasting and write result by batch
        Parameters
        ----------
        test_st : datetime
            forecast date
        fcst_pr : int
            number of periods to forecast for each rolling
        fcst_freq : {"d", "m", "q", "y"}, optional
            forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
        pr_st : int
            starting period for each forecast (default 0/1)
        top_model : int
            number of top models to ensemble
        test_back : int
            number of periods to test back
        fcst_ens : {"mean", "median"}, optional
            method to ensemble forecast
        error_ens : {"mae", "mape"}, optional
            method to ensemble error
        error_dsp : {"mae", "mape"}, optional
            display error by
        chunk_sz : int
            number of item to run for each chunk
        cpu : int
            number of running processors
        """
        self.lg.logger.info("start forecasting")
        # set parameter
        items = self.df_y['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        fcst_st = datetime.datetime.combine(fcst_st, datetime.datetime.min.time())
        self.lg.logger.info("total items: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # rank the models
        self.df_fsctlog = self.df_fsctlog[self.df_fsctlog['dsr'] < fcst_st]
        fcst_model = list(self.df_fsctlog['model'].unique())
        em = EnsembleModel(self.df_y, self.df_fsctlog, fcst_freq)
        em.rank(test_back, error_ens, error_dsp)
        # forecast
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logger.info("run at {} processor(s)".format(cpu_count))
        for i, c in enumerate(chunker(items, chunk_sz), 1):
            df_fcst = pd.DataFrame()
            if cpu_count==1:
                for r in [self.forecast_byitem(x, fcst_model, fcst_st, fcst_pr, fcst_freq, pr_st, i) for x in c]:
                    df_fcst = df_fcst.append(r, ignore_index = True)
            else:
                pool = multiprocessing.Pool(processes=cpu_count)
                for r in pool.starmap(self.forecast_byitem, [[x, fcst_model, fcst_st, fcst_pr, fcst_freq, pr_st, i] for x in c]):
                    df_fcst = df_fcst.append(r, ignore_index = True)
                pool.close()
                pool.join()
            # write forecasting result
            output_path = os.path.join(self.output_dir, "output_forecastlog_{:04d}-{:04d}.csv".format(i, n_chunk))
            df_fcst.to_csv(output_path, encoding='utf-8', index=False)
            self.lg.logger.info("write output file ({}/{}): {}".format(i, n_chunk, os.path.basename(output_path)))
            # ensemble forecast result
            df_ens = em.ensemble(df_fcst, top_model, fcst_ens, error_ens)
            output_path = os.path.join(self.output_dir, "output_forecast_{:04d}-{:04d}.csv".format(i, n_chunk))
            df_ens.to_csv(output_path, encoding='utf-8', index=False)
            self.lg.logger.info("write output file ({}/{}): {}".format(i, n_chunk, os.path.basename(output_path)))
        # combine forecasting log to single file and write to csv
        files = [f.path for f in os.scandir(self.output_dir) if f.is_file() and os.path.basename(f).startswith("output_forecastlog_")]
        df_fcstlog = [pd.read_csv(f, dtype={'id': str}, parse_dates=['ds', 'dsr'], date_parser=self.dateparse, encoding='utf-8') for f in files]
        self.df_fcstlogall = pd.concat(df_fcstlog + [self.df_fsctlog], ignore_index=True)
        fcstlogall_path = os.path.join(self.output_dir, "output_forecastlog.csv")
        self.df_fcstlogall.to_csv(fcstlogall_path, encoding='utf-8', index=False)
        self.lg.logger.info("write output file: {}".format(os.path.basename(fcstlogall_path)))
        # combine forecasting result to single file and write to csv
        files = [f.path for f in os.scandir(self.output_dir) if f.is_file() and os.path.basename(f).startswith("output_forecast_")]
        df_fcst = [pd.read_csv(f, dtype={'id': str}, parse_dates=['ds', 'dsr'], date_parser=self.dateparse, encoding='utf-8') for f in files]
        self.df_fcst = pd.concat(df_fcst, ignore_index=True)
        fcst_path = os.path.join(self.output_dir, "output_forecast.csv")
        self.df_fcst.to_csv(fcst_path, encoding='utf-8', index=False)
        self.lg.logger.info("write output file: {}".format(os.path.basename(fcst_path)))
        self.lg.logger.info("end forecasting")
        self.lg.logger.info("[END PRODUCTION FORECAST]")
        # zip output
        zip_path = os.path.join(self.output_dir, "{}.zip".format(os.path.basename(self.output_dir)))
        zip_output(self.output_dir, zip_path)
        self.lg.logger.info("write output file: {}".format(os.path.basename(zip_path)))

    def forecast_byitem(self, id_y, fcst_model, fcst_st, fcst_pr, fcst_freq, pr_st, batch_no):
        """Forecast data by item for parallel computing"""
        df_y = self.df_y[self.df_y['id']==id_y].copy()
        if self.df_x is not None:
            df_x = self.df_x[['id', 'ds', 'x']].copy()
            df_lag = self.df_lag[self.df_lag['id_y']==id_y].rename(columns={'id_x': 'id'})[['id', 'lag']].copy()
        else:
            df_x = None
            df_lag = None
        model = TimeSeriesForecasting(df_y=df_y, act_st=df_y['ds'].min(), fcst_st=fcst_st, fcst_pr=fcst_pr, fcst_freq=fcst_freq, df_x=df_x, df_lag=df_lag)
        df_r = pd.DataFrame()
        for m in fcst_model:
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
                error_txt = "{} ({})".format(error_item, str(traceback.format_exc()))
                self.lg.logger.error(error_txt)
        return df_r


class FeatSelectionBatch:
    """Feature selection by ID
    Init Parameters
    ----------
    output_dir : str
        directory to write output
    log_path : str
        log path to write log
    log_tz : str (e.g. Asia/Bangkok)
        timezone for logging
    log_stream : True/False
        stream log or not
    """
    def __init__(self, output_dir, log_path, log_tz, log_stream=True):
        # init variable
        self.output_dir = output_dir
        self.dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        self.lg = Logging(log_path, log_tz, stream=log_stream)
        self.lg.logger.info("[START FEATURE SELECTION]")
    
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
        df_x = pd.read_csv(x_path, dtype={'id': str}, parse_dates=['ds'], date_parser=self.dateparse)
        df_y = pd.read_csv(y_path, dtype={'id': str}, parse_dates=['ds'], date_parser=self.dateparse)
        self.df_x = df_x[['id', 'ds', 'x']]
        self.df_y = df_y[['id', 'ds', 'y']]
        self.lg.logger.info("load data: {} | {}".format(os.path.basename(x_path), os.path.basename(y_path)))

    def run(self, model_name='aicc01', freq='m', growth=True, min_data_points=72, min_lag=2, max_lag=23, max_features=10, chunk_sz=1, cpu=1):
        """Forecast and write result by batch
        Parameters
        ----------
        model_name : str
            model to run feature selection
        freq : {"d", "m", "q", "y"}, optional
            data frequency (d-daily, m-monthly, q-quarterly, y-yearly)
        growth : True/False
            run model by growth or not
        min_data_points : int
            minimum data points for x-series
        min_lag : int
            minimum lag available for x-series
        max_lag : int
            maximum lag available for x-series
        max_features : int
            maximum total features for each y-series
        chunk_size : int
            number of item to run for each chunk
        cpu : int
            number of running processors
        """
        self.lg.logger.info("start selection")
        # initial model
        f = FeatSelection(freq, growth)
        f.set_x(self.df_x, min_data_points)
        self.lg.logger.info("total x-series: {}/{}".format(f.total_test_x, f.total_x))
        # separate chunk
        items = self.df_y['id'].unique()
        n_chunk = len([x for x in chunker(items, chunk_sz)])
        self.lg.logger.info("total y-series: {} | chunk size: {} | total chunk: {}".format(len(items), chunk_sz, n_chunk))
        # select based on y-series
        cpu_count = 1 if cpu<=1 else multiprocessing.cpu_count() if cpu>=multiprocessing.cpu_count() else cpu
        self.lg.logger.info("run at {} processor(s)".format(cpu_count))
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
            output_path = os.path.join(self.output_dir, "output_feature_{:04d}-{:04d}.csv".format(i, n_chunk))
            df_select.to_csv(output_path, encoding='utf-8', index=False)
            self.lg.logger.info("write output file ({}/{}): {}".format(i, n_chunk, os.path.basename(output_path)))
        # combine selection result single file and write to csv
        files = [f.path for f in os.scandir(self.output_dir) if f.is_file() and os.path.basename(f).startswith("output_feature_")]
        df_feat = [pd.read_csv(f, dtype={'id_y': str, 'id_x': str}, encoding='utf-8') for f in files]
        self.df_feat = pd.concat(df_feat, ignore_index=True)
        feat_path = os.path.join(self.output_dir, "output_feature.csv")
        self.df_feat.to_csv(feat_path, encoding='utf-8', index=False)
        self.lg.logger.info("write output file: {}".format(os.path.basename(feat_path)))
        self.lg.logger.info("end selection")
        self.lg.logger.info("[END FEATURE SELECTION]")
        # zip output
        zip_path = os.path.join(self.output_dir, "{}.zip".format(os.path.basename(self.output_dir)))
        zip_output(self.output_dir, zip_path)
        self.lg.logger.info("write output file: {}".format(os.path.basename(zip_path)))

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
            error_txt = "{} ({})".format(error_item, str(traceback.format_exc()))
            self.lg.logger.error(error_txt, error=True)
