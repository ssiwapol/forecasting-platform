# -*- coding: utf-8 -*-
import os
import argparse
import traceback

import yaml

from mod import ForecastInit, ForecastProd, FeatSelectionBatch
from utils import update_status
from settings import app_config, job_db, tmp_dir

parser = argparse.ArgumentParser()
parser.add_argument('module', action='store', help='Running module')
parser.add_argument('job', action='store', help='Job ID')
args = parser.parse_args()


if __name__=="__main__":
    job_id = args.job
    output_dir = os.path.join(tmp_dir, job_id)
    log_path = os.path.join(tmp_dir, job_id, 'logfile.log')
    log_tz = app_config['timezone']
    log_stream = app_config['log_stream']
    with open(os.path.join(tmp_dir, job_id, 'config.yaml')) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    update_status(job_db, job_id, 'running')

    # run initial forecast
    if args.module == 'forecast-init':
        y_path = os.path.join(tmp_dir, job_id, 'input_y.csv')
        x_path = os.path.join(tmp_dir, job_id, 'input_x.csv')
        x_path = x_path if os.path.isfile(x_path) else None
        lag_path = os.path.join(tmp_dir, job_id, 'input_lag.csv')
        lag_path = lag_path if os.path.isfile(lag_path) else None
        f = ForecastInit(output_dir, log_path, log_tz, log_stream)
        try:
            f.lg.logger.info('job id: {}'.format(job_id))
            f.lg.logger.info('config: {}'.format(conf))
            f.loaddata(y_path, x_path, lag_path)
            f.run(conf['test_start'], conf['test_end'], conf['test_model'], \
                conf['forecast_period'], conf['forecast_frequency'], conf['period_start'], \
                conf['chunk_size'], conf['cpu'])
            f.evaluate(conf['top_model'], conf['test_back'], conf['forecast_ensemble'], conf['error_ensemble'], conf['error_display'])
            update_status(job_db, job_id, 'finish')
        except Exception:
            f.lg.logger.error(str(traceback.format_exc()))
            update_status(job_db, job_id, 'error')

    # run production forecast
    if args.module == 'forecast-prod':
        y_path = os.path.join(tmp_dir, job_id, 'input_y.csv')
        x_path = os.path.join(tmp_dir, job_id, 'input_x.csv')
        x_path = x_path if os.path.isfile(x_path) else None
        lag_path = os.path.join(tmp_dir, job_id, 'input_lag.csv')
        lag_path = lag_path if os.path.isfile(lag_path) else None
        fcstlog_path = os.path.join(tmp_dir, job_id, 'input_forecastlog.csv')
        f = ForecastProd(output_dir, log_path, log_tz, log_stream)
        try:
            f.lg.logger.info('job id: {}'.format(job_id))
            f.lg.logger.info('config: {}'.format(conf))
            f.loaddata(y_path, fcstlog_path, x_path, lag_path)
            f.run(conf['forecast_start'], conf['forecast_period'], conf['forecast_frequency'], conf['period_start'], \
                conf['top_model'], conf['test_back'], conf['forecast_ensemble'], conf['error_ensemble'], conf['error_display'], \
                conf['chunk_size'], conf['cpu'])
            update_status(job_db, job_id, 'finish')
        except Exception:
            f.lg.logger.error(str(traceback.format_exc()))
            update_status(job_db, job_id, 'error')

    # run feature selection
    if args.module == 'feat-selection':
        y_path = os.path.join(tmp_dir, job_id, 'input_y.csv')
        x_path = os.path.join(tmp_dir, job_id, 'input_x.csv')
        f = FeatSelectionBatch(output_dir, log_path, log_tz, log_stream)
        try:
            f.lg.logger.info('job id: {}'.format(job_id))
            f.lg.logger.info('config: {}'.format(conf))
            f.loaddata(x_path, y_path)
            f.run(conf['model_name'], conf['frequency'], conf['growth'], \
                conf['min_data_points'], conf['min_lag'], conf['max_lag'], conf['max_features'], \
                conf['chunk_size'], conf['cpu'])
            update_status(job_db, job_id, 'finish')
        except Exception:
            f.lg.logger.error(str(traceback.format_exc()))
            update_status(job_db, job_id, 'error')
