# -*- coding: utf-8 -*-
import os
import argparse

import yaml

from mod import Validation, Forecasting
from utils import FilePath
from app import tmp_dir, config_path, cloudauth_path

parser = argparse.ArgumentParser()
parser.add_argument('run', action='store', help='Running type')
parser.add_argument('runpath', action='store', help='Configuration path')
args = parser.parse_args()


if __name__=="__main__":
    # load config file
    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    fp = FilePath(conf['PLATFORM'], cloud_auth=cloudauth_path)

    # load running config file
    with fp.loadfile(args.runpath) as f:
        r = yaml.load(f, Loader=yaml.Loader)

    # add prefix if run on local
    if conf['PLATFORM'] == "local":
        path_list = ['ACT_PATH', 'EXT_PATH', 'EXTLAG_PATH', 'FCST_PATH', 'OUTPUT_DIR']
        r = dict((k, os.path.join(tmp_dir, v)) if k in path_list and v is not None else (k, v) for k, v in r.items())

    # run validate
    if args.run == "validate":
        v = Validation(conf['PLATFORM'], conf['LOG_TAG'], conf['TIMEZONE'], cloudauth_path)
        try:
            v.lg.logtxt("run detail: {}".format(r))
            v.loaddata(r['ACT_PATH'], r['EXT_PATH'], r['EXTLAG_PATH'])
            v.validate(
                r['OUTPUT_DIR'], r['ACT_START'], 
                r['TEST_START'], r['TEST_PERIOD'], r['TEST_MODEL'], 
                r['FCST_PERIOD'], r['FCST_FREQ'], r['PERIOD_START'], 
                r['CHUNKSIZE'], r['CPU']
                )
        except Exception as e:
            v.lg.logtxt("ERROR: {}".format(str(e)), error=True)

    # run forecast
    elif args.run == "forecast":
        f = Forecasting(conf['PLATFORM'], conf['LOG_TAG'], conf['TIMEZONE'], cloudauth_path)
        try:
            f.lg.logtxt("run detail: {}".format(r))
            f.loaddata(r['ACT_PATH'], r['FCSTLOG_PATH'], r['EXT_PATH'], r['EXTLAG_PATH'])
            f.forecast(
                r['OUTPUT_DIR'], r['ACT_START'], r['FCST_START'], r['FCST_MODEL'], 
                r['TEST_TYPE'], r['TEST_BACK'], 
                r['TOP_MODEL'], r['ENSEMBLE_METHOD'], 
                r['CHUNKSIZE'], r['CPU']
                )
        except Exception as e:
            f.lg.logtxt("ERROR: {}".format(str(e)), error=True)

    else:
        pass
