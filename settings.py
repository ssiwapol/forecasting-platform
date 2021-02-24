import os

import yaml


tmp_dir = 'tmp'
module_file = {
    'forecast-init':{
        'y': {
            'require': True,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'y': float},
            'filename': 'input_y.csv'
            },
        'x': {
            'require': False,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'x': float},
            'filename': 'input_x.csv'
            }, 
        'lag': {
            'require': False,
            'filetype': 'csv',
            'dtypes': {'id_y': str, 'id_x': str, 'lag': float},
            'filename': 'input_lag.csv'
            }
        }, 
    'forecast-prod': {
        'y': {
            'require': True,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'y': float},
            'filename': 'input_y.csv'
            },
        'x': {
            'require': False,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'x': float},
            'filename': 'input_x.csv'
            }, 
        'lag': {
            'require': False,
            'filetype': 'csv',
            'dtypes': {'id_y': str, 'id_x': str, 'lag': float},
            'filename': 'input_lag.csv'
            },
        'forecastlog': {
            'require': True,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'dsr': 'ds', 'period': float, 'model': str, 'forecast': float, 'time': float},
            'filename': 'input_forecastlog.csv'
            }
        },
    'feat-selection': {
        'y': {
            'require': True,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'y': float},
            'filename': 'input_y.csv'
            },
        'x': {
            'require': True,
            'filetype': 'csv',
            'dtypes': {'id': str, 'ds': 'ds', 'x': float},
            'filename': 'input_x.csv'
            }
        },
    }
module_config_type = {
    "forecast-init": {
        "test_start": "ds",
        "test_end": "ds",
        "test_model": list,
        "forecast_period": int,
        "forecast_frequency": ["d", "m", "q", "y"],
        "period_start": int,
        "top_model": int,
        "test_back": int, 
        "forecast_ensemble": ["mean", "median"], 
        "error_ensemble": ["mean", "median"], 
        "error_display": ["mae", "mape"],
        "chunk_size": int,
        "cpu": int
    },
    "forecast-prod": {
        "forecast_start": "ds",
        "forecast_period": int,
        "forecast_frequency": ["d", "m", "q", "y"],
        "period_start": int,
        "top_model": int,
        "test_back": int, 
        "forecast_ensemble": ["mean", "median"], 
        "error_ensemble": ["mean", "median"], 
        "error_display": ["mae", "mape"],
        "chunk_size": int,
        "cpu": int
    },
    "feat-selection": {
        "model_name": ["aicc01", "aic01", "r2adj01"],
        "frequency": ["d", "m", "q", "y"],
        "growth": bool,
        "min_data_points": int,
        "min_lag": int, 
        "max_lag": int, 
        "max_features": int, 
        "chunk_size": int,
        "cpu": int
    }
}

with open('config.yaml') as f:
    app_config = yaml.load(f, Loader=yaml.Loader)
job_db = os.path.join(tmp_dir, app_config['database'])
