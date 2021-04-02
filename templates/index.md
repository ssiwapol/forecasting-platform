# FORECASTING PLATFORM

**API Reference**

- [Run Module](#run-module)
    - [Request](#run-module-request)
    - [Modules](#run-module-module)
        1. [Initial Forecast](#initial-forecast)
        2. [Production Forecast](#production-forecast)
        3. [Feature Selection](#feature-selection)
    - [Response](#run-module-response)
    - [File](#run-module-file)
    - [Time series forecasting models](#time-series-forecasting-model)
    - [Feature Selection Model](#feature-selection-model)
- [Stop running job](#stop-job)
    - [Request](#stop-job-request)
    - [Response](#stop-job-response)
- [List all jobs](#list-job)
    - [Request](#list-job-request)
    - [Response](#list-job-response)
- [Get job status](#job-status)
    - [Request](#job-status-request)
    - [Response](#job-status-response)
- [Download output](#download-output)
    - [Request](#download-output-request)
    - [Response](#download-output-response)
- [Remarks](#remarks)
    - [Job status](#remarks-job-status)

---

# Run Module <a name="run-module"></a>

Run spcific forecasting module

## Request <a name="run-module-request"></a>

- **URL**: `/api/run`

- **Method**: `POST`


### Request headers

| Key    | Value      |
| :----- | :--------- |
| apikey | [AUTH_KEY] |

### Request body
form-data

| Key             | Type     | Description                              |
| :-------------- | :------- | :--------------------------------------- |
| module          | string   | module to run                            |
| config          | json     | running configuration                    |
| user (optional) | string   | user who run this module                 |
| y               | csv file | historical time series input data        |
| x               | csv file | external features time series data       |
| lag             | csv file | length of lagging from external features |
| forecastlog     | csv file | forecasting log                          |

## Module <a name="run-module-module"></a>

### Initial Forecast <a name="initial-forecast"></a>

Rolling forecast from start date to end date based on selected time series forecasting model.

- **Request body**

| Key             | Value                   |
| :-------------- | :---------------------- |
| module          | forecast-init           |
| config    |<pre lang="json">{<br>  "test_start": "[YYYY-MM-DD]",               # test start date<br>  "test_end": "[YYYY-MM-DD]",                 # test end date<br>  "test_model": ["[MODEL1]", "[MODEL2]"],     # list of time series models to run<br>  "forecast_period": [N],                     # forecasting period<br>  "forecast_frequency": "[d/m/q/y]",          # forecasting frequency (d-daily, m-monthly, q-quarterly, y-yearly)<br>  "period_start": [N],                        # number of periods to forecast for each rolling<br>  "top_model": [N],                           # top N best models to use<br>  "test_back": [N],                           # number of periods to test back<br>  "forecast_ensemble": "[mean/median]",       # ensemble method to combine the models<br>  "error_ensemble": "[mean/median]",          # ensemble method to test error<br>  "error_display": "[mae/mape]",              # method to display error<br>  "chunk_size": [N],                          # number of item to run for each chunk<br>  "cpu": [N]                                  # number of running processors<br>}</pre>|
| user (optional) | [USER]                  |
| y               | [PATH_TO_INPUT_Y.CSV]   |
| x (optional)    | [PATH_TO_INPUT_x.CSV]   |
| lag (optional)  | [PATH_TO_INPUT_LAG.CSV] |


- **Request body example**

| Key       | Value         |
| :-------- | :------------ |
| module    | forecast-init |
| config    |<pre lang="json">{<br>  "test_start": "2017-01-01",<br>  "test_end": "2020-12-01",<br>  "test_model": ["expo01", "expo02"],<br>  "forecast_period": 4,<br>  "forecast_frequency": "m",<br>  "period_start": 0,<br>  "top_model": 3,<br>  "test_back": 6,<br>  "forecast_ensemble": "mean",<br>  "error_ensemble": "mean",<br>  "error_display": "mape",<br>  "chunk_size": 5,<br>  "cpu": 1<br>}</pre>|
| user      | admin         |
| y         | input_y.csv   |
| x         | input_x.csv   |
| lag       | input_lag.csv |

- **Output file**
    - output_forecastlog.csv: forecasting log from all time series forecasting models for all rolling date
    - output_forecast.csv: top model forecasting result for all rolling date
    - output_evaluate.csv: combination of forecasting log (all models), forecast (top model) and actual data in file in order to evaluate forecasting result

### Production Forecast <a name="production-forecast"></a>

Next period forecast from top model selection based on last forecasting log. 

- **Request body**

| Key             | Value                           |
| :-------------- | :------------------------------ |
| module          | forecast-prod                   |
| user (optional) | [USER]                          |
| config    |<pre lang="json">{<br>  "forecast_start": "[YYYY-MM-DD]",           # forecast start date<br>  "forecast_period": [N],                     # forecasting period<br>  "forecast_frequency": "[d/m/q/y]",          # forecasting frequency (d-daily, m-monthly, q-quarterly, y-yearly)<br>  "period_start": [N],                        # number of periods to forecast for each rolling<br>  "top_model": [N],                           # top N best models to use<br>  "test_back": [N],                           # number of periods to test back<br>  "forecast_ensemble": "[mean/median]",       # ensemble method to combine the models<br>  "error_ensemble": "[mean/median]",          # ensemble method to test error<br>  "error_display": "[mae/mape]",              # method to display error<br>  "chunk_size": [N],                          # number of item to run for each chunk<br>  "cpu": [N]                                  # number of running processors<br>}</pre>|
| y               | [PATH_TO_INPUT_Y.CSV]           |
| forecastlog     | [PATH_TO_INPUT_FORECASTLOG.CSV] |
| x (optional)    | [PATH_TO_INPUT_x.CSV]           |
| lag (optional)  | [PATH_TO_INPUT_LAG.CSV]         |

- **Request body example**

| Key         | Value                 |
| :---------- | :-------------------- |
| module      | forecast-prod         |
| config    |<pre lang="json">{<br>  "forecast_start": "2021-01-01",<br>  "forecast_period": 4,<br>  "forecast_frequency": "m",<br>  "period_start": 0,<br>  "top_model": 3,<br>  "test_back": 6,<br>  "forecast_ensemble": "mean",<br>  "error_ensemble": "mean",<br>  "error_display": "mape",<br>  "chunk_size": 5,<br>  "cpu": 1<br>}</pre>|
| user        | admin                 |
| y           | input_y.csv           |
| forecastlog | input_forecastlog.csv |
| x           | input_x.csv           |
| lag         | input_lag.csv         |

- **Output file**: 
    - output_forecastlog.csv: forecasting log from all time series forecasting models for all rolling date (input + latest output)
    - output_forecast.csv: top model forecasting result


### Feature Selection <a name="feature-selection"></a>

Select x-series (external features) and their lag periods which are the most correlated for forecasting each y-series.

- **Request body**

| Key             | Value                           |
| :-------------- | :------------------------------ |
| module          | feat-selection                  |
| config    |<pre lang="json">{<br>  "model_name": "[MODEL]",         # model to run feature selection<br>  "frequency": "[d/m/q/y]",        # data frequency (d-daily, m-monthly, q-quarterly, y-yearly)<br>  "growth": [true/false],          # run model by growth or not<br>  "min_data_points": [N],          # minimum data points for x-series<br>  "min_lag": [N],                  # minimum lag available for x-series<br>  "max_lag": [N],                  # maximum lag available for x-series<br>  "max_features": [N],             # maximum total features for each y-series<br>  "chunk_size": [N],               # number of item to run for each chunk<br>  "cpu": [N]                       # number of running processors<br>}</pre>|
| user (optional) | [USER]                          |
| y               | [PATH_TO_INPUT_Y.CSV]           |
| x     | [PATH_TO_INPUT_x.CSV]           |

- **Request body example**

| Key       | Value          |
| :-------- | :------------- |
| module    | feat-selection |
| config    |<pre lang="json">{<br>  "model_name": "aicc01",<br>  "frequency": "m",<br>  "growth": true,<br>  "min_data_points": 72,<br>  "min_lag": 2,<br>  "max_lag": 23,<br>  "max_features": 10,<br>  "chunk_size": 2,<br>  "cpu": 1<br>}</pre>|
| user      | admin          |
| y         | input_y.csv    |
| x         | input_x.csv    |

## Response <a name="run-module-response"></a>

### Success Response
- **Code**: 200
- **Content**: 
```json
{
  "message": "Module is successfully run in background",
  "job_id": "[JOB_ID]",
  "job_status": "[URL_TO_GET_JOB_STATUS]"
}
```

### Error Response
- **Code**: 401 Unauthorized
- **Content**: 
```json
{
  "message": "Unauthorized"
}
```

OR

- **Code**: 400 Bad Request
- **Content**: 
```json
{
  "message": "[ERROR_TYPE]", 
  "error": "[ERROR_DETAIL]"
}
```

## File <a name="run-module-file"></a>

### Input file

- **y.csv**: Historical time series input data
```
id,ds,y
a,2014-01-01,100
a,2014-01-02,200
a,2014-01-03,300
a,2014-01-04,400
a,2014-01-05,500
b,2014-01-01,100
b,2014-01-02,200
b,2014-01-03,300
b,2014-01-04,400
b,2014-01-05,500
```
- **x.csv**: External features time series data
```
id,ds,x
x1,2014-01-01,100
x1,2014-01-02,200
x1,2014-01-03,300
x1,2014-01-04,400
x1,2014-01-05,500
x2,2014-01-01,100
x2,2014-01-02,200
x2,2014-01-03,300
x2,2014-01-04,400
x2,2014-01-05,500
```
- **lag.csv**: Length of lagging from external features
```
id_y,id_x,lag
a,x1,2
b,x2,4
```

- **forecastinglog.csv**: Forecasting log
```
id,ds,dsr,period,model,forecast,time
a,2019-01-01,2019-01-01,0,MODEL1,500,0.01
a,2019-02-01,2019-01-01,1,MODEL1,600,0.01
a,2019-03-01,2019-01-01,2,MODEL1,700,0.01
a,2019-01-01,2019-01-01,0,MODEL2,800,0.01
a,2019-02-01,2019-01-01,1,MODEL2,900,0.01
a,2019-03-01,2019-01-01,2,MODEL2,900,0.01
b,2019-01-01,2019-01-01,0,MODEL1,500,0.01
b,2019-02-01,2019-01-01,1,MODEL1,600,0.01
b,2019-03-01,2019-01-01,2,MODEL1,700,0.01
b,2019-01-01,2019-01-01,0,MODEL2,800,0.01
b,2019-02-01,2019-01-01,1,MODEL2,900,0.01
b,2019-03-01,2019-01-01,2,MODEL2,800,0.01
```

### Output file

- **output_forecast_log.csv**: Forecasting log (including last forecasting log)
```
id,ds,dsr,period,model,forecast,time
a,2019-01-01,2019-01-01,0,MODEL1,500,0.01
a,2019-02-01,2019-01-01,1,MODEL1,600,0.01
a,2019-03-01,2019-01-01,2,MODEL1,700,0.01
a,2019-01-01,2019-01-01,0,MODEL2,800,0.01
a,2019-02-01,2019-01-01,1,MODEL2,900,0.01
a,2019-03-01,2019-01-01,2,MODEL2,900,0.01
b,2019-01-01,2019-01-01,0,MODEL1,500,0.01
b,2019-02-01,2019-01-01,1,MODEL1,600,0.01
b,2019-03-01,2019-01-01,2,MODEL1,700,0.01
b,2019-01-01,2019-01-01,0,MODEL2,800,0.01
b,2019-02-01,2019-01-01,1,MODEL2,900,0.01
b,2019-03-01,2019-01-01,2,MODEL2,800,0.01
```
- **output_forecast.csv**: Forecasting output for top model
```
id,ds,dsr,period,forecast,error,top_model,test_back
a,2020-01-01,2020-01-01,0,500,0,[MODEL1, MODEL2],6
a,2020-02-01,2020-01-01,1,600,0,[MODEL1, MODEL2],6
a,2020-03-01,2020-01-01,2,700,0,[MODEL1, MODEL2],6
b,2020-01-01,2020-01-01,0,500,0,[MODEL1, MODEL2],6
b,2020-02-01,2020-01-01,1,600,0,[MODEL1, MODEL2],6
b,2020-03-01,2020-01-01,2,700,0,[MODEL1, MODEL2],6
```
- **output_evaluate.csv**: Combination of forecastlog, forecast and actual data
```
id,ds,dsr,period,model,forecast,actual,mae,mape,top_model,test_back,y,y_type
a,2015-01-01,2015-01-01,0,MODEL1,500,500,0,0,,,500,forecast
a,2015-02-01,2015-01-01,1,MODEL1,600,500,100,0.2,,,600,forecast
a,2015-03-01,2015-01-01,2,MODEL1,700,500,200,0.4,,,700,forecast
a,2015-01-01,2015-01-01,0,MODEL2,800,500,300,0.6,,,800,forecast
a,2015-02-01,2015-01-01,1,MODEL2,900,500,400,0.8,,,900,forecast
a,2015-03-01,2015-01-01,2,MODEL2,900,500,400,0.8,,,900,forecast
a,2015-01-01,2015-01-01,0,top3,650,500,150,0.3,[MODEL1, MODEL2],4,650,forecast
a,2015-02-01,2015-01-01,1,top3,750,500,250,0.5,[MODEL1, MODEL2],4,750,forecast
a,2015-03-01,2015-01-01,2,top3,800,500,300,0.6,[MODEL1, MODEL2],4,800,forecast
a,2015-01-01,2015-01-01,0,,,,,,,,500,actual
a,2015-02-01,2015-01-01,1,,,,,,,,500,actual
a,2015-03-01,2015-01-01,2,,,,,,,,500,actual
```
- **output_feature.csv**: Best features for each y-series
```
id_y,id_x,lag
a,x1,2
a,x2,4
b,x1,5
```

## Time series forecasting models <a name="time-series-forecasting-model"></a>

| Model           | Description                                          | Y Type  | External Features |
| :-------------- | :--------------------------------------------------- | :-----: | :---------------: |
| expo01          | Single Exponential Smoothing (Simple Smoothing)      | Nominal | No                |
| expo02          | Double Exponential Smoothing (Holt’s Method)         | Nominal | No                |
| expo03          | Triple Exponential Smoothing (Holt-Winters’ Method)  | Nominal | No                |
| naive01         | Naive model                                          | Nominal | No                |
| snaive01        | Seasonal Naive model                                 | Nominal | No                |
| sma01           | Simple Moving Average (short n)                      | Nominal | No                |
| sma02           | Simple Moving Average (middle n)                     | Nominal | No                |
| sma03           | Simple Moving Average (long n)                       | Nominal | No                |
| wma01           | Weighted Moving Average (short n)                    | Nominal | No                |
| wma02           | Weighted Moving Average (middle n)                   | Nominal | No                |
| wma03           | Weighted Moving Average (long n)                     | Nominal | No                |
| ema01           | Exponential Moving Average (short n)                 | Nominal | No                |
| ema02           | Exponential Moving Average (middle n)                | Nominal | No                |
| ema03           | Exponential Moving Average (long n)                  | Nominal | No                |
| arima01         | ARIMA model with fixed parameter                     | Nominal | No                |
| arima02         | ARIMA model with fixed parameter                     | Growth  | No                |
| arimax01        | ARIMAX model with fixed parameter                    | Nominal | Yes               |
| arimax02        | ARIMAX model with fixed parameter                    | Growth  | Yes               |
| autoarima01     | ARIMA model with optimal parameter                   | Nominal | No                |
| autoarima02     | ARIMA model with optimal parameter                   | Growth  | No                |
| autoarimax01    | ARIMAX model with optimal parameter                  | Nominal | Yes               |
| autoarimax02    | ARIMAX model with optimal parameter                  | Growth  | Yes               |
| prophet01       | Prophet by Facebook                                  | Nominal | No                |
| linear01        | Linear Regression                                    | Nominal | No                |
| linear02        | Linear Regression                                    | Growth  | No                |
| linearx01       | Linear Regression                                    | Nominal | Yes               |
| linearx02       | Linear Regression                                    | Growth  | Yes               |
| randomforest01  | Random Forest                                        | Nominal | No                |
| randomforest02  | Random Forest                                        | Growth  | No                |
| randomforestx01 | Random Forest                                        | Nominal | Yes               |
| randomforestx02 | Random Forest                                        | Growth  | Yes               |
| xgboost01       | XGBoost                                              | Nominal | No                |
| xgboost02       | XGBoost                                              | Growth  | No                |
| xgboostx01      | XGBoost                                              | Nominal | Yes               |
| xgboostx02      | XGBoost                                              | Growth  | Yes               |
| lstm01          | Long Short-Term Memory                               | Nominal | No                |
| lstm02          | Long Short-Term Memory                               | Growth  | No                |
| lstmr01         | Long Short-Term Memory with rolling forecast         | Nominal | No                |
| lstmr02         | Long Short-Term Memory with rolling forecast         | Growth  | No                |
| lstmx01         | Long Short-Term Memory                               | Nominal | Yes               |
| lstmx02         | Long Short-Term Memory                               | Growth  | Yes               |

## Feature Selection Model <a name="feature-selection-model"></a>

| Model   | Method         | Detail                                         |
| :------ | :------------- | :--------------------------------------------- |
| aicc01  | Optimize score | Forward selection and AICc score               |
| aic01   | Optimize score | Forward selection and AIC score                |
| r2adj01 | Optimize score | Forward selection and adjusted R-squared score |

# Stop running job <a name="stop-job"></a>

Stop running job by job id.

## Request <a name="stop-job-request"></a>

- **URL**: `/api/stop`

- **Method**: `POST`

### Request headers

| Key    | Value      |
| :----- | :--------- |
| apikey | [AUTH_KEY] |

### Request body
form-data

| Key    | Type   | Description          |
| :----- | :----- | :------------------- |
| job_id | string | job id to be stopped |

## Response <a name="stop-job-response"></a>

### Success Response
- **Code**: 200
- **Content**: 
```json
{
  "message": "Stop job",
  "job_id": "[JOB_ID]"
}
```

### Error Response
- **Code**: 401 Unauthorized
- **Content**: 
```json
{
  "message": "Unauthorized"
}
```

OR

- **Code**: 400 Bad Request
- **Content**: 
```json
{
  "message": "[ERROR_TYPE]", 
  "error": "[ERROR_DETAIL]"
}
```


# List all jobs <a name="list-job"></a>

Lists all jobs and their status (by user if the user parameter is provided).

## Request <a name="list-job-request"></a>

- **URL**: `/api/jobs`

- **Method**: `POST`

### Request headers

| Key    | Value      |
| :----- | :--------- |
| apikey | [AUTH_KEY] |

### Request body
form-data

| Key             | Type   | Description                        |
| :-------------- | :----- | :--------------------------------- |
| user (optional) | string | specify user to filter job results |

## Response <a name="list-job-response"></a>

### Success Response
- **Code**: 200
- **Content**: 
```json
{
  "JOB1_ID": "JOB1_STATUS",
  "JOB2_ID": "JOB2_STATUS"
}
```

### Error Response
- **Code**: 401 Unauthorized
- **Content**: 
```json
{
  "message": "Unauthorized"
}
```

OR

- **Code**: 400 Bad Request
- **Content**: 
```json
{
  "message": "[ERROR_TYPE]", 
  "error": "[ERROR_DETAIL]"
}
```

# Get job status <a name="job-status"></a>

## Request <a name="job-status-request"></a>

- **URL**: `/api/job/[JOB_ID]`

- **Method**: `GET`

## Response <a name="job-status-response"></a>

### Success Response
- **Code**: 200
- **Content**: 
```json
{
  "dt": "[DATETIME_START_JOB]",
  "files": ["FILE1", "FILE2", "FILE3"],
  "id": "[JOB_ID]",
  "log": "[SHORT_LOG]",
  "logs": "[FULL_LOG]",
  "module": "[RUNNING_MODULE]",
  "status": "[JOB_STATUS]",
  "user": "[USER]"
}
```

# Download output <a name="download-output"></a>

## Request <a name="download-output-request"></a>

- **URL**: 
    - `/api/job/[JOB_ID]/[FILE]` to get specific file 
    - `/api/job/[JOB_ID]/zip` to get all files in zip format

- **Method**: `GET`

## Response <a name="download-output-response"></a>

### Success Response
- **Code**: 200
- **Content**: file

# Remarks <a name="remarks"></a>

## Job status <a name="remarks-job-status"></a>
1. **start**: The module starts to run.
2. **running**: The module is running.
3. **finish**: The module has finised.
4. **stop**: The module was stopped due to the request by a user.
5. **error**: The module was terminated due to the errors occurred.
