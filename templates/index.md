# TIME SERIES FORECASTING
---

## REST API

### Request URL
[please copy this url](/api)

### Request headers
apikey: [AUTH_KEY]

### Request body

```json
{
	"run": "[RUN OPTION(validate/forecast)]",
	"path": "[PATH_TO_RUNNING_CONFIGURATION_FILE].yaml"
}
```


## MODELS
MODEL | DESCRIPTION | INPUT | OUTPUT | YTYPE
--- | --- | --- | --- | ---
expo01 | Single Exponential Smoothing (Simple Smoothing) | Daily / Monthly | Monthly | Nominal
expo02 | Double Exponential Smoothing (Holt’s Method) | Daily / Monthly | Monthly | Nominal
expo03 | Triple Exponential Smoothing (Holt-Winters’ Method) | Daily / Monthly | Monthly | Nominal
sma01 | Simple Moving Average (n=3) | Daily / Monthly | Monthly | Nominal
sma02 | Simple Moving Average (n=6) | Daily / Monthly | Monthly | Nominal
sma03 | Simple Moving Average (n=12) | Daily / Monthly | Monthly | Nominal
wma01 | Weighted Moving Average (n=3) | Daily / Monthly | Monthly | Nominal
wma02 | Weighted Moving Average (n=6) | Daily / Monthly | Monthly | Nominal
wma03 | Weighted Moving Average (n=12) | Daily / Monthly | Monthly | Nominal
ema01 | Exponential Moving Average (n=3) | Daily / Monthly | Monthly | Nominal
ema02 | Exponential Moving Average (n=6) | Daily / Monthly | Monthly | Nominal
ema03 | Exponential Moving Average (n=12) | Daily / Monthly | Monthly | Nominal
arima01 | ARIMA model with fixed parameter | Daily / Monthly | Monthly | Nominal
arima02 | ARIMA model with fixed parameter | Daily / Monthly | Monthly | Growth
arimax01 | ARIMAX model with fixed parameter and external features | Daily / Monthly | Monthly | Nominal
arimax02 | ARIMAX model with fixed parameter and external features | Daily / Monthly | Monthly | Growth
autoarima01 | ARIMA model with optimal parameter | Daily / Monthly | Monthly | Nominal
autoarima02 | ARIMA model with optimal parameter | Daily / Monthly | Monthly | Growth
autoarimax01 | ARIMAX model with optimal parameter and external features | Daily / Monthly | Monthly | Nominal
autoarimax02 | ARIMAX model with optimal parameter and external features | Daily / Monthly | Monthly | Growth
prophet01 | Prophet by Facebook | Daily | Monthly | Nominal
prophetd01 | Prophet by Facebook | Daily | Daily | Nominal
lineard01 | Linear Regression used latest trend to date | Daily | Daily | Nominal
lineard02 | Linear Regression used exact trend to date | Daily | Daily | Nominal
randomforest01 | Random Forest | Daily / Monthly | Monthly | Nominal
randomforest02 | Random Forest | Daily / Monthly | Monthly | Growth
randomforestx01 | Random Forest with external features | Daily / Monthly | Monthly | Nominal
randomforestx02 | Random Forest with external features | Daily / Monthly | Monthly | Growth
xgboost01 | XGBoost | Daily / Monthly | Monthly | Nominal
xgboost02 | XGBoost | Daily / Monthly | Monthly | Growth
xgboostx01 | XGBoost with external features | Daily / Monthly | Monthly | Nominal
xgboostx02 | XGBoost with external features | Daily / Monthly | Monthly | Growth
lstm01 | Long Short-Term Memory | Daily / Monthly | Monthly | Nominal
lstm02 | Long Short-Term Memory | Daily / Monthly | Monthly | Growth
lstmr01 | Long Short-Term Memory with rolling forecast | Daily / Monthly | Monthly | Nominal
lstmr02 | Long Short-Term Memory with rolling forecast | Daily / Monthly | Monthly | Growth
lstmx01 | Long Short-Term Memory with external | Daily / Monthly | Monthly | Nominal
lstmx02 | Long Short-Term Memory with external | Daily / Monthly | Monthly | Growth


## CONFIGURE RUN
- configuration: validate.yaml
```yaml
# actual sales data (daily/monthly)
ACT_PATH: [PATH_TO_FILE].csv
# external data (monthly)
EXT_PATH: [PATH_TO_FILE].csv
# external lagging of each item
EXTLAG_PATH: [PATH_TO_FILE].csv
# output directory
OUTPUT_DIR: [PATH_TO_DIRECTORY]
# actual sales start date
ACT_START: [YYYY-MM-DD]
# test start date
TEST_START: [YYYY-MM-DD]
# number of rolling period to test (months)
TEST_PERIOD: [N]
# list of model to test
TEST_MODEL: [[MODEL1], [MODEL2], [MODEL3], [MODELN]]
# number of periods to forecast for each rolling
FCST_PERIOD: [N]
# forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
FCST_FREQ: [d/m/q/y]
# starting period for each forecast (default 0/1)
PERIOD_START: [N]
# number of item to validate for each chunk
CHUNKSIZE: [N]
# number of running processors
CPU: [N]
```

- configuration: forecast.yaml
```yaml
# actual sales data (daily/monthly)
ACT_PATH: [PATH_TO_FILE].csv
# forecasting log for walk-forward validation
FCSTLOG_PATH: [PATH_TO_FILE].csv
# external data (monthly)
EXT_PATH: [PATH_TO_FILE].csv
# external lagging of each item
EXTLAG_PATH: [PATH_TO_FILE].csv
# output directory
OUTPUT_DIR: [PATH_TO_DIRECTORY]
# actual sales start date
ACT_START: [YYYY-MM-DD]
# forecast date
FCST_START: [YYYY-MM-DD]
# forecast model options for each periods
FCST_MODEL:
  0: [MODEL1, MODEL2, MODEL3, MODELN]
  1: [MODEL1, MODEL2, MODEL3, MODELN]
  2: [MODEL1, MODEL2, MODEL3, MODELN]
  3: [MODEL1, MODEL2, MODEL3, MODELN]
  n: [MODEL1, MODEL2, MODEL3, MODELN]
# forecast frequency (d-daily, m-monthly, q-quarterly, y-yearly)
FCST_FREQ: [d/m/q/y]
# number of months to test back
TEST_BACK: [N]
# top N best models to use
TOP_MODEL: [N]
# ensemble method to combine the models
ENSEMBLE_METHOD: [mean/median]
# number of item to validate for each chunk
CHUNKSIZE: [N]
# number of running processors
CPU: [N]
```

### How to specify path?

- file
    - local path: [PATH_TO_FILE]/[FILE_NAME]
    - gcp path: gs://[BUCKET_NAME]/[PATH_TO_FILE]/[FILE_NAME]
  
- directory
    - local path: [PATH_TO_DIR]/
    - gcp path: gs://[BUCKET_NAME]/[PATH_TO_DIR]/


## EXAMPLE FILES
- input: input_actual.csv
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
- input: input_external.csv
```
id,ds,y
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
- input: input_external_lag.csv
```
y_id,ext_id,lag
a,x1,2
b,x2,4
```
- input: input_forecast.csv
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
- output: output_validate_0001-0100.csv
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
- output: output_forecast_0001-0100.csv
```
id,ds,dsr,period,forecast,error
a,2020-01-01,2020-01-01,0,500,0
a,2020-02-01,2020-01-01,1,600,0
a,2020-03-01,2020-01-01,2,700,0
b,2020-01-01,2020-01-01,0,500,0
b,2020-02-01,2020-01-01,1,600,0
b,2020-03-01,2020-01-01,2,700,0
```
- output: output_forecastlog_0001-0100.csv
```
id,ds,dsr,period,model,forecast,time
a,2020-01-01,2020-01-01,0,MODEL1,500,0.01
a,2020-02-01,2020-01-01,1,MODEL1,600,0.01
a,2020-03-01,2020-01-01,2,MODEL1,700,0.01
a,2020-01-01,2020-01-01,0,MODEL2,800,0.01
a,2020-02-01,2020-01-01,1,MODEL2,900,0.01
a,2020-03-01,2020-01-01,2,MODEL2,900,0.01
b,2020-01-01,2020-01-01,0,MODEL1,500,0.01
b,2020-02-01,2020-01-01,1,MODEL1,600,0.01
b,2020-03-01,2020-01-01,2,MODEL1,700,0.01
b,2020-01-01,2020-01-01,0,MODEL2,800,0.01
b,2020-02-01,2020-01-01,1,MODEL2,900,0.01
b,2020-03-01,2020-01-01,2,MODEL2,800,0.01
```
