# FORECASTING PLATFORM
---

## REST API

### Request URL
[please copy this url](/api)

### Request headers
apikey: [AUTH_KEY]

### Request body

```json
{
	"run": "[RUN OPTION(validate/forecast/featselection)]",
	"path": "[PATH_TO_RUNNING_CONFIGURATION_FILE].yaml"
}
```


## TIME-SERIES-FORECASTING MODELS
MODEL | DESCRIPTION | YTYPE | EXTERNAL FEATURES
----- | ----------- | ----- | -----------------
expo01 | Single Exponential Smoothing (Simple Smoothing) | Nominal | No
expo02 | Double Exponential Smoothing (Holt’s Method) | Nominal | No
expo03 | Triple Exponential Smoothing (Holt-Winters’ Method) | Nominal | No
naive01 | Naive model | Nominal | No
snaive01 | Seasonal Naive model | Nominal | No
sma01 | Simple Moving Average (short n) | Nominal | No
sma02 | Simple Moving Average (middle n) | Nominal | No
sma03 | Simple Moving Average (long n) | Nominal | No
wma01 | Weighted Moving Average (short n) | Nominal | No
wma02 | Weighted Moving Average (middle n) | Nominal | No
wma03 | Weighted Moving Average (long n) | Nominal | No
ema01 | Exponential Moving Average (short n) | Nominal | No
ema02 | Exponential Moving Average (middle n) | Nominal | No
ema03 | Exponential Moving Average (long n) | Nominal | No
arima01 | ARIMA model with fixed parameter | Nominal | No
arima02 | ARIMA model with fixed parameter | Growth | No
arimax01 | ARIMAX model with fixed parameter | Nominal | Yes
arimax02 | ARIMAX model with fixed parameter | Growth | Yes
autoarima01 | ARIMA model with optimal parameter | Nominal | No
autoarima02 | ARIMA model with optimal parameter | Growth | No
autoarimax01 | ARIMAX model with optimal parameter | Nominal | Yes
autoarimax02 | ARIMAX model with optimal parameter | Growth | Yes
prophet01 | Prophet by Facebook | Nominal | No
linear01 | Linear Regression | Nominal | No
linear02 | Linear Regression | Growth | No
linearx01 | Linear Regression | Nominal | Yes
linearx02 | Linear Regression | Growth | Yes
randomforest01 | Random Forest | Nominal | No
randomforest02 | Random Forest | Growth | No
randomforestx01 | Random Forest | Nominal | Yes
randomforestx02 | Random Forest | Growth | Yes
xgboost01 | XGBoost | Nominal | No
xgboost02 | XGBoost | Growth | No
xgboostx01 | XGBoost | Nominal | Yes
xgboostx02 | XGBoost | Growth | Yes
lstm01 | Long Short-Term Memory | Nominal | No
lstm02 | Long Short-Term Memory | Growth | No
lstmr01 | Long Short-Term Memory with rolling forecast | Nominal | No
lstmr02 | Long Short-Term Memory with rolling forecast | Growth | No
lstmx01 | Long Short-Term Memory | Nominal | Yes
lstmx02 | Long Short-Term Memory | Growth | Yes

## FEATURE SELECTION MODELS
MODEL | METHOD | DETAIL
----- | ------ | ------
aicc01 | Optimize score | Forward selection and AICc score
aic01 | Optimize score | Forward selection and AIC score
r2adj01 | Optimize score | Forward selection and adjusted R-squared score


## CONFIGURE RUN
- configuration: validate.yaml
```yaml
# actual series data
ACT_PATH: [PATH_TO_FILE].csv

# external series data
EXT_PATH: [PATH_TO_FILE].csv

# external lag of each item
EXTLAG_PATH: [PATH_TO_FILE].csv

# output directory
OUTPUT_DIR: [PATH_TO_DIRECTORY]

# actual series start date
ACT_START: [YYYY-MM-DD]

# test start date
TEST_START: [YYYY-MM-DD]

# number of rolling period to test
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
# actual series data (daily/monthly)
ACT_PATH: [PATH_TO_FILE].csv

# forecasting log for walk-forward validation
FCSTLOG_PATH: [PATH_TO_FILE].csv

# external series data
EXT_PATH: [PATH_TO_FILE].csv

# external lag of each item
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

# number of periods to test back
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

- configuration: featselection.yaml
```yaml
# x-series to be selected
X_PATH: [PATH_TO_FILE].csv

# y-series
Y_PATH: [PATH_TO_FILE].csv

# output directory
OUTPUT_DIR: [PATH_TO_DIRECTORY]

# selected to model to run feature selection
MODEL: [MODEL]

# data frequency (d-daily, m-monthly, q-quarterly, y-yearly)
FREQ: [d/m/q/y]

# run model by growth or not
GROWTH: [True/False]

# minimum data points for x-series (default 72)
MIN_DATA_POINTS: [N]

# minimum lag available for x-series (default 2)
MIN_LAG: [N]

# maximum lag available for x-series (default 23)
MAX_LAG: [N]

# maximum total features for each y-series (default 10)
MAX_FEATURES: [N]

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
### Input
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
- input: input_external_lag.csv
```
id_y,id_x,lag
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
- input: input_x.csv
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
- input: input_y.csv
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

### Output
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
id,ds,dsr,period,forecast,error,top_model,test_back
a,2020-01-01,2020-01-01,0,500,0,[MODEL1, MODEL2],6
a,2020-02-01,2020-01-01,1,600,0,[MODEL1, MODEL2],6
a,2020-03-01,2020-01-01,2,700,0,[MODEL1, MODEL2],6
b,2020-01-01,2020-01-01,0,500,0,[MODEL1, MODEL2],6
b,2020-02-01,2020-01-01,1,600,0,[MODEL1, MODEL2],6
b,2020-03-01,2020-01-01,2,700,0,[MODEL1, MODEL2],6
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
- output: output_feature_0001-0010.csv
```
id_y,id_x,lag
a,x1,2
b,x2,4
```
