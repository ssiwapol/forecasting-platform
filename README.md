# FORECASTING PLATFORM

# Modules
- **Initial forecast**: Rolling forecast from start date to end date based on selected time series forecasting model.
- **Production forecast**: Next period forecast from top model selection based on last forecasting log.
- **Feature selection**: Select x-series (external features) and their lag periods which are the most correlated for forecasting each y-series.

## Time Series Forecasting Model
- Single Exponential Smoothing (Simple Smoothing)
- Double Exponential Smoothing (Holt’s Method)
- Triple Exponential Smoothing (Holt-Winters’ Method)
- Naive (Normal and Seasonal)
- Simple Moving Average
- Weighted Moving Average
- Exponential Moving Average
- ARIMA (Autoregressive Integrated Moving Average)
- ARIMAX (Autoregressive Integrated Moving Average with Explanatory Variable)
- [Prophet by Facebook](https://facebook.github.io/prophet/)
- Linear Regression
- Random Forest
- XGBoost
- LSTM (Long Short-Term Memory)

## Feature Selection Model
- Optimze score by AICc, AIC, Adjusted R-squared


# REST API

## Features
- Request API to run job module in the background
- Monitor job
- Download output file
- Control job access by user
- Limit maximum job in the platform

## API
- Run Module (`POST`)
- Stop running job (`POST`)
- List all jobs (`POST`)
- Get job status (`GET`)
- Download output (`GET`)


# Deployment

## Run without configuration
1. Build Container
```
git clone [REPO_URL]
cd forecasting-platform
docker build -t [IMAGE_NAME] .
```
2. Run Container
```
docker run --name [CONTAINER_NAME] -d -p [PORT]:5000 [IMAGE_NAME]
```

## Run by changing configuration file
1. Build Container
```
git clone [REPO_URL]
cd forecasting-platform
docker build -t [IMAGE_NAME] .
```

2. Prepare config.yaml (other directory)

config.yaml
```yaml
# API key to authenticate while requesting
apikey: [API_KEY]

# Timezone in platform (logfile, database)
timezone: [TIMEZONE]

# Database name (can use from last running container)
database: [JOB_DATABASE_NAME.db]

# Maximum job in the platform to reduce memory of machine
max_job: [N]

# Show debug in webpage or not
debug: [True/False]

# Stream log in container log or not
log_stream: [True/False]

# Number of lines to display in short log in job status
log_lines: [N]
```
- example of config.yaml file can be found [here](/config.yaml)
- list of all timezone can be found [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)


3. Run Container
```
# run without mounting job data
docker run --name [CONTAINER_NAME] \
-v $(pwd)/config.yaml:/app/config.yaml \
-d -p [PORT]:5000 [IMAGE_NAME]

# run by mounting last job database and data
docker run --name [CONTAINER_NAME] \
-v $(pwd)/config.yaml:/app/config.yaml \
-v $(pwd)/tmp:/app/tmp \
-d -p [PORT]:5000 [IMAGE_NAME]
```
