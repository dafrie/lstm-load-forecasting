# Electricity load forecasting with LSTM
Demo project for electricity load forecasting with a LSTM (abbr. "Long Term Short Term Memory", a Recurrent Neural Network) with data for Switzerland.

## Getting started

It is recommended to use a Python and R packages and environment management tool like [Anaconda](https://www.continuum.io/downloads). 
This will install also already most of the packages used in this project and will allow to run the notebooks in ./notebooks/ with the Jupyter Notebook server.

### Main packages used

```
* [Keras 2.0.2](https://github.com/fchollet/keras) - High-level Neural Network API used for LSTM modelling. Backend used was TensorFlow.
* [forecast 8.0](https://cran.r-project.org/web/packages/forecast/index.html) - Forecasting package for R used for TBATS and ARIMA benchmark forecasts. 
```

### Data: 

-   Hourly series for actual load and forecasted load. Source: [ENTSO-E API](https://transparency.entsoe.eu/)
-   Hourly series for temperature (in °F) and a mapping of the qualitative weather in one of the 3 categories defined in the configuration file. These historical observations are collected for the major cities in Switzerland. Source: [Dark Sky API](https://darksky.net/)
-   Calendar dummy variables. Holidays, month, day of week, hour of day.
-   In total: 20756x78 observations

### File structure:

    .
    ├── data                        # Load, calendar and weather data
    ├── lstm_load_forecasting       # Helper functions for data preparation and LSTM model building
    ├── models                      # All trained models saved in HDF5 file format
    ├── notebooks                   # LSTM Model selection and forecast comparison
    ├── results                     # Results and parameters from model training run comparison
    ├── config.json                 # Config file with API Keys, Weather stations, holiday calendar.  
    ├── LICENSE
    └── README.md

### Notebooks

Different "categories" (named data modules in the data helper functions...) of models have been defined and then for each category, a number of models based on possible parameter combinations have been estimated:

* [Model 1](notebooks/1_entsoe_forecast_only.ipynb): Using only the ENTSO-E forecast as input.
* [Model 2](notebooks/2_calendar_only.ipynb): Using only the calendar dummy variables as input.
* [Model 3](notebooks/3_weather_only.ipynb): Using only the weather data as input.
* [Model 4](notebooks/4_entsoe_calendar.ipynb): Using ENTSO-E and calendar data as input.
* [Model 5](notebooks/5_calendar_weather.ipynb): Using weather data and calendar dummies as input.
* [Model 6](notebooks/6_all_modules.ipynb): Using ENTSO-E, calendar and weather as input. 

The [LSTM model results notebook](notebooks/lstm_model_results.ipynb) then combines all results.
The best models models in category 4, 5 6 then are compared in the [Forecast Comparison](notebooks/forecast_comparison.ipynb) notebook with a TBATS and a ARIMA model forecasts that have been generated in the respective R notebooks.
The comparison then is further extended with rolling window (static, using at each time step the newly available data) in the [Rolling Forecast](notebooks/rolling_forecast.ipynb) notebook.


## Acknowledgments and References
* [Marino et al. (2016)](https://arxiv.org/abs/1610.09460)
* http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
