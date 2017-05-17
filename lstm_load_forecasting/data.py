import datetime
import numpy as np
import pandas as pd
from dateutil.tz import tzutc
from datetime import date
import pytz
from lstm_load_forecasting import entsoe, weather
import json

def load_dataset(path=None, update_date=None, modules=None):
    indicator_vars = {'bsl_1':10,'bsl_2':11,'bsl_3':12,'brn_1':13,'brn_2':14,'brn_3':15,'zrh_1':16,'zrh_2':17,
                      'zrh_3':18,'lug_1':19,'lug_2':20,'lug_3':21,'lau_1':22,'lau_2':23,'lau_3':24,'gen_1':25,
                      'gen_2':26,'gen_3':27,'stg_1':28,'stg_2':29,'stg_3':30,'luz_1':31,'luz_2':32,'luz_3':33,
                      'holiday':34,'weekday_0':35,'weekday_1':36,'weekday_2':37,'weekday_3':38,'weekday_4':39,'weekday_5':40,
                      'weekday_6':41,'hour_0': 42, 'hour_1':42, 'hour_2':44, 'hour_3':45, 'hour_4':46, 'hour_5':47,
                      'hour_6':48, 'hour_7':49, 'hour_8':50,'hour_9':51, 'hour_10':52, 'hour_11':53, 'hour_12':54,
                      'hour_13':55, 'hour_14':56, 'hour_15':57, 'hour_16':58,'hour_17':59, 'hour_18':60, 'hour_19':61,
                      'hour_20':62, 'hour_21':63, 'hour_22':64, 'hour_23':65,'month_1':66, 'month_2':67, 'month_3':68,
                      'month_4':69, 'month_5':70, 'month_6':71, 'month_7':72, 'month_8':73,'month_9':74, 'month_10':75,
                      'month_11':76, 'month_12':77}
    df = pd.read_csv(path, delimiter=';', parse_dates=[0], index_col = 0)
    df[list(indicator_vars.keys())] = df[list(indicator_vars.keys())].astype('int')
    df = df.tz_localize('utc')
    df = df.sort_index()
    
    if update_date:
        last_actual_obs = df['actual'].last_valid_index()
        last_obs = df.index[-1]
        local_timezone = pytz.timezone('Europe/Zurich')
        update_date = local_timezone.localize(update_date)
        print('============================================')
        if update_date - pd.DateOffset(hours=1) > last_actual_obs:
            df, df_n = update_dataset(df, update_date)
            df.to_csv('data/fulldataset.csv', sep=';')
            print('Updated: {}'.format(df_n.shape))
            print('New size: {}'.format(df.shape))
        else:
            print('Nothing to update')
    
    columns = []
    if 'actual' in modules or 'all' in modules:
        columns.append('actual')
    if 'entsoe' in modules or 'all' in modules:
        columns.append('entsoe')
    if 'weather' in modules or 'all' in modules:
        columns.extend(['bsl_t','brn_t','zrh_t','lug_t','lau_t','gen_t','stg_t','luz_t',
                        'bsl_1', 'bsl_2','bsl_3','brn_1','brn_2','brn_3','zrh_1','zrh_2','zrh_3','lug_1','lug_2','lug_3',
                        'lau_1','lau_2','lau_3','gen_1','gen_2','gen_3','stg_1','stg_2','stg_3','luz_1','luz_2','luz_3'])
    if 'calendar' in modules or 'all' in modules:
        columns.extend(['holiday', 'weekday_0','weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6',
                        'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8',
                        'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16',
                        'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                        'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
                        'month_9', 'month_10', 'month_11', 'month_12'])
    df = df[columns]
    return df


def update_dataset(df=None, to_date=None):
    
    # Set up
    df = df.sort_index()
    last_obs = df.index[-1]
    last_actual_obs = df['actual'].last_valid_index()
    
    columns = df.columns  
    starting = last_actual_obs + pd.DateOffset(hours=1)
    ending = to_date
    starting = starting.replace(minute=0, second=0)
    ending = ending.replace(minute=0, second=0)
    fmt = '%Y%m%d%H%M'
    starting = starting.tz_convert('utc')
    ending = ending.astimezone(pytz.utc) - pd.DateOffset(hours=1)
    df_n = pd.DataFrame(index=pd.date_range(starting, ending, freq='60min'))
    
    
    # Get the actual values
    now = datetime.datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    now = now.astimezone(pytz.utc)
    if starting < now:
        actual = entsoe.fetch_load_data(starting.strftime(fmt), (ending+pd.DateOffset(hours=1)).strftime(fmt))
        df_a = pd.DataFrame(actual)
        df_a = df_a.transpose()
        df_a.columns = ['actual', 'time']
        df_a['time'] = pd.to_datetime(df_a['time'], format=fmt)
        df_a.index = df_a['time']
        df_a.index = df_a.index.tz_localize('utc')
        df_a = df_a.drop('time', 1)
        df_n = df_n.combine_first(df_a)
    else:
        print('Skipped actual values')
    
    # Get the forecast values
    forecast = entsoe.fetch_load_forecast_data(starting.strftime(fmt), (ending+pd.DateOffset(hours=1)).strftime(fmt))
    df_f = pd.DataFrame(forecast)
    df_f = df_f.transpose()
    df_f.columns = ['entsoe', 'time']
    df_f['time'] = pd.to_datetime(df_f['time'], format=fmt)
    df_f.index = df_f['time']
    df_f.index = df_f.index.tz_localize('utc')
    df_f = df_f.drop('time', 1) 
    df_n = df_n.combine_first(df_f)
    
    # Get the weather
    df_n = update_weather(df=df_n, starting=starting, ending=ending)
    
    # Holiday
    with open('config.json') as config_file:
        CONFIG = json.load(config_file)
    holidays = CONFIG['HOLIDAYS']
    df_h = pd.DataFrame(index=holidays)
    df_h = pd.to_datetime(df_h.index, format='%Y-%m-%d')
    df_n['holiday'] = df_n.index.date
    df_n['holiday'] = df_n['holiday'].isin(df_h.date).values.astype('int')
  
    # Weekday
    weekday = pd.get_dummies(data=pd.Series(df_n.index.weekday).astype('category', categories=[d for d in range(7)]),
                             prefix='weekday')
    for d in weekday.columns.values:
        df_n[d] = weekday[d].values.astype('int')
    
    # Hour
    hour = pd.get_dummies(data=pd.Series(df_n.index.hour).astype('category', categories=[h for h in range(24)]),
                          prefix='hour')
    for h in hour.columns.values:
        df_n[h] = hour[h].values.astype('int')

    # Month
    month = pd.get_dummies(data=pd.Series(df_n.index.month).astype('category', categories=[m+1 for m in range(12)]),
                           prefix='month')
    for m in month.columns.values:
        df_n[m] = month[m].values.astype('int')
    
    df = df.combine_first(df_n)
    df = df[columns]
    
    return df, df_n


def update_weather(df, starting, ending):
    # Get the weather
    df_w = pd.DataFrame(index=pd.date_range(starting, ending, freq='60min'))
    for idx, row in enumerate(df.iloc[::24, :].iterrows()):
        forecasts = weather.fetch_stations_forecasts(row[0])
        for stat, series in forecasts.items(): 
            time = []
            for t in series['Time']:
                time.append(datetime.datetime.fromtimestamp(t, tz=pytz.UTC))
            temperature = series['Temperature']
            icon = series['Icon']
            data = {
                stat.lower() + '_t': temperature,
                stat.lower() + '_1': np.equal(icon, 1).astype(int),
                stat.lower() + '_2': np.equal(icon, 2).astype(int),
                stat.lower() + '_3': np.equal(icon, 3).astype(int),
                }
            df_s = pd.DataFrame(index=time, data=data)
            # When first iteration, then concat, otherwise join
            if idx == 0:
                df_w = pd.concat([df_w, df_s], axis=1)
            elif idx > 0:
                df_w = df_w.combine_first(df_s)
    df = df.combine_first(df_w)
    return df