"""
    Weather interface
"""

import requests, json, urllib, datetime
import pytz


def query_darksky(session, time, lat, lng, params):
    with open('config.json') as json_data_file:
        CONFIG = json.load(json_data_file)
    token = CONFIG['DARKSKY']['token']
    search_str = "%s,%s,%s" % (lat, lng, int(time))
    url = urllib.parse.urljoin(CONFIG['DARKSKY']['endpoint'], token)
    url = "%s/%s" % (url, search_str)
    return session.get(url, params=params)


def get_forecast(time, lat, lng):
    session = requests.session()
    exclude = ["currently", "minutely", "daily", "alerts", "flags"]
    payload = {'exclude': exclude}
    response = query_darksky(session, time, lat, lng, params=payload)
    if response.ok:
        return response.text
    else:
        raise Exception('Error while receiving data - Message: %s' % response.text)


def parse_forecast(raw_data):
    with open('config.json') as config_file:
        CONFIG = json.load(config_file)
    json_file = json.loads(raw_data)
    time = []
    temperature = []
    icon = []
    for hour in json_file["hourly"]["data"]:
        time.append(hour["time"])
        temperature.append(hour["temperature"])
        ic = hour["icon"]
        ic = CONFIG["DARKSKY"]["iconmap"].get(ic)
        icon.append(ic)
    forecast = {"Time": time, "Temperature": temperature, "Icon": icon}

    return forecast


def fetch_forecast(time, lat, lng):
    raw_data = get_forecast(time, lat, lng)
    return parse_forecast(raw_data)


def fetch_stations_forecasts(time):
    with open('config.json') as config_file:
        CONFIG = json.load(config_file)
    forecasts = {}
    timestamp = (time - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)) / datetime.timedelta(seconds=1)
    
    for s in CONFIG['WEATHER_STATIONS']:
        lat = CONFIG['WEATHER_STATIONS'][s]['lat']
        lng = CONFIG['WEATHER_STATIONS'][s]['lng']
        forecast = fetch_forecast(timestamp, lat, lng)
        
        forecasts[s] = forecast
        
    return forecasts
