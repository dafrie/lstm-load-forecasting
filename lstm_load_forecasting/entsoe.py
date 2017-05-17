"""
    This script can be used to pull data from the ENTSOE-API
    https://transparency.entsoe.eu/
"""
import json, datetime, requests, re
from bs4 import BeautifulSoup


def query_entsoe(session, params):
    # Load config file
    with open('config.json') as json_data_file:
        data = json.load(json_data_file)
    CONFIG = data
    params['securityToken'] = CONFIG['ENTSOE']['token']
    return session.get(CONFIG['ENTSOE']['endpoint'], params=params)


def get_load_data(params):
    session = requests.session()
    response = query_entsoe(session, params)
    if response.ok:
        return response.text
    else:
        raise Exception('Error while receiving data - Message: %s' % response.text )


def parse_load_data(raw_data):
    soup = BeautifulSoup(raw_data, 'html.parser')

    position = []
    quantities = []
    time = []

    for series in soup.find_all('timeseries'):
        resolution = series.find_all('resolution')[0].contents
        start = datetime.datetime.strptime(series.find_all('start')[0].contents[0], '%Y-%m-%dT%H:%MZ')
        end = datetime.datetime.strptime(series.find_all('end')[0].contents[0], '%Y-%m-%dT%H:%MZ')
        for item in series.find_all('point'):
            total_position = int(item.find_all('position')[0].contents[0])
            position.append(int(item.find_all('position')[0].contents[0]))
            quantities.append(float(item.find_all('quantity')[0].contents[0]))

            # TODO: No hardcoding 
            t = (start + (total_position-1) * datetime.timedelta(minutes=60))
            time.append(t.strftime('%Y%m%d%H%M'))

    return [quantities, time]


def fetch_load_data(starting, ending):
    params = {
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_domain': '10YCH-SWISSGRIDZ',
        'periodStart': starting,
        'periodEnd': ending
    }

    raw_data = get_load_data(params)
    data = parse_load_data(raw_data)

    return data


def fetch_load_forecast_data(starting, ending):
    # Forecast data always returns the whole day, take care!
    params = {
        'documentType': 'A65',
        'processType': 'A01',
        'outBiddingZone_domain': '10YCH-SWISSGRIDZ',
        'periodStart': starting,
        'periodEnd': ending
    }

    raw_data = get_load_data(params)
    data = parse_load_data(raw_data)
    return data

