import datetime
import meteostat
import pandas as pd
from sodapy import Socrata

import config
import helper

def fetch_weather(longitude, latitude, start_date, end_date=None, radius_km = 5.5, max_stations = 5, max_retries=3, level='daily', timezone='US/Central'):
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is None:
        end_date = start_date
    start_time = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_time = datetime.datetime.combine(end_date, datetime.datetime.min.time())
        
    ## initialize the location
    stations = meteostat.Stations().nearby(latitude, longitude, radius = radius_km*1000).fetch(max_stations)
    data_found = False

    # traverse each weather station
    for station_id in stations.index:
        for attempt in range(max_retries + 1):
            try:
                if level == 'hourly':
                    # fetch hourly data
                    data_hourly = meteostat.Hourly(station_id, start_time, end_time, timezone=timezone)
                    data = data_hourly.fetch()
                elif level == 'daily':
                    data_daily = meteostat.Daily(station_id, start_time, end_time)
                    data = data_daily.fetch()

                # return if fetch data successfully
                if not data.empty:
                    data['lon_bin'] = longitude
                    data['lat_bin'] = latitude
                    data_found = True
                    break
            except Exception as e:
                print(f"Error fetching data: {e}")
                if attempt == max_retries:
                    print("Max retries reached, moving to next station.")
        if data_found:
            return data
    
    return {}


def fetch_crime_record(city_name, date, offline=True, lat_bin=None, lon_bin=None):
    
    if city_name.lower() == 'dallas':
        if offline:
            return helper.offline_record_loader(city_name, date, lon_bin, lat_bin)
        
        else:
            client = Socrata(config.DPD_API_URL, None)
            print(f"Successfully connected to API at {config.DPD_API_URL}")
            
            # Get previous day in Dallas time
            date_filter = f"""date1 between '{(datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=14)).strftime('%Y-%m-%d')} 00:00:00.0000000' and '{date} 00:00:00.0000000'"""
            
            results = client.get(config.DPD_DATASET_ID, where=date_filter, limit = 20000)
            results_df = pd.DataFrame.from_records(results)
            
            print("Data columns:", results_df.columns.tolist())
            print("Sample data:", results_df.head(1).to_dict())
        
            if not results_df:
                print("Warning: No data found for the date")
                return pd.DataFrame({
                    'geocoded_column': [],
                    'date1': [],
                    'incidentnum': [],
                    'Location1': []
                })
        
    return results_df