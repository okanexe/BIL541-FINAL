import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.ensemble import RandomForestRegressor # for building the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rent_predict_hourly_count_for_station(station_name):
    df = pd.read_csv("../data/BikeshareQ3.csv")
    # choose which station to work
    station = df[df["from_station_name"] == station_name]

    # Transfer format.
    station['trip_start_time'] = pd.to_datetime(station['trip_start_time'])
    station['trip_stop_time'] = pd.to_datetime(station['trip_stop_time'])

    # Split the month, day, hour and minute to separate columns
    station['start_hour'] = station['trip_start_time'].apply(lambda x: x.hour)
    station['start_minute'] = station['trip_start_time'].apply(lambda x: x.minute)
    station['start_month'] = station['trip_start_time'].apply(lambda x: x.month)
    station['start_day'] = station['trip_start_time'].apply(lambda x: x.day)

    station['stop_hour'] = station['trip_stop_time'].apply(lambda x: x.hour)
    station['stop_minute'] = station['trip_stop_time'].apply(lambda x: x.minute)
    station['stop_month'] = station['trip_stop_time'].apply(lambda x: x.month)
    station['stop_day'] = station['trip_stop_time'].apply(lambda x: x.day)

    weather = pd.read_csv("../data/Toronto_temp.csv")
    weather["Date/Time"] = weather["Date/Time"].str.replace(',',  '')
    weather['Date/Time'] = pd.to_datetime(weather['Date/Time'])
    weather.rename(columns={'Date/Time': 'Date'}, inplace=True)
    weather['Date'] = pd.to_datetime(weather['Date']).dt.date

    station['trip_start_time'] = pd.to_datetime(station['trip_start_time']).dt.date

    merged = station.merge(weather, left_on='trip_start_time', right_on='Date')
    train = merged.drop(['trip_start_time', 'trip_stop_time', 'from_station_name', 'user_type',
                         'Year', 'Month', 'Day', 'Date'], axis=1)

    rent_bike = train.drop(['trip_id', 'to_station_name', 'start_minute', 'stop_hour',
                            'stop_minute', 'stop_month', 'stop_day'], axis=1)
    rent_bike = rent_bike[rent_bike.start_month == 8]
    rent_bike['hourly_count'] = rent_bike.groupby('start_hour')['start_hour'].transform('count')
    rent_bike['duration_avg'] = rent_bike.groupby(['start_hour', 'start_month',
                                                   'start_day'])['trip_duration_seconds'].transform('mean')
    rent_bike = rent_bike.drop_duplicates(subset=['start_hour', 'start_month', 'start_day', 'duration_avg'], keep='last')
    rent_bike = rent_bike.drop(['trip_duration_seconds', 'season'], axis=1)

    x = rent_bike.drop('hourly_count', axis=1)  # Features
    y = rent_bike['hourly_count']  # Target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

    # Initializing the Random Forest Regression model with 10 decision trees
    model = RandomForestRegressor(n_estimators=10, random_state=0)

    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    # Predicting the target values of the test set
    y_pred = model.predict(x_test)

    # RMSE (Root Mean Square Error) and r2 score
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE: ", rmse)
    print('R2:', r2_score(y_test, y_pred))

    station_month7 = train.drop(['trip_id', 'to_station_name', 'start_minute', 'stop_hour',
                      'stop_minute', 'stop_month', 'stop_day', 'season'], axis=1)
    station_month7 = station_month7[station_month7.start_month == 7]
    station_month7['hourly_count'] = station_month7.groupby('start_hour')['start_hour'].transform('count')
    station_month7['duration_avg'] = station_month7.groupby(
        ['start_hour', 'start_month', 'start_day'])['trip_duration_seconds'].transform('mean')
    station_month7 = station_month7.drop(['trip_duration_seconds'], axis=1)
    station_month7 = station_month7.drop_duplicates(subset=['start_hour', 'start_month', 'start_day', 'duration_avg'],
                                                    keep='last')
    df7_train = station_month7.drop('hourly_count', axis=1)
    df7_target = station_month7['hourly_count']

    df7_pred = model.predict(df7_train)
    print('R2:', r2_score(df7_target, df7_pred))

    data = df7_train.copy()
    data['hourly_count'] = df7_pred
    data = data.groupby(['start_hour']).mean()
    data = data[['hourly_count']]
    # data.to_csv('predicted_data.csv')
    return data


def return_predict_hourly_count_for_station(station_name):
    df = pd.read_csv("../data/BikeshareQ3.csv")
    # choose which station to work
    station = df[df["to_station_name"] == station_name]

    # Transfer format.
    # station['trip_start_time'] = pd.to_datetime(station['trip_start_time'])
    station['trip_stop_time'] = pd.to_datetime(station['trip_stop_time'])

    # Split the month, day, hour and minute to separate columns
    # station['start_hour'] = station['trip_start_time'].apply(lambda x: x.hour)
    # station['start_minute'] = station['trip_start_time'].apply(lambda x: x.minute)
    # station['start_month'] = station['trip_start_time'].apply(lambda x: x.month)
    # station['start_day'] = station['trip_start_time'].apply(lambda x: x.day)

    station['stop_hour'] = station['trip_stop_time'].apply(lambda x: x.hour)
    # station['stop_minute'] = station['trip_stop_time'].apply(lambda x: x.minute)
    station['stop_month'] = station['trip_stop_time'].apply(lambda x: x.month)
    station['stop_day'] = station['trip_stop_time'].apply(lambda x: x.day)

    weather = pd.read_csv("../data/Toronto_temp.csv")
    weather["Date/Time"] = weather["Date/Time"].str.replace(',',  '')
    weather['Date/Time'] = pd.to_datetime(weather['Date/Time'])
    weather.rename(columns={'Date/Time': 'Date'}, inplace=True)
    weather['Date'] = pd.to_datetime(weather['Date']).dt.date

    station['trip_stop_time'] = pd.to_datetime(station['trip_stop_time']).dt.date

    merged = station.merge(weather, left_on='trip_stop_time', right_on='Date')
    train = merged.drop(['trip_start_time', 'trip_stop_time', 'to_station_name', 'user_type',
                         'Year', 'Month', 'Day', 'Date'], axis=1)

    # rent_bike = train.drop(['trip_id', 'from_station_name', 'start_minute', 'start_hour',
    #                         'stop_minute', 'start_month', 'start_day'], axis=1)
    rent_bike = train.drop(['trip_id', 'from_station_name'], axis=1)
    rent_bike = rent_bike[rent_bike.stop_month == 8]
    rent_bike['hourly_count'] = rent_bike.groupby('stop_hour')['stop_hour'].transform('count')
    rent_bike['duration_avg'] = rent_bike.groupby(['stop_hour', 'stop_month',
                                                   'stop_day'])['trip_duration_seconds'].transform('mean')
    rent_bike = rent_bike.drop_duplicates(subset=['stop_hour', 'stop_month', 'stop_day', 'duration_avg'], keep='last')
    rent_bike = rent_bike.drop(['trip_duration_seconds', 'season'], axis=1)

    x = rent_bike.drop('hourly_count', axis=1)  # Features
    y = rent_bike['hourly_count']  # Target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

    # Initializing the Random Forest Regression model with 10 decision trees
    model = RandomForestRegressor(n_estimators=10, random_state=0)

    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    # Predicting the target values of the test set
    y_pred = model.predict(x_test)

    # RMSE (Root Mean Square Error) and r2 score
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE: ", rmse)
    print('R2:', r2_score(y_test, y_pred))

    station_month7 = train.drop(['trip_id', 'from_station_name', 'season'], axis=1)
    station_month7 = station_month7[station_month7.stop_month == 7]
    station_month7['hourly_count'] = station_month7.groupby('stop_hour')['stop_hour'].transform('count')
    station_month7['duration_avg'] = station_month7.groupby(
        ['stop_hour', 'stop_month', 'stop_day'])['trip_duration_seconds'].transform('mean')
    station_month7 = station_month7.drop(['trip_duration_seconds'], axis=1)
    station_month7 = station_month7.drop_duplicates(subset=['stop_hour', 'stop_month', 'stop_day', 'duration_avg'],
                                                    keep='last')
    df7_train = station_month7.drop('hourly_count', axis=1)
    df7_target = station_month7['hourly_count']

    df7_pred = model.predict(df7_train)
    print('R2:', r2_score(df7_target, df7_pred))

    data = df7_train.copy()
    data['hourly_count'] = df7_pred
    data = data.groupby(['stop_hour']).mean()
    data = data[['hourly_count']]
    # data.to_csv('predicted_data.csv')
    return data


if __name__ == '__main__':
    # print(data)
    station_list = ["Bay St / St. Joseph St", "Union Station",
                    "College St / Major St", "Queens Quay / Yonge St", "Madison Ave / Bloor St W"]
    # ind = 0
    # for s in station_list:
    #     data = rent_predict_hourly_count_for_station(s)
    #     data.to_csv('../data/rent_predicted_data_{station_id}.csv'.format(station_id=ind))
    #     ind += 1

    ind = 0
    for s in station_list:
        data = return_predict_hourly_count_for_station(s)
        data.to_csv('../data/return_predicted_data_{station_id}.csv'.format(station_id=ind))
        ind += 1
