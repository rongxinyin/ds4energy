__author__ = "Rongxin Yin"
__email__ = "ryin@lbl.gov"

import os
import json
import csv
import pandas as pd
import numpy as np
import pathlib
from datetime import datetime, timedelta
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt


class BaselineModels(object):
    def __init__(self, site, input_data, event):
        self.site = site
        self.data = input_data

        # event day
        self.event = event
        self.event_start = datetime.strftime(datetime.strptime(
            event['shed_start_time'], '%m/%d/%y %H:%M'), '%H:%M')
        self.shed_start = datetime.strftime(datetime.strptime(event['shed_start_time'], '%m/%d/%y %H:%M') + timedelta(minutes=15),
                                            '%H:%M')
        self.shed_end = datetime.strftime(datetime.strptime(
            event['shed_end_time'], '%m/%d/%y %H:%M'), '%H:%M')

    def get_x_baseline_days(self, x):
        train_start_date = datetime.strftime(datetime.strptime(self.event['date'], '%Y-%m-%d') - timedelta(days=x),
                                             '%Y/%m/%d')
        train_end_date = datetime.strftime(datetime.strptime(self.event['date'], '%Y-%m-%d') - timedelta(days=1),
                                           '%Y/%m/%d')
        train_data = self.data[self.data['valid_dates']
                               == 1][train_start_date:train_end_date]
        if train_data.empty:
            print('training baseline data is empty')
        else:
            print('training baseline data is not empty')
        # print(train_data.head())
        return train_data

    def calc_x_y_baseline_days(self, x, y):
        # event type and data_day timestamp
        if x > y:
            print("Input of 'x' should be equal or less than 'y'.")
        # event_date = self.event.date
        # data_wk = self.data[(self.data['valid_dates'] == 1)]
        train_data = self.get_x_baseline_days(x=30)
        baseline_y_days = sorted(set(train_data.day))[-y:]
        baseline_y_data = train_data[train_data['day'].isin(baseline_y_days)]
        # calculate and sort daily peak power
        selected_x_days = baseline_y_data.groupby(
            'day')['power'].max().sort_values()[-x:].index.tolist()
        selected_x_data = baseline_y_data[baseline_y_data['day'].isin(
            selected_x_days)]

        return selected_x_data

    def calc_match_baseline(self, x, y):
        # get the matching baseline day from previous 10 valid baseline days
        baseline_data = self.calc_x_y_baseline_days(x, y)
        # calculate avg baseline
        dr_data = self.data[self.event.date][[
            'time', 'day', 'hour', 'oat', 'power']].copy()
        oa_sum_diff = pd.Series(index=list(set(baseline_data.day)))
        for i in baseline_data.day.unique():
            #             print(baseline_data[i].oat.values-dr_data.oat.values)
            try:
                oa_sum_diff[i] = np.sum(
                    np.square(baseline_data[i].oat.values - dr_data.oat.values), axis=0)
            except ValueError:
                print('Missing OA data on {}'.format(i))
        match_day = oa_sum_diff.idxmin()
        try:
            dr_data['baseline'] = baseline_data[match_day].power.values
        except KeyError:
            print("Data is missing on {}".format(self.event.date))
            pass
        return dr_data[['oat', 'baseline', 'power']]

    def calc_avg_baseline_adj(self, x, y, adj_limit):
        # event
        adj_start = datetime.strftime(datetime.strptime(
            self.event_start, '%H:%M') + timedelta(hours=-5), '%H:%M')
        adj_end = datetime.strftime(datetime.strptime(
            self.event_start, '%H:%M') + timedelta(hours=-2), '%H:%M')
        # get baseline days
        baseline_data = self.calc_x_y_baseline_days(x, y)
        # calculate avg baseline
        dr_data = self.data[self.event['date']][[
            'time', 'day', 'hour', 'oat', 'power']].copy()
        # dr_data.index = dr_data.time
        dr_data['avg_baseline'] = baseline_data.groupby('time')['power'].mean()
        try:
            dr_data.loc[self.shed_start:self.shed_end,
                        ['event_hours']] = 1
            # calculate avg model adjustment factor
            adj_factor = np.mean(
                dr_data[adj_start:adj_end]['power'] / dr_data[adj_start:adj_end]['avg_baseline'])
            if adj_factor > (1 + adj_limit):
                adj_factor = (1 + adj_limit)
            elif adj_factor < (1 - adj_limit):
                adj_factor = (1 - adj_limit)
            dr_data['baseline'] = dr_data['avg_baseline'] * adj_factor
        except KeyError:
            print("Data is missing on {}".format(self.event.date))
            pass
        return dr_data[['oat', 'baseline', 'power']]

    def oat_reg_model(data):
        data['hr'] = data['hour'].astype(str)
        X = data[['oat', 'hr']]
        X_hr = pd.get_dummies(X)
        y_hr = data['power']

        X_train, X_test, y_train, y_test = train_test_split(X_hr, y_hr,
                                                            test_size=0.2, random_state=101)
        
        # train the regression model
        model = linear_model.LinearRegression()
        model.fit(X_train,y_train)

        # output model
        coeff_parameter = pd.DataFrame(model.coef_, X.columns,columns=['Coefficient'])
        print('Intercept: \n', model.intercept_)
        print('Coefficients: \n', model.coef_)
        return model


class SiteData(object):
    def __init__(self, site):
        self.site = site

        # Create sublevel directories
        self.root_dir = pathlib.Path.cwd()
        self.data_dir = self.root_dir.joinpath('example')
        self.out_dir = self.root_dir.joinpath('output/{}'.format(site))
        self.plot_dir = self.root_dir.joinpath(
            'plot/{}'.format(site['site_id']))

        # Create directories
        for dir_inst in [self.out_dir, self.plot_dir]:
            try:
                if not os.path.exists(dir_inst):
                    os.makedirs(dir_inst)
            except FileExistsError:
                continue
        # read meter, weather and event data
        self.holidays = self.read_special_days()
        self.event_days = self.read_event_days()

    def read_special_days(self):
        holidays = pd.read_csv(
            self.root_dir.joinpath(self.data_dir.joinpath('special-days.csv')))
        holidays['date'] = pd.to_datetime(
            pd.Series(holidays['date']), format='%m/%d/%y')
        holidays['day'] = holidays.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        return holidays

    def read_event_days(self):
        dr_event = pd.read_csv(self.data_dir.joinpath('dr-event.csv'))
        dr_event['event_date'] = dr_event.event_date.apply(
            lambda x: datetime.strptime(x, '%m/%d/%y'))
        dr_event['date'] = dr_event.event_date.apply(
            lambda x: x.strftime('%Y-%m-%d'))
        return dr_event

    def read_site_data(self):
        df = pd.read_csv(self.data_dir.joinpath(
            'site-meter-weather.csv', index_col=[0], parse_dates=True))
        # fill missing data or na
        df = df.fillna(method='ffill')
        # remove duplicated index
        df = df[~df.index.duplicated(keep='first')]

        # clean dataframe
        df['date'] = df.index
        # df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
        df['time'] = df.date.apply(lambda x: x.strftime('%H:%M'))
        # df['month'] = df.date.apply(lambda x: int(x.strftime('%m')))
        df['day'] = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        df['hour'] = df.date.apply(lambda x: int(x.strftime('%H')))
        # df['minute'] = df.date.apply(lambda x: x.strftime('%M'))
        df['weekday'] = df.date.apply(lambda x: int(x.strftime('%w')))
        df['DR'] = df.date.apply(
            lambda x: x.strftime('%Y-%m-%d') in self.event_days.date.values)
        df['DR'] = df.DR.astype(int)
        df['holiday'] = df.date.apply(lambda x: x.strftime(
            '%Y-%m-%d') in self.holidays.day.values)
        df['holiday'] = df.holiday.astype(int)
        df['valid_dates'] = 0
        df.loc[(df['weekday'] > 0) & (df['weekday'] < 6) & (
            df['holiday'] == 0) & (df['DR'] == 0), ['valid_dates']] = 1
        print('read the preprocessed meter and weather data.')
        return df

    # read weather data if needed
    def read_weather_data(self):
        df = pd.read_csv(self.data_dir.joinpath('example/site_weather.csv'),
                         index_col=[0], parse_dates=True)
        # df.index = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S')
        df['oat'] = df.temperature*1.8+32
        df = df.fillna(method='ffill')
        df = df[~df.index.duplicated(keep='last')]

        # resample to 15 minutes
        df = df.resample('15min').interpolate(method='linear')
        print('read the weather data.')
        # print(df.head())
        return df['oat']

    def calc_load_stats(self):
        test2 = self.data[(self.data['year'] == '2019')][[
            'power', 'oat', 'month']].resample('1h').mean()
        df_wk = test2[(test2.hour > 6) & (test2.hour < 23) &
                      (test2.weekday > 0) & (test2.weekday < 7)]
        df_wkd = test2[~((test2.hour > 6) & (test2.hour < 23)
                         & (test2.weekday > 0) & (test2.weekday < 7))]
        test2_order = test2.sort_values(by=['power'], ascending=False)['power']
        metric = pd.Series(index=['Annual Electric Consumption',
                                  'Peak Electric Demand Summer',
                                  'Peak Electric Demand Winter',
                                  'Demand Threshold at the Top 50 Hours',
                                  'Annual Average Electric Energy Intensity',
                                  'Annual Peak Electric Demand Intensity',
                                  'Weather Sensitivity Occupied',
                                  'Weather Sensitivity Unoccupied'])
        metric['Annual Electric Consumption'] = test2.power.sum()
        metric['Peak Electric Demand Summer'] = test2[test2.month.isin(
            [6, 7, 8])].power.max()
        metric['Peak Electric Demand Winter'] = test2[test2.month.isin(
            [12, 1, 2])].power.max()
        metric['Annual Average Electric Energy Intensity'] = metric['Annual Electric Consumption'] / float(
            self.site['floor_area'])
        metric['Annual Peak Electric Demand Intensity'] = test2.power.max(
        ) * 1000 / float(self.site['floor_area'])
        metric['Weather Sensitivity Occupied'] = round(
            df_wk[['oat', 'power']].corr()['power'][0], 2)
        metric['Weather Sensitivity Unoccupied'] = round(
            df_wkd[['oat', 'power']].corr()['power'][0], 2)
        metric['Demand Threshold at the Top 50 Hours'] = test2_order[50]
        metric.to_csv(self.out_dir.joinpath(
            '{}_load-metric.csv'.format(self.site['site_id'])))
        return

# Calculate model metrics


def calc_model_metrics(y, y_predicted, num_predictor):
    # calculate model metrics
    MAE = "{:.3f}".format(mean_absolute_error(y, y_predicted))
    nMAE = "{:.3f}".format(mean_absolute_error(y, y_predicted)/np.mean(y))
    MBE = "{:.3f}".format(np.mean(y_predicted-y))
    nMBE = "{:.3f}".format(np.sum(y_predicted-y)/(len(y)-1)/np.mean(y))
    RSME = "{:.3f}".format(sqrt(mean_squared_error(y, y_predicted)))
    nRSME = "{:.3f}".format(
        sqrt(mean_squared_error(y, y_predicted))/np.mean(y))
    R2 = "{:.3f}".format(r2_score(y, y_predicted))
    if y.shape[0] - num_predictor - 1 == 0:
        R2_adj = 0
    else:
        R2_adj = 1 - (1 - r2_score(y, y_predicted)) * \
            ((y.shape[0] - 1) / (y.shape[0] - num_predictor - 1))
    return [MAE, nMAE, MBE, nMBE, RSME, nRSME, R2, R2_adj]
