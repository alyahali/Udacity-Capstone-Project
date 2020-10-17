#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import plotly.graph_objs as go

import flask
import pandas as pd
import numpy as np
import colorlover as cl
import yfinance as yf
import re

from app import app
from utils import bbands

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


# In[2]:


# define business day
B_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# In[3]:


def download_stock_to_df(symbol, start, end):
    """
    Get current stocks data from yahoo fiance and save to dataframe

    Params:
        symbol: stock to pull data
        start: start date of pulled data
        end: end date of pulled data

    Return:
        dataframe of stock within specified date range
    """
    df_stock=yf.download(symbol,start,end,progress=False)
    df_stock.reset_index(level=0, inplace=True)
    return df_stock


# In[4]:


def generate_train_test_sequence(t_data, lookback_days, forcast_days, sequence, last_index):
    """
    Generate sequence array for train data, test data
    inclduing both X and y
    """
    X = []
    y = []
    m = lookback_days
    n =forcast_days
    N = len(t_data)
 
    # train sequence is continuous = 1 
    # test sequence is not continuous, will be in 20 groups, each group contains 7 days

    for i in range(0, N, sequence):
        # input sequence : x[i].....x[i+m]
        # output sequence: x[i+m+1]....x[i+m+n] 
        # last index is (i+m+n)
        end_index = i + m + n # find the end of this sequence
        # check if we are out of index
        if end_index > N-last_index:
            break
        seq_x = t_data[i:(i+m)]
        seq_y = t_data[(i+m):(i+m+n)]
        X.append(seq_x)
        y.append(seq_y)
        
    array_X = np.array(X) # shape (N, m, 1)
    array_y = np.array([list(a.ravel()) for a in np.array(y)]) # shape (N, n, 1) convert to (N, n)
    return array_X, array_y


# In[5]:


def predict_future_price(df, lookback_days, forcast_days, model, sc):
    """
    Prediction of future stock price
    """

    # calculate forcast dates and indexs, used for plot
    B_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar()) # business day
    forcast_dates = [df['Date'].iloc[-1]+i*B_DAY for i in range(1,forcast_days+1)]
    forcast_indexs = [df['Date'].index[-1]+i for i in range(1,forcast_days+1)]

    # input X for forcast
    X_scaled = df[['Scaled_Close']][-lookback_days:].values
    X_scaled = X_scaled.reshape((1, lookback_days, 1))

    # prediction the scaled price with model
    forcast_price_scaled = model.predict(X_scaled)

    # transform back to the normal price
    forcast_prices = sc.inverse_transform(forcast_price_scaled)

    # create the forcast_dataframe
    forcast_dataframe = pd.DataFrame({
        'index' : forcast_indexs,
        'Date' : forcast_dates,
        'Adj Close' : forcast_prices[0],
        'Scaled_Close' : forcast_price_scaled[0]
    })

    forcast_dataframe = forcast_dataframe.set_index('index', drop=True)
    return forcast_dataframe


# In[13]:


def init_callbacks(symbol):
    """
    Function to init all callbacks in the dash app

    Parameters
    ----------
    df: DataFrame

    Returns:
    ---------
    None
    """

    @app.callback(
        dash.dependencies.Output('output','children'), [ dash.dependencies.Input('reset_button','n_clicks')])
    def update(reset):
        if reset > 0:
            reset = 0
            return 'all clear'



    @app.callback(
        dash.dependencies.Output(component_id='graph', component_property='children'),
        [
            dash.dependencies.Input(component_id='stock-symbol-input', component_property='value'),
            dash.dependencies.Input('start-date-picker', 'date'),
            dash.dependencies.Input('end-date-picker', 'date'),
            dash.dependencies.Input('input_button', 'n_clicks')
        ]
    )

    def update_graph(symbol, start_date, end_date, n_clicks):

        if n_clicks < 1:
            return "Click Run Model after date selections."

        start_date = str(start_date)
        end_date = str(end_date)

        df_stock = download_stock_to_df(symbol, start_date, end_date) 
        df = df_stock.copy(deep=True)

        df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # scale all data
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0.01,0.99))
        df['Scaled_Close'] = sc.fit_transform(df[['Adj Close']])

        # define some parameters
        num_periods = 40
        forcast_days = 7
        lookback_days = 15

        N = df.shape[0]
        N_test = num_periods*forcast_days
        N_train = N - N_test

        # split into train and test sets
        dataset_train , dataset_test = train_test_split(df, train_size=N_train, test_size=N_test, shuffle=False)
        dataset_test_extend = dataset_train[-lookback_days:].append(dataset_test)

        train_set = dataset_train[['Scaled_Close']].values
        test_set = dataset_test[['Scaled_Close']].values
        test_set_extend = dataset_test_extend[['Scaled_Close']].values

        # create train set sequence
        X_train, y_train = generate_train_test_sequence(train_set, lookback_days, forcast_days, 1, 1)
        # X_train, y_train = generate_train_sequence(train_set_scaled, lookback_days)

        # define model
        model = Sequential()
        model.add(LSTM(units=30, activation='relu', input_shape=(lookback_days,1)))
        model.add(Dense(forcast_days))
        model.compile(optimizer='adam', loss='mean_squared_error')

        num_epochs = 200
        history  = model.fit(X_train,y_train,epochs=num_epochs,batch_size=32)

        # create test set sequence
        X_test, y_test = generate_train_test_sequence(test_set_extend, lookback_days, forcast_days,forcast_days,0)
        

        # run prediction for the test dataset
        LSTM_prediction_scaled = model.predict(X_test)
        LSTM_prediction = sc.inverse_transform(LSTM_prediction_scaled)

        train_set = train_set.reshape((-1))
        test_set = test_set.reshape((-1))
        LSTM_prediction = LSTM_prediction.reshape((-1))

        dataset_test['LSTM_Prediction'] = LSTM_prediction

        # forcast
        dataset_forcast = predict_future_price(df, lookback_days, forcast_days, model, sc)


        trace1 = go.Scatter(
            x = dataset_train['Date'],
            y = dataset_train['Adj Close'],
            # mode = 'lines',
            line = dict(color='blue', width=2),
            name = 'Train'
        )
        trace2 = go.Scatter(
            x = dataset_test['Date'],
            y = dataset_test['Adj Close'],
            # mode='lines',
            line = dict(color='blue', width=2),
            name = 'Test-True-Price'
        )
        trace3 = go.Scatter(
            x = dataset_test['Date'],
            y = dataset_test['LSTM_Prediction'],
            # mode = 'lines',
            line = dict(color='red', width=2),
            name = 'Test-LSTM-Prediction'
        )
        trace4 = go.Scatter(
        x = dataset_forcast['Date'],
        y = dataset_forcast['Adj Close'],
        line = dict(color='green', width=2, dash='dot'),
        name = 'Forcast'
    )
        layout = go.Layout(
            title = "Stock".format(symbol),
            xaxis = {'title' : "Date"},
            yaxis = {'title' : "Adj Close ($)"}
        )
        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
        fig.update_layout(xaxis_range=[dataset_train['Date'].iloc[0], dataset_forcast['Date'].iloc[-1]+100*B_DAY])

        graph = dcc.Graph(
            id='stock_prediction_graph',
            figure=fig
        )

        return graph


# In[ ]:




