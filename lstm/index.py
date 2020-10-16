import dash
import dash_core_components as dcc
import dash_html_components as html
import datetime
from datetime import datetime as dt

import pandas as pd
import numpy as np

from app import app
from callbacks import init_callbacks

today = datetime.date.today()

def run_server():
    """
    Main function to run the server
    """

    symbol_options = [
        {'label': 'Booking', 'value': 'BKNG'},
        {'label': 'Expedia', 'value': 'EXPE'},
        {'label': 'Netflix', 'value': 'NFLX'},
        {'label': 'Disney', 'value': 'DIS'},
        {'label': 'Six Flags', 'value': 'SIX'},  
        {'label': 'United Air', 'value': 'UAL'},
        {'label': 'Delta Air', 'value': 'DAL'},
        {'label': 'Carnival', 'value': 'CCL'},
        {'label': 'Royal', 'value': 'RCL'},
        {'label': 'Hyatt', 'value': 'H'},
        {'label': 'Marriott', 'value': 'MAR'}

    ]

    app.layout = html.Div([
        html.Div([
            html.H2('LSTM Stock Predictor for BEACH industries(booking, entertainment, airlines, cruises, and hotels)',
                    style={'display': 'inline',
                           'float': 'left',
                           'font-size': '2.65em',
                           'margin-left': '7px',
                           'font-weight': 'bolder',
                           'font-family': 'Product Sans',
                           'color': "rgba(117, 117, 117, 0.95)",
                           'margin-top': '20px',
                           'margin-bottom': '0'
                           }),
        ]),
        dcc.Dropdown(
            id='stock-symbol-input',
            # options=[{'label': s, 'value': s}
            #          for s in symbols],
            options = symbol_options,
            value='BKNG'
        ),

       
        html.Div([
            dcc.DatePickerSingle(
                id='start-date-picker',
                min_date_allowed=dt(1990, 1, 1),
                max_date_allowed=today,
                initial_visible_month=dt(2000, 1, 1),
                # date=dt(2015, 1, 1)
            ),
            # html.Div(id='start-date-output-container')
        ]),


        html.Div([
            dcc.DatePickerSingle(
                id='end-date-picker',
                min_date_allowed=dt(1990, 1, 1),
                max_date_allowed=today,
                initial_visible_month=dt(2020, 10, 5),
                # date=today
            ),
        ]),

        html.Button('Run Model', id='input_button', n_clicks=0),
   

        html.Div(id='graph')
    ], className="container")


    # initialize callbacks
    init_callbacks('BKNG')

    # app.run_server(debug=True, port=8080, threaded=False)
    app.run_server(debug=True, port=8080)


if __name__ == '__main__':
    run_server()
