import dash

# style sheets for app
external_stylesheets = []

# style js scripts for app
external_scripts  = []

# init the dash app
app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
