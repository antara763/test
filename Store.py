from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
import base64
import io
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats import norm
import os
from pathlib import Path
import uuid
import atexit

CWD = Path(__file__).parent
os.makedirs(CWD / "uploads", exist_ok=True)

url_theme1 = dbc.themes.PULSE
url_theme2 = dbc.themes.SLATE

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[url_theme1])

def delete_csv_file(file_path):
    if file_path:
        os.remove(file_path)

atexit.register(delete_csv_file, file_path=None)

app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
    html.Div(style={'flex': 'none', 'padding': '20px', 'textAlign': 'center'}, children=[
        html.H1("Outlier Detection")
    ]),
    html.Div(style={'flex': '1', 'display': 'flex'}, children=[
        html.Div(style={'flex': '3', 'padding': '20px', 'border': '1px dotted black'}, children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Choose CSV file'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '10px'
                },
                multiple=False,
            ),
             html.Div(id='upload-message', children="No file uploaded"),
            html.Div([
                html.Div("Selected Column:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[],
                    style={'width': '100%', 'margin': '10px' },
                    # placeholder="Select a column",
                    persistence=True,
                )
            ]),
            html.H3('Selected Algorithm:'), 
            dcc.Dropdown(
                id='algorithm-dropdown',
                options=[
                    {'label': 'IQR', 'value': 'IQR'},
                    {'label': 'SDOutlierDetection', 'value': 'SDOutlierDetection'},
                    {'label': 'ZScore', 'value': 'ZScore'},
                    {'label': 'ZScoreModifier', 'value': 'ZScoreModifier'},
                    {'label': 'IsolationForestDetector', 'value': 'IsolationForestDetector'},
                    {'label': 'DBSCANOutlierDetection', 'value': 'DBSCANOutlierDetection'}
                ],
                value='IQR',
                style={'width': '100%', 'marginBottom': '10px'},
                persistence=True 
            ),
            html.H3('Selected Graph:'),
            dcc.Dropdown(
                id='graph-dropdown',
                options=[
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 'Violin Plot', 'value': 'violin'},
                    {'label': 'Q-Q Plot', 'value': 'qq_plot'}
                ],
                value='box',
                style={'width': '100%', 'marginBottom': '10px'},
                persistence=True 
            )
        ]),
        
        html.Div(style={'flex': '7', 'padding': '20px', 'border': '1px dotted black'}, children=[
            dcc.Graph(
                id='output-graph',
                config={'displaylogo': False}
            )
        ])
    ]),
    html.Div([
        ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2])
    ]),
    dcc.Store(id='memory', storage_type='session'),
])

@app.callback(
    Output('upload-message', 'children'),
    [Input('memory', 'data')]
)
def update_upload_message(data):
    if data:
        return "Upload CSV file"
    else:
        return "No file uploaded"

@app.callback(
    Output('memory', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
)
def store_data(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        
        file_prefix, file_ext = os.path.splitext(filename)
        if file_ext not in [".csv"]:
            return ""

        upload_file_path = CWD / "uploads" / f"{file_prefix}_{str(uuid.uuid4())[:8]}{file_ext}"
        
        with open(upload_file_path, 'w', encoding='utf-8') as f:
            decoded = base64.b64decode(content_string)    
            f.write(decoded.decode('utf-8'))
        

        atexit.unregister(delete_csv_file)
        atexit.register(delete_csv_file, file_path=str(upload_file_path))
        
        return str(upload_file_path)
    else:
        raise PreventUpdate


@app.callback(
    Output('column-dropdown', 'options'),
    Output('column-dropdown', 'value'),
    Input('memory', 'modified_timestamp'),
    State('memory', 'data')
)
def update_output(modified_timestamp, data):
    if not data:
        raise PreventUpdate
    
    with open(data, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    column_names = [{'label': col, 'value': col} for col in df.columns]
    selected_column = column_names[0]['value']

    return column_names, selected_column


def detect_outliers(data, algorithm):
    if algorithm == 'IQR':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = np.where((data < lower_bound) | (data > upper_bound), 1, 0)
    elif algorithm == 'SDOutlierDetection':
        mean = np.mean(data)
        std = np.std(data)
        threshold = 2 * std
        lower_bound = mean - threshold
        upper_bound = mean + threshold
        outliers = np.where((data < lower_bound) | (data > upper_bound), 1, 0)
    elif algorithm == 'ZScore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        threshold = 3
        outliers = np.where(z_scores > threshold, 1, 0)
    elif algorithm == 'ZScoreModifier':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        threshold = 3
        modified_data = np.where(z_scores > threshold, np.mean(data), data)
        outliers = np.where(z_scores > threshold, 1, 0)
        data = modified_data
    elif algorithm == 'IsolationForestDetector':
        clf = IsolationForest(random_state=0)
        outliers = clf.fit_predict(data.reshape(-1, 1))
    elif algorithm == 'DBSCANOutlierDetection':
        clf = DBSCAN(eps=3, min_samples=2)
        outliers = clf.fit_predict(data.reshape(-1, 1))
    else:
        outliers = np.zeros(len(data))
    return outliers


@app.callback(
    Output('output-graph', 'figure'),
    Input('memory', 'modified_timestamp'),
    Input('column-dropdown', 'value'),
    Input('algorithm-dropdown', 'value'),
    Input('graph-dropdown', 'value'),
    State('memory', 'data'),
    prevent_initial_call=True,
)
def update_graph(modified_timestamp, selected_column, selected_algorithm, selected_graph, file_path):
    if file_path is None:
        return {}  
    df = pd.read_csv(file_path)

    data = df[selected_column].values
    outliers = detect_outliers(data, selected_algorithm)
    
    if selected_graph == 'box':
        fig = go.Figure()
        fig.add_trace(go.Box(y=data, name='Box Plot'))
        
        fig.update_layout(title='Box Plot ')
        return fig
    elif selected_graph == 'scatter':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data, mode='markers', name='Non Outlier'))

        outliers_indices = np.where(outliers == 1)[0]
        if len(outliers_indices) > 0:
            outliers_values = [data[i] for i in outliers_indices]
            fig.add_trace(go.Scatter(x=outliers_indices, y=outliers_values, mode='markers', name='Outliers', marker=dict(color='red', size=9, symbol='circle')))
        
        fig.update_layout(title='Scatter Plot ')
        return fig
    elif selected_graph == 'violin':
        fig = go.Figure()
        fig.add_trace(go.Violin(y=data, box_visible=True,line_color='black', name='Violin Plot'))
        
        fig.update_layout(title='Violin Plot')
        return fig

    elif selected_graph == 'qq_plot':
        sorted_data = np.sort(data)
        mu, sigma = norm.fit(data)
        fitted_samples = np.random.normal(mu, sigma, len(data))
        theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(data)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Non Outlier'))
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, mode='lines', name='Expected Quantiles', line=dict(color='red', width=2)))
        outliers_indices = np.where(outliers == 1)[0]
        if len(outliers_indices) > 0:
            outliers_values = [sorted_data[i] for i in outliers_indices]
            fig.add_trace(go.Scatter(x=outliers_values, y=np.zeros(len(outliers_values)), mode='markers', name='Outliers', marker=dict(color='black', size=9, symbol='circle')))
    
        fig.update_layout(title='Q-Q Plot')
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)
