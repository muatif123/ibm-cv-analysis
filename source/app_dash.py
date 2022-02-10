from http import server
import imp
from multiprocessing.sharedctypes import Value
import os
from turtle import ht, title
import pandas as pd
import math
from joblib import load
from source.data_processing import make_full_pipeline
import pickle

import dash
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

HOME_PATH = os.getcwd()
MODELS_PATH = os.path.join(HOME_PATH, 'model')
ASSETS_PATH = os.path.join(HOME_PATH, 'assets')

df = pd.read_csv('https://raw.githubusercontent.com/Athena75/IBM-Customer-Value-Dashboarding/main/data/Customer-Value-Analysis.csv', index_col = 'Customer')
sk_best = load(os.path.join(MODELS_PATH, 'best.joblib'))

full_pipeline = make_full_pipeline(df)

ohe_path = os.path.join(MODELS_PATH, 'ohe_categories.pkl')
perfs_path = os.path.join(MODELS_PATH, 'sk_best_performances.pkl')

with open(ohe_path, 'rb') as input:
    ohe_categories = pickle.load(input)

categories = []
for i, j in ohe_categories.items():
    categories.append([f'{i}_{catg}' for catg in list(j)])
flatten = lambda j: [item for sublist in j for item in sublist]
categories = flatten(categories)

with open(perfs_path, 'rb') as input:
    perfs = pickle.load(input)

# Scaling
catgs = [var for var, var_type in df.dtypes.items() if var_type == 'object']
numls = [var for var in df.columns if var not in catgs]
catgs.remove('Response')

Top = 10

# Creating a DataFrame to store the features, importance and their corresponding label
df_feature_importances = pd.DataFrame(sk_best.feature_importances_ * 100, columns = ['Importance'],
                                        index = numls + categories)
df_feature_importances = df_feature_importances.sort_values('Importance', ascending = False)
df_feature_importances = df_feature_importances.loc[df_feature_importances.index[:Top]]


# Creating a Feature Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x = df_feature_importances.index,
                                    y = df_feature_importances['Importance'],
                                    marker_color = 'rgb(171, 226, 251)'))
fig_features_importance.update_layout(title_text = '<b>Features Importance of the model<b>', title_x = 0.5)

# Creating Feature Performances Bar Chart
fig_performaces = go.Figure()
fig_performaces.add_trace(go.Bar(y = list(perfs.keys()),
                                 x = list(perfs.values()),
                                 marker_color = 'rgb(171, 226, 251)',
                                 orientation = 'h'))
fig_performaces.update_layout(title_text = '<b>Best Model Performaces<b>', title_x = 0.5)

catg_children = []
for var in catgs:
    sorted_modalities = list(df[var].value_counts().index)
    catg_children.append(html.H4(children = var))
    catg_children.append(dcc.Dropdown(
        id = '{}-dropdown'.format(var),
        options = [{'label': value, 'value': value} for value in sorted_modalities],
        value = sorted_modalities[0]
    ))

linear_children = []
for var in numls:
    linear_children.append(html.H4(children = var))
    desc = df[var].describe()
    linear_children.append(dcc.Slider(
        id = '{}-dropdown'.format(var),
        min = math.floor(desc['min']),
        max = round(desc['max']),
        step = None,
        value = round(desc['mean']),
        marks = {i: '{}'.format(i) for i in 
                range(int(desc['min']), int(desc['max']) + 1, max(int((desc['std'] / 1.5)), 1))}
    ))


app = dash.Dash(__name__,
        external_stylesheets = ["https://rawcdn.githack.com/Athena75/IBM-Customer-Value-Dashboarding/df971ae38117d85c8512a72643ce6158cde7a4eb/assets/style.css"])

app.layout = html.Div(children = [
    html.Div(children = [
        html.Div(children = [
            html.H1(children = 'Simulation Tool : IBM Customer Churn'),
        ],
        className = 'title'),
    ],
    style = {"display": "block"}),

    # second row :
    html.Div(children = [
        # first column : fig feature importance + linear + prediction
        html.Div(children = [
            html.Div(children = [
                dcc.Graph(figure = fig_features_importance, className = 'graph')] + linear_children),
            # prediction result
            html.Div(children = [
                html.H2(children = "Prediction:"),
                               html.H2(id = "prediction_result")],
                     className = 'prediction')],
                 className = 'column'),
        # second column : fig performances categorical
        html.Div(children = [
             dcc.Graph(figure = fig_performaces, className = 'graph')] + catg_children,
                 className = 'column')
    ],
        className = 'row')
])

# The callback fn will provide one output in the form of a string 
@app.callback(Output(component_id = 'prediction_result', component_property = "childres"),
            [Input('{}-dropdown'.format(var), 'value') for var in numls + catgs])


def update_prediction(*X):
    payload = dict(zip(numls + catgs, X))
    frame_X = pd.DataFrame(payload, index = [0])

    X_processed = full_pipeline.transform(frame_X)
    prediction = sk_best.predict_proba(X_processed)[0]

    return " {}% No , {}% Yes".format("%.2f" % (prediction[0] * 100),
                                      "%.2f" % (prediction[1] * 100))

server = app.server

    