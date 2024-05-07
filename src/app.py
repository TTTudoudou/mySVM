import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import svm2
import plattSMO
import lssvm2
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__)

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Define app layout
app.layout = html.Div([
    html.H1("SVM Classifier"),
    html.Label("Select dataset:"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Small Dataset', 'value': 'SmallDataSet.txt'},
            {'label': 'Medium Dataset', 'value': 'MediumDataSet.txt'},
            {'label': 'Large Dataset', 'value': 'LargeDataSet.txt'}
        ],
        value='SmallDataSet.txt'
    ),
    html.Button('Train SVM (svm)', id='svm-button', n_clicks=0),
    html.Button('Train SVM (SMO)', id='plattsmo-button', n_clicks=0),
    html.Button('Train SVM (lssvm)', id='lssvm-button', n_clicks=0),
    html.Div(id='svm-output'),
    html.Div(id='platt-output'),
    html.Div(id='lssvm-output'),
    html.Div([
        dcc.Graph(id='svm-graph'),
        dcc.Graph(id='platt-graph'),
        dcc.Graph(id='lssvm-graph')
    ])

])


# 定义回调函数
@app.callback(
    [Output('svm-output', 'children'),
     Output('svm-graph', 'figure')],
    [Input('svm-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def update_plot_svm(n_clicks, dataset):
    if n_clicks > 0:
        dataMat, labelMat = svm2.loadDataSet(dataset)
        svm_classifier = svm2.SVM(gamma=0.1)
        svm_classifier.fit(np.array(dataMat), np.array(labelMat))

        figure = plot_decision_boundary(dataMat, labelMat, svm_classifier)

        execution_time, errorRate = svm2.performance_measure(dataMat, labelMat)
        children = f"svm: Execution Time: {execution_time:.2f} sec, Error Rate: {errorRate:.2f}"

        return children, figure
    else:
        return "", {}


def plot_decision_boundary(dataMat, labelMat, svm_classifier):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    x_min, x_max = min(dataMat[:, 0]), max(dataMat[:, 0])
    y_min, y_max = min(dataMat[:, 1]), max(dataMat[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    trace_contour = go.Contour(
        x=np.linspace(x_min, x_max, 50),
        y=np.linspace(y_min, y_max, 50),
        z=Z,
        colorscale='Greys',
        opacity=0.5,
        contours=dict(
            start=-1,
            end=1,
            size=1,
        ),
        line=dict(
            dash='dash',
            color='black',
            width=2
        ),
    )

    trace_data = [go.Scatter(x=dataMat[labelMat == 1][:, 0], y=dataMat[labelMat == 1][:, 1], mode='markers',
                             marker=dict(color='pink', symbol='square', size=10), name='Class 1'),
                  go.Scatter(x=dataMat[labelMat == -1][:, 0], y=dataMat[labelMat == -1][:, 1], mode='markers',
                             marker=dict(color='green', symbol='square', size=10), name='Class -1')]

    layout = go.Layout(title='SVM Decision Boundary')

    return {'data': trace_data + [trace_contour], 'layout': layout}


@app.callback(
    [Output('platt-output', 'children'),
     Output('platt-graph', 'figure')],
    [Input('plattsmo-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def update_plot_platt(n_clicks, dataset):
    if n_clicks > 0:
        dataMat, labelMat = plattSMO.loadDataSet(dataset)
        dataMat = np.array(dataMat)
        labelMat = np.array(labelMat)
        smo = plattSMO.PlattSMO(dataMat, labelMat, 6, 0.0001, 10000, name='rbf', theta=1.3)
        smo.smoP()

        figure = plot_decision_boundary_smo(dataMat, labelMat, smo.SV, smo)

        execution_time, error = plattSMO.performance_measure(dataMat, labelMat)
        children = f"SMO： Execution Time: {execution_time:.2f} sec, Error Rate: {error:.2f}"

        return children, figure
    else:
        return "", {}


def plot_decision_boundary_smo(data, label, SV, smo):
    dataMat = np.array(data)
    labelMat = np.array(label)
    x_min, x_max = min(dataMat[:, 0]), max(dataMat[:, 0])
    y_min, y_max = min(dataMat[:, 1]), max(dataMat[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    Z = np.array(smo.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    trace_contour = go.Contour(
        x=np.linspace(x_min, x_max, 50),
        y=np.linspace(y_min, y_max, 50),
        z=Z,
        colorscale='Greys',
        opacity=0.5,
        contours=dict(
            start=-1,
            end=1,
            size=1,
        ),
        line=dict(
            dash='dash',
            color='black',
            width=2
        ),
    )

    trace_data = [go.Scatter(x=dataMat[labelMat == 1][:, 0], y=dataMat[labelMat == 1][:, 1], mode='markers',
                             marker=dict(color='pink', symbol='square', size=10), name='Class 1'),
                  go.Scatter(x=dataMat[labelMat == -1][:, 0], y=dataMat[labelMat == -1][:, 1], mode='markers',
                             marker=dict(color='green', symbol='square', size=10), name='Class -1')]

    layout = go.Layout(title='SVM Decision Boundary')

    return {'data': trace_data + [trace_contour], 'layout': layout}


@app.callback(
    [Output('lssvm-output', 'children'),
     Output('lssvm-graph', 'figure')],
    [Input('lssvm-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def update_plot_lssvm(n_clicks, dataset):
    if n_clicks > 0:
        dataMat, labelMat = lssvm2.loadDataSet(dataset)
        dataMat = np.array(dataMat)
        labelMat = np.array(labelMat)
        C = 0.6
        k1 = 0.3
        kernel = 'rbf'
        kTup = (kernel, k1)
        alphas, b, K = lssvm2.leastSquares(dataMat, labelMat, C, kTup)

        # Plot decision boundary
        figure = plot_decision_boundary_lssvm(dataMat, labelMat, alphas, b, K)

        execution_time, error = lssvm2.performance_measure(dataMat, labelMat, C, kTup)
        children = f"lssvm： Execution Time: {execution_time:.2f} sec, Error Rate: {error:.2f}"

        return children, figure
    else:
        return "", {}


def plot_decision_boundary_lssvm(dataMat, labelMat, alphas, b, K):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    x_min, x_max = min(dataMat[:, 0]), max(dataMat[:, 0])
    y_min, y_max = min(dataMat[:, 1]), max(dataMat[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.sign(np.dot(alphas * labelMat, K) + b)
    Z = Z.reshape(xx.shape)

    trace_contour = go.Contour(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=Z,
        colorscale='Greys',
        opacity=0.5,
        contours=dict(
            start=-1,
            end=1,
            size=1,
        ),
        line=dict(
            dash='dash',
            color='black',
            width=2
        ),
    )

    trace_data = [go.Scatter(x=dataMat[labelMat == 1][:, 0], y=dataMat[labelMat == 1][:, 1], mode='markers',
                             marker=dict(color='pink', symbol='square', size=10), name='Class 1'),
                  go.Scatter(x=dataMat[labelMat == -1][:, 0], y=dataMat[labelMat == -1][:, 1], mode='markers',
                             marker=dict(color='green', symbol='square', size=10), name='Class -1')]

    layout = go.Layout(title='LSSVM Decision Boundary')

    return {'data': trace_data + [trace_contour], 'layout': layout}



if __name__ == '__main__':
    app.run_server(debug=True)
