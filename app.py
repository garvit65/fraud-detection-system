import base64
import io
import dash
from dash import html, dcc, dash_table, Input, Output
from backend.prediction import predict_fraud
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
# Run server
import os

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([

    html.H1("Financial Fraud Detection System", className="text-center mt-4 mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H4("Upload Transaction Dataset"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
                multiple=False
            )
        ])
    ], className="mb-4"),

    dcc.Store(id='stored-data'),
    html.Div(id='output-data-upload'),

    dcc.Interval(
        id='interval-component',
        interval=1000,
        n_intervals=0
    )

], fluid=True)

# Function to parse uploaded file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = predict_fraud(df)

        # Summary calculations
        total = len(df)
        fraud = df['Prediction'].sum()
        fraud_rate = round((fraud / total) * 100, 2)

        # Summary cards
        summary_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Total Transactions"),
                    html.H3(total)
                ])
            ]), width=4),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Fraud Transactions"),
                    html.H3(fraud, style={"color": "red"})
                ])
            ]), width=4),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Fraud Rate (%)"),
                    html.H3(f"{fraud_rate}%")
                ])
            ]), width=4),
        ])

        # Alert
        alert = None
        if fraud > 0:
            alert = dbc.Alert(
                f" {fraud} high-risk transactions detected!",
                color="danger",
                className="mt-3"
            )

    else:
        return html.Div(['Unsupported file format'])

    return html.Div([
        html.H5(f"File: {filename}"),
        html.H6(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"),
        summary_cards,
        alert,
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'center'
            },

            style_data_conditional=[
                {
                    'if': {'filter_query': '{Prediction} = 1'},
                    'backgroundColor': '#FFCCCC',
                    'color': 'black'
                },
                {
                    'if': {'filter_query': '{Risk} = "High"'},
                    'backgroundColor': '#FF6666',
                    'color': 'white'
                },
                {
                    'if': {'filter_query': '{Risk} = "Medium"'},
                    'backgroundColor': '#FFF3CD',
                    'color': 'black'
                }
            ]
        )
    ])

# Callback
@app.callback(
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename')
)
def store_data(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df.to_dict('records')

    return None

@app.callback(
    Output('output-data-upload', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('stored-data', 'data')
)
def update_live(n, data):

    if data is None:
        return html.Div()

    df = pd.DataFrame(data)

    # Simulate streaming (1 row at a time)
    df_live = df.head(n + 1)

    df_live = predict_fraud(df_live)

    # Chart 1: Fraud vs Normal count
    fraud_counts = df_live['Prediction'].value_counts().reset_index()
    fraud_counts.columns = ['Prediction', 'Count']

    bar_chart = px.bar(
        fraud_counts,
        x='Prediction',
        y='Count',
        title="Fraud vs Normal Transactions",
        color='Prediction'
    )

    # Chart 2: Risk distribution
    risk_counts = df_live['Risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Count']

    pie_chart = px.pie(
        risk_counts,
        names='Risk',
        values='Count',
        title="Risk Distribution"
    )

    # Summary calculations
    total = len(df_live)
    fraud = df_live['Prediction'].sum()
    fraud_rate = round((fraud / total) * 100, 2)

    summary_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Transactions"),
                html.H3(total)
            ])
        ]), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Fraud Transactions"),
                html.H3(fraud, style={"color": "red"})
            ])
        ]), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Fraud Rate (%)"),
                html.H3(f"{fraud_rate}%")
            ])
        ]), width=4),
    ])

    alert = None
    if fraud > 0:
        alert = dbc.Alert(
            f" {fraud} high-risk transactions detected!",
            color="danger",
            className="mt-3"
        )

    return html.Div([
        html.H5("Live Transaction Feed"),
        summary_cards,
        alert,
        html.Div([
            dcc.Graph(figure=bar_chart),
            dcc.Graph(figure=pie_chart)
        ], className="mt-4"),

        dash_table.DataTable(
            data=df_live.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_live.columns],
            page_size=10,

            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'center'},

            style_data_conditional=[
                {
                    'if': {'filter_query': '{Prediction} = 1'},
                    'backgroundColor': '#FFCCCC',
                },
                {
                    'if': {'filter_query': '{Risk} = "High"'},
                    'backgroundColor': '#FF6666',
                    'color': 'white'
                }
            ]
        )
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)