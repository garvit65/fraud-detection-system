import base64
import io
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
from backend.prediction import predict_fraud
from backend.model_metrics import get_model_metrics
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Initialize app with custom theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Fraud Detection System"

# Load model metrics on startup
MODEL_METRICS = get_model_metrics()

# Custom CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card-metric {
                background: white;
                border-radius: 12px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .card-metric:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }
            .metric-label {
                font-size: 0.9rem;
                color: #6c757d;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .card-fraud {
                border-left-color: #dc3545;
            }
            .card-fraud .metric-value {
                color: #dc3545;
            }
            .card-success {
                border-left-color: #28a745;
            }
            .card-success .metric-value {
                color: #28a745;
            }
            .upload-box {
                border: 2px dashed #667eea;
                border-radius: 12px;
                background-color: #f0f4ff;
                padding: 40px !important;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-box:hover {
                border-color: #764ba2;
                background-color: #e8ecff;
            }
            .alert-custom {
                border-radius: 8px;
                border-left: 4px solid;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            }
            .title-main {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .subtitle-main {
                color: #6c757d;
                font-size: 1rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Build the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Financial Fraud Detection System", className="title-main"),
            html.P("Real-time transaction monitoring", className="subtitle-main")
        ], width=12, className="mb-4 mt-4")
    ]),
    
    # Main Tabs
    dcc.Tabs(id="main-tabs", value="tab-upload", children=[
        # TAB 1: Upload Data
        dcc.Tab(label="Upload Data", value="tab-upload", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Upload Transaction Dataset", className="mb-4"),
                    html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.Br(),
                                html.B("Drag and drop CSV file here"),
                                html.Br(),
                                html.Span("or click to select", style={'fontSize': '0.9rem', 'color': '#6c757d'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '180px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '12px',
                                'textAlign': 'center',
                                'display': 'flex',
                                'flexDirection': 'column',
                                'justifyContent': 'center',
                                'alignItems': 'center',
                                'background': '#f0f4ff',
                                'borderColor': '#667eea',
                                'cursor': 'pointer',
                                'transition': 'all 0.3s'
                            },
                            multiple=False
                        )
                    ], className="upload-box"),
                    
                    dcc.Loading(
                        id="loading-upload",
                        type="default",
                        children=[
                            html.Div(id='output-data-upload', style={'marginTop': '20px'})
                        ]
                    )
                ])
            ], className="mt-3")
        ]),
        
        # TAB 2: Live Monitoring
        dcc.Tab(label="Live Monitoring", value="tab-live", children=[
            dbc.Card([
                dbc.CardBody([
                    dcc.Store(id='stored-data'),
                    dcc.Store(id='predicted-data'),
                    dcc.Interval(id='interval-component', interval=500, n_intervals=0),
                    
                    html.Div(id='streaming-title'),
                    html.Div(id='streaming-cards', style={'marginTop': '10px'}),
                    html.Div(id='streaming-alert'),
                    html.Div(id='streaming-charts', style={'marginTop': '20px'}),
                    html.Div(id='streaming-table-header', style={'marginTop': '20px'}),
                    html.Div(id='streaming-table')
                ])
            ], className="mt-3")
        ]),
        
        # TAB 3: Model Metrics
        dcc.Tab(label="Model Metrics", value="tab-metrics", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Training Data Overview", className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div("Total Samples", className="metric-label"),
                                    html.Div(f"{MODEL_METRICS['total_samples']:,}", className="metric-value"),
                                ])
                            ], className="card-metric")
                        ], md=3, sm=6, className="mb-4"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div("Fraud Cases", className="metric-label"),
                                    html.Div(f"{MODEL_METRICS['fraud_cases']}", className="metric-value card-fraud"),
                                ])
                            ], className="card-metric card-fraud")
                        ], md=3, sm=6, className="mb-4"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div("Normal Cases", className="metric-label"),
                                    html.Div(f"{MODEL_METRICS['normal_cases']:,}", className="metric-value card-success"),
                                ])
                            ], className="card-metric card-success")
                        ], md=3, sm=6, className="mb-4"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div("Fraud Rate", className="metric-label"),
                                    html.Div(f"{MODEL_METRICS['fraud_percentage']:.2f}%", className="metric-value card-fraud"),
                                ])
                            ], className="card-metric card-fraud")
                        ], md=3, sm=6, className="mb-4"),
                    ]),
                    
                    # Info Box
                    dbc.Alert([
                        html.Strong("Model Information"),
                        html.Br(),
                        "Algorithm: Random Forest Classifier",
                        html.Br(),
                        "Training Data: PaySim Financial Simulation Dataset",
                        html.Br(),
                        "Features: Amount, Old Balance, New Balance, Balance Difference",
                        html.Br(),
                        "Prediction Threshold: 0.2 (tuned for recall)"
                    ], color="info", className="mt-4 alert-custom")
                ])
            ], className="mt-3")
        ]),
    ], style={"marginBottom": "30px"}),
    
], fluid=True, style={'backgroundColor': '#f8f9fa', 'paddingBottom': '30px'})


# Callback: Store uploaded data
@app.callback(
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def store_data(contents, filename):
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                return df.to_dict('records')
        except Exception as e:
            return None
    return None


# Callback: Cache predictions to avoid recalculating every interval
@app.callback(
    Output('predicted-data', 'data'),
    Input('stored-data', 'data')
)
def cache_predictions(data):
    if data is None:
        return None
    try:
        df = pd.DataFrame(data)
        df = predict_fraud(df)
        return df.to_dict('records')
    except:
        return None


# Callback: Parse and display uploaded file
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_contents(contents, filename):
    if contents is None:
        return html.Div()
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = predict_fraud(df)
            
            # Validate required columns exist
            if 'Prediction' not in df.columns or 'Probability' not in df.columns or 'Risk' not in df.columns:
                return dbc.Alert(
                    "❌ Error: Predictions missing required columns. Please try uploading again.",
                    color="danger",
                    className="mt-3"
                )

            # Summary calculations
            total = len(df)
            fraud = df['Prediction'].sum()
            fraud_rate = round((fraud / total) * 100, 2)

            # Summary cards
            summary_cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div("Total Transactions", className="metric-label"),
                            html.Div(f"{total:,}", className="metric-value"),
                        ])
                    ], className="card-metric")
                ], md=4, sm=6, className="mb-3"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div("Fraud Detected", className="metric-label"),
                            html.Div(f"{fraud}", className="metric-value", style={'color': '#dc3545'}),
                        ])
                    ], className="card-metric card-fraud")
                ], md=4, sm=6, className="mb-3"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div("Fraud Rate", className="metric-label"),
                            html.Div(f"{fraud_rate}%", className="metric-value", style={'color': '#dc3545'}),
                        ])
                    ], className="card-metric card-fraud")
                ], md=4, sm=6, className="mb-3"),
            ])

            # Alert
            alert = None
            if fraud > 0:
                alert = dbc.Alert(
                    [html.Strong(f"Warning: "), f"{fraud} fraudulent transaction(s) detected!"],
                    color="danger",
                    className="mt-4 alert-custom"
                )

            # Data table with better styling
            table = dash_table.DataTable(
                data=df.head(20).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto', 'marginTop': '20px', 'borderRadius': '8px', 'overflow': 'hidden'},
                style_header={
                    'backgroundColor': '#667eea',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'padding': '12px'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '12px',
                    'fontSize': '0.9rem'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Prediction} = 1'},
                        'backgroundColor': '#ffe6e6',
                        'color': '#721c24',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'filter_query': '{Risk} = "High"'},
                        'backgroundColor': '#ff6b6b',
                        'color': 'white'
                    },
                    {
                        'if': {'filter_query': '{Risk} = "Medium"'},
                        'backgroundColor': '#ffd166',
                        'color': '#333'
                    }
                ]
            )

            return html.Div([
                html.H5(f"File: {filename}", className="mt-3 mb-2"),
                html.P(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}", style={'color': '#6c757d'}),
                summary_cards,
                alert,
                html.H6("Preview (First 20 rows)", className="mt-4 mb-3"),
                table
            ])

    except Exception as e:
        return dbc.Alert(
            [html.Strong("❌ Error processing file: "), str(e)],
            color="danger",
            className="mt-3"
        )

    return dbc.Alert("❌ Unsupported file format. Please upload a CSV file.", color="danger", className="mt-3")




# Callback: Update streaming title
@app.callback(
    Output('streaming-title', 'children'),
    Input('predicted-data', 'data')
)
def update_title(data):
    if data is None:
        return html.Div(
            html.P("📤 Upload a CSV file from the 'Upload Data' tab to start live monitoring", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'paddingTop': '40px'}),
            style={'textAlign': 'center'}
        )
    return html.H5("Live Transaction Stream", className="mb-4")


# Callback: Update metric cards
@app.callback(
    Output('streaming-cards', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('predicted-data', 'data')
)
def update_cards(n, data):
    if data is None:
        return html.Div()
    
    try:
        df_live = pd.DataFrame(data)
        
        # Check if Prediction column exists
        if 'Prediction' not in df_live.columns:
            return html.Div()
        
        step_size = 20
        df_live = df_live.head((n + 1) * step_size)
        
        total = len(df_live)
        fraud = df_live['Prediction'].sum()
        fraud_rate = round((fraud / total) * 100, 2) if total > 0 else 0
    except Exception as e:
        return html.Div()
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Streaming Transactions", className="metric-label"),
                    html.Div(f"{total:,}", className="metric-value"),
                ])
            ], className="card-metric")
        ], md=4, sm=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Fraud Detected", className="metric-label"),
                    html.Div(f"{fraud}", className="metric-value", style={'color': '#dc3545'}),
                ])
            ], className="card-metric card-fraud")
        ], md=4, sm=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Detection Rate", className="metric-label"),
                    html.Div(f"{fraud_rate}%", className="metric-value", style={'color': '#dc3545'}),
                ])
            ], className="card-metric card-fraud")
        ], md=4, sm=6, className="mb-3"),
    ])


# Callback: Update alert
@app.callback(
    Output('streaming-alert', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('predicted-data', 'data')
)
def update_alert(n, data):
    if data is None:
        return html.Div()
    
    try:
        df_live = pd.DataFrame(data)
        
        # Check if Prediction column exists
        if 'Prediction' not in df_live.columns:
            return html.Div()
        
        step_size = 20
        df_live = df_live.head((n + 1) * step_size)
        fraud = df_live['Prediction'].sum()
    except Exception as e:
        return html.Div()
    
    if fraud > 0:
        return dbc.Alert(
            [html.Strong("🚨 Alert: "), f"{fraud} high-risk transaction(s) in live stream!"],
            color="danger",
            className="mt-3 alert-custom"
        )
    return html.Div()


# Callback: Update charts
@app.callback(
    Output('streaming-charts', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('predicted-data', 'data')
)
def update_charts(n, data):
    if data is None:
        return html.Div()
    
    try:
        df_live = pd.DataFrame(data)
        
        # Check if required columns exist
        if 'Prediction' not in df_live.columns or 'Risk' not in df_live.columns:
            return html.Div()
        
        step_size = 20
        df_live = df_live.head((n + 1) * step_size)
        
        # Bar chart
        fraud_counts = df_live['Prediction'].value_counts().reset_index()
        fraud_counts.columns = ['Prediction', 'Count']
        fraud_counts['Label'] = fraud_counts['Prediction'].map({0: 'Normal', 1: 'Fraud'})
        
        bar_chart = px.bar(
            fraud_counts,
            x='Label',
            y='Count',
            title="Fraud vs Normal Transactions",
            color='Label',
            color_discrete_map={'Normal': '#28a745', 'Fraud': '#dc3545'},
            text='Count'
        )
        bar_chart.update_traces(textposition='auto')
        bar_chart.update_layout(showlegend=False, hovermode='x unified', template='plotly_white', height=400)
        
        # Pie chart
        risk_counts = df_live['Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        pie_chart = px.pie(
            risk_counts,
            names='Risk',
            values='Count',
            title="Risk Distribution",
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        pie_chart.update_layout(height=400)
        
        return html.Div([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=bar_chart)], md=6),
                dbc.Col([dcc.Graph(figure=pie_chart)], md=6),
            ], className="mt-4")
        ])
    except Exception as e:
        return html.Div()


# Callback: Update table header
@app.callback(
    Output('streaming-table-header', 'children'),
    Input('predicted-data', 'data')
)
def update_table_header(data):
    if data is None:
        return html.Div()
    return html.H6("Transaction Details", className="mt-4 mb-3")


# Callback: Update table
@app.callback(
    Output('streaming-table', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('predicted-data', 'data')
)
def update_table(n, data):
    if data is None:
        return html.Div()
    
    try:
        df_live = pd.DataFrame(data)
        
        # Check if required columns exist
        if 'Prediction' not in df_live.columns or 'Risk' not in df_live.columns:
            return html.Div()
        
        step_size = 20
        df_live = df_live.head((n + 1) * step_size)
        
        return dash_table.DataTable(
            data=df_live.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_live.columns],
            page_size=10,
            style_table={'overflowX': 'auto', 'marginTop': '20px', 'borderRadius': '8px', 'overflow': 'hidden'},
            style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '12px'},
            style_cell={'textAlign': 'center', 'padding': '12px', 'fontSize': '0.9rem'},
            style_data_conditional=[
                {'if': {'filter_query': '{Prediction} = 1'}, 'backgroundColor': '#ffe6e6', 'color': '#721c24', 'fontWeight': 'bold'},
                {'if': {'filter_query': '{Risk} = "High"'}, 'backgroundColor': '#ff6b6b', 'color': 'white'},
                {'if': {'filter_query': '{Risk} = "Medium"'}, 'backgroundColor': '#ffd166', 'color': '#333'}
            ]
        )
    except Exception as e:
        return html.Div()




# Run server
if __name__ == '__main__':
    app.run(debug=True)
