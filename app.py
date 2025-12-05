import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ----------------------------------------------------------------------------------
# We load the data once when the server starts.
# For V0, we focus on the main application data.
try:
    # Loading only necessary columns for V0 to save memory
    cols_to_use = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'NAME_EDUCATION_TYPE', 
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'AGE_YEARS' # We will create AGE_YEARS
    ]
    
    # Read CSV (Assuming file is in the same directory)
    # Using 'low_memory=False' to silence mixed type warnings on large files
    df = pd.read_csv('application_data.csv')
    
    # Data Cleaning for V0
    # Create a readable Target label
    df['Risk_Label'] = df['TARGET'].map({0: 'Safe (0)', 1: 'High Risk (1)'})
    
    # Convert DAYS_BIRTH to Years (It is negative in the raw file)
    df['AGE_YEARS'] = (df['DAYS_BIRTH'] / -365).astype(int)

except FileNotFoundError:
    # Fallback data for demonstration if file is missing
    print("WARNING: 'application_data.csv' not found. Using dummy data.")
    df = pd.DataFrame({
        'TARGET': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], 1000),
        'CODE_GENDER': np.random.choice(['M', 'F'], 1000),
        'AMT_INCOME_TOTAL': np.random.normal(150000, 50000, 1000),
        'NAME_EDUCATION_TYPE': np.random.choice(['Higher education', 'Secondary'], 1000),
        'AGE_YEARS': np.random.randint(20, 70, 1000)
    })
    df['Risk_Label'] = df['TARGET'].map({0: 'Safe', 1: 'Risk'})

# ----------------------------------------------------------------------------------
# 2. INITIALIZE DASH APP
# ----------------------------------------------------------------------------------
# Using the FLATLY theme for a clean, corporate financial look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Expose server for deployment

# ----------------------------------------------------------------------------------
# 3. LAYOUT DEFINITION
# ----------------------------------------------------------------------------------
# Sidebar styling
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Main content styling
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("Risk Analyst", className="display-6"),
        html.Hr(),
        html.P("Filter the Portfolio", className="lead"),
        
        # Filter 1: Contract Type
        html.Label("Contract Type"),
        dcc.Dropdown(
            id='filter-contract',
            options=[{'label': i, 'value': i} for i in df['NAME_CONTRACT_TYPE'].unique()],
            value=None,  # None means Select All
            placeholder="All Types",
            className="mb-3"
        ),

        # Filter 2: Gender
        html.Label("Gender"),
        dcc.Dropdown(
            id='filter-gender',
            options=[{'label': i, 'value': i} for i in df['CODE_GENDER'].unique()],
            value=None,
            placeholder="All Genders",
            className="mb-3"
        ),
        
        html.Hr(),
        html.P("Project V0.1", className="text-muted small"),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    [
        html.H2("Portfolio Overview Dashboard", className="mb-4"),
        
        # Row 1: KPI Cards
        dbc.Row(
            [
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Total Applications"),
                    dbc.CardBody(html.H4(id='kpi-total-apps', className="card-title"))
                ], color="primary", inverse=True), width=3),
                
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Default Rate (%)"),
                    dbc.CardBody(html.H4(id='kpi-default-rate', className="card-title"))
                ], color="danger", inverse=True), width=3),
                
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Avg Income (Safe)"),
                    dbc.CardBody(html.H4(id='kpi-avg-income-safe', className="card-title"))
                ], color="success", inverse=True), width=3),
                 dbc.Col(dbc.Card([
                    dbc.CardHeader("Avg Income (Risk)"),
                    dbc.CardBody(html.H4(id='kpi-avg-income-risk', className="card-title"))
                ], color="warning", inverse=True), width=3),
            ],
            className="mb-4"
        ),

        # Row 2: Main Charts
        dbc.Row(
            [
                # Chart 1: Target Variable Distribution
                dbc.Col(
                    dcc.Loading(children=[dcc.Graph(id='graph-target-dist')]), 
                    width=6
                ),
                # Chart 2: Default Rate by Education
                dbc.Col(
                    dcc.Loading(children=[dcc.Graph(id='graph-education-risk')]), 
                    width=6
                ),
            ],
            className="mb-4"
        ),

        # Row 3: Continuous Variable Analysis
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(children=[dcc.Graph(id='graph-income-dist')]),
                    width=12
                )
            ]
        )
    ],
    style=CONTENT_STYLE,
)

app.layout = html.Div([sidebar, content])

# ----------------------------------------------------------------------------------
# 4. CALLBACKS (LOGIC)
# ----------------------------------------------------------------------------------
@app.callback(
    [
        Output('kpi-total-apps', 'children'),
        Output('kpi-default-rate', 'children'),
        Output('kpi-avg-income-safe', 'children'),
        Output('kpi-avg-income-risk', 'children'),
        Output('graph-target-dist', 'figure'),
        Output('graph-education-risk', 'figure'),
        Output('graph-income-dist', 'figure')
    ],
    [
        Input('filter-contract', 'value'),
        Input('filter-gender', 'value')
    ]
)
def update_dashboard(contract_filter, gender_filter):
    # 1. Filter Data
    dff = df.copy()
    if contract_filter:
        dff = dff[dff['NAME_CONTRACT_TYPE'] == contract_filter]
    if gender_filter:
        dff = dff[dff['CODE_GENDER'] == gender_filter]

    # 2. Calculate KPIs
    total_apps = len(dff)
    default_rate = (dff['TARGET'].mean() * 100)
    avg_income_safe = dff[dff['TARGET'] == 0]['AMT_INCOME_TOTAL'].mean()
    avg_income_risk = dff[dff['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()

    # Format KPIs
    kpi_total = f"{total_apps:,.0f}"
    kpi_rate = f"{default_rate:.2f}%"
    kpi_inc_safe = f"${avg_income_safe:,.0f}" if not pd.isna(avg_income_safe) else "N/A"
    kpi_inc_risk = f"${avg_income_risk:,.0f}" if not pd.isna(avg_income_risk) else "N/A"

    # 3. Create Visualizations
    
    # Graph 1: Target Distribution (Donut Chart)
    fig_target = px.pie(
        dff, names='Risk_Label', title='Portfolio Risk Distribution',
        color='Risk_Label',
        color_discrete_map={'Safe (0)': '#2ecc71', 'High Risk (1)': '#e74c3c'},
        hole=0.4
    )

    # Graph 2: Risk by Education (Bar Chart)
    # We calculate the mean TARGET per education level
    risk_by_edu = dff.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index()
    risk_by_edu = risk_by_edu.sort_values('TARGET', ascending=False)
    
    fig_edu = px.bar(
        risk_by_edu, x='TARGET', y='NAME_EDUCATION_TYPE', orientation='h',
        title='Default Probability by Education Level',
        labels={'TARGET': 'Default Rate', 'NAME_EDUCATION_TYPE': 'Education'},
        color='TARGET', color_continuous_scale='Reds'
    )
    fig_edu.update_layout(xaxis_tickformat=".1%")

    # Graph 3: Income Distribution Box Plot
    # Limiting income to avoid outliers skewing the view (Visual aid only)
    # Using 95th percentile for better scaling
    cutoff = dff['AMT_INCOME_TOTAL'].quantile(0.95)
    dff_viz = dff[dff['AMT_INCOME_TOTAL'] < cutoff]
    
    fig_inc = px.box(
        dff_viz, x='AMT_INCOME_TOTAL', y='Risk_Label', color='Risk_Label',
        title='Income Distribution: Safe vs Risk (Outliers Removed)',
        color_discrete_map={'Safe (0)': '#2ecc71', 'High Risk (1)': '#e74c3c'}
    )

    return kpi_total, kpi_rate, kpi_inc_safe, kpi_inc_risk, fig_target, fig_edu, fig_inc

# ----------------------------------------------------------------------------------
# 5. RUN SERVER
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)