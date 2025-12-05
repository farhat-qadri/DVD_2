import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ----------------------------------------------------------------------------------
print("Loading data... this might take a moment.")

try:
    # --- Load Current Application Data ---
    app_cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE', 'DAYS_BIRTH'
    ]
    df_curr = pd.read_csv('application_data.csv', usecols=app_cols)
    
    # Feature Engineering (Current)
    df_curr['Risk_Label'] = df_curr['TARGET'].map({0: 'Safe (0)', 1: 'High Risk (1)'})
    df_curr['AGE_YEARS'] = (df_curr['DAYS_BIRTH'] / -365).astype(int)

    # --- Load Previous Application Data ---
    # We only need ID and Status to calculate refusal history
    prev_cols = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS']
    df_prev = pd.read_csv('previous_application.csv', usecols=prev_cols)

    # --- AGGREGATION STEP ---
    # We must compress the 'previous' many-to-one relationship into 1 row per client
    
    # 1. Create a binary flag for refused loans
    df_prev['is_refused'] = (df_prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    # 2. Group by Client ID
    prev_agg = df_prev.groupby('SK_ID_CURR').agg(
        prev_app_count=('NAME_CONTRACT_STATUS', 'count'),
        prev_refused_count=('is_refused', 'sum')
    ).reset_index()

    # 3. Calculate Refusal Rate (Refusals / Total Apps)
    prev_agg['prev_refusal_rate'] = prev_agg['prev_refused_count'] / prev_agg['prev_app_count']

    # --- MERGE STEP ---
    # Merge history into current application (Left Join)
    # Clients with no history will get NaN, which we fill with 0
    df = df_curr.merge(prev_agg, on='SK_ID_CURR', how='left')
    
    # Fill missing values for clients who have NO previous history
    df['prev_app_count'] = df['prev_app_count'].fillna(0)
    df['prev_refused_count'] = df['prev_refused_count'].fillna(0)
    df['prev_refusal_rate'] = df['prev_refusal_rate'].fillna(0)

    print("Data loaded and merged successfully.")

except FileNotFoundError:
    print("WARNING: Files not found. Generating Dummy Data.")
    # Dummy data generator for testing without files
    df = pd.DataFrame({
        'TARGET': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], 1000),
        'CODE_GENDER': np.random.choice(['M', 'F'], 1000),
        'AMT_INCOME_TOTAL': np.random.normal(150000, 50000, 1000),
        'NAME_EDUCATION_TYPE': np.random.choice(['Higher education', 'Secondary'], 1000),
        'AGE_YEARS': np.random.randint(20, 70, 1000),
        'prev_app_count': np.random.randint(0, 10, 1000),
        'prev_refused_count': np.random.randint(0, 5, 1000)
    })
    df['Risk_Label'] = df['TARGET'].map({0: 'Safe', 1: 'Risk'})
    df['prev_refusal_rate'] = df['prev_refused_count'] / df['prev_app_count']
    df['prev_refusal_rate'] = df['prev_refusal_rate'].fillna(0)

# ----------------------------------------------------------------------------------
# 2. INITIALIZE DASH APP
# ----------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# ----------------------------------------------------------------------------------
# 3. LAYOUT DEFINITION
# ----------------------------------------------------------------------------------
sidebar = html.Div(
    [
        html.H3("Risk Analyst", className="display-6"),
        html.Hr(),
        html.P("Filter the Portfolio", className="lead"),
        
        html.Label("Contract Type"),
        dcc.Dropdown(
            id='filter-contract',
            options=[{'label': i, 'value': i} for i in df['NAME_CONTRACT_TYPE'].unique()],
            value=None,
            placeholder="All Types",
            className="mb-3"
        ),
        html.Label("Gender"),
        dcc.Dropdown(
            id='filter-gender',
            options=[{'label': i, 'value': i} for i in df['CODE_GENDER'].unique()],
            value=None,
            placeholder="All Genders",
            className="mb-3"
        ),
        html.Hr(),
        html.P("Project V1.0", className="text-muted small"),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "18rem", "padding": "2rem 1rem", "background-color": "#f8f9fa"},
)

# Tab 1: Original content (Demographics)
tab1_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Total Apps"), dbc.CardBody(html.H4(id='kpi-total'))], color="primary", inverse=True)),
            dbc.Col(dbc.Card([dbc.CardHeader("Default Rate"), dbc.CardBody(html.H4(id='kpi-rate'))], color="danger", inverse=True)),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph-target-dist'), width=6),
            dbc.Col(dcc.Graph(id='graph-education-risk'), width=6),
        ]),
         dbc.Row([
            dbc.Col(dcc.Graph(id='graph-income-dist'), width=12),
        ])
    ]),
    className="mt-3"
)

# Tab 2: New Content (History Analysis)
tab2_content = dbc.Card(
    dbc.CardBody([
        html.H4("Impact of Credit History on Default Risk", className="mb-3"),
        html.P("Does a history of rejection or frequent borrowing affect current repayment?", className="text-muted"),
        
        dbc.Row([
            # Graph: Previous Rejection Rate vs Current Risk
            dbc.Col(dcc.Graph(id='graph-refusal-impact'), width=6),
            
            # Graph: Number of Previous Loans vs Current Risk
            dbc.Col(dcc.Graph(id='graph-count-impact'), width=6),
        ]),
        
        dbc.Row([
            dbc.Col(
                dbc.Alert(
                    "Insight: High refusal rates in the past often correlate with higher default risk now. "
                    "However, customers with many successful past loans (high count, low refusal) are usually safer.",
                    color="info", className="mt-4"
                )
            )
        ])
    ]),
    className="mt-3"
)

content = html.Div(
    [
        html.H2("Credit Risk Dashboard", className="mb-4"),
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label="Current Profile Analysis", tab_id="tab-1"),
                dbc.Tab(tab2_content, label="History & Behavior Analysis", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ],
    style={"margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem"},
)

app.layout = html.Div([sidebar, content])

# ----------------------------------------------------------------------------------
# 4. CALLBACKS
# ----------------------------------------------------------------------------------
@app.callback(
    [
        Output('kpi-total', 'children'),
        Output('kpi-rate', 'children'),
        Output('graph-target-dist', 'figure'),
        Output('graph-education-risk', 'figure'),
        Output('graph-income-dist', 'figure'),
        Output('graph-refusal-impact', 'figure'),
        Output('graph-count-impact', 'figure')
    ],
    [
        Input('filter-contract', 'value'),
        Input('filter-gender', 'value')
    ]
)
def update_dashboard(contract_filter, gender_filter):
    # Filter Data
    dff = df.copy()
    if contract_filter:
        dff = dff[dff['NAME_CONTRACT_TYPE'] == contract_filter]
    if gender_filter:
        dff = dff[dff['CODE_GENDER'] == gender_filter]

    # KPIs
    kpi_total = f"{len(dff):,.0f}"
    kpi_rate = f"{(dff['TARGET'].mean() * 100):.2f}%"

    # --- TAB 1 CHARTS ---
    # 1. Pie
    fig_target = px.pie(dff, names='Risk_Label', title='Risk Distribution', color='Risk_Label',
                        color_discrete_map={'Safe (0)': '#2ecc71', 'High Risk (1)': '#e74c3c'}, hole=0.4)
    
    # 2. Education Bar
    edu_risk = dff.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index().sort_values('TARGET')
    fig_edu = px.bar(edu_risk, x='TARGET', y='NAME_EDUCATION_TYPE', orientation='h',
                     title='Default Rate by Education', color='TARGET', color_continuous_scale='Reds')
    fig_edu.update_layout(xaxis_tickformat=".1%")

    # 3. Income Box (No outliers)
    cutoff = dff['AMT_INCOME_TOTAL'].quantile(0.95)
    fig_inc = px.box(dff[dff['AMT_INCOME_TOTAL'] < cutoff], x='AMT_INCOME_TOTAL', y='Risk_Label', 
                     color='Risk_Label', title='Income vs Risk', 
                     color_discrete_map={'Safe (0)': '#2ecc71', 'High Risk (1)': '#e74c3c'})

    # --- TAB 2 CHARTS (NEW) ---
    
    # 4. Impact of Previous Refusals
    # Binning the refusal rate to make it readable (0%, 0-50%, 50-100%)
    dff['Refusal_Bin'] = pd.cut(dff['prev_refusal_rate'], bins=[-0.1, 0, 0.5, 1.0], labels=['No Refusals', 'Some Refusals', 'Mostly Refused'])
    refusal_risk = dff.groupby('Refusal_Bin', observed=False)['TARGET'].mean().reset_index()
    
    fig_refusal = px.bar(
        refusal_risk, x='Refusal_Bin', y='TARGET', 
        title='Does Past Rejection Predict Future Default?',
        labels={'TARGET': 'Current Default Rate', 'Refusal_Bin': 'Past Refusal History'},
        color='TARGET', color_continuous_scale='Reds'
    )
    fig_refusal.update_layout(yaxis_tickformat=".1%")

    # 5. Impact of Previous Loan Count
    # Binning count to handle outliers (0, 1-3, 4-7, 8+)
    dff['Count_Bin'] = pd.cut(dff['prev_app_count'], bins=[-1, 0, 3, 7, 100], labels=['0', '1-3', '4-7', '8+'])
    count_risk = dff.groupby('Count_Bin', observed=False)['TARGET'].mean().reset_index()
    
    fig_count = px.bar(
        count_risk, x='Count_Bin', y='TARGET',
        title='Does Loan Frequency Predict Default?',
        labels={'TARGET': 'Current Default Rate', 'Count_Bin': 'Number of Previous Applications'},
        color='TARGET', color_continuous_scale='Blues'
    )
    fig_count.update_layout(yaxis_tickformat=".1%")

    return kpi_total, kpi_rate, fig_target, fig_edu, fig_inc, fig_refusal, fig_count

# ----------------------------------------------------------------------------------
# 5. RUN SERVER
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)