import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ----------------------------------------------------------------------------------
print("Loading data...")

try:
    # --- Load Current Application Data ---
    app_cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE'
    ]
    df_curr = pd.read_csv('application_data.csv', usecols=app_cols)
    
    # Feature Engineering
    df_curr['Risk_Label'] = df_curr['TARGET'].map({0: 'Safe (0)', 1: 'High Risk (1)'})
    
    # --- Load Previous Application Data ---
    prev_cols = [
        'SK_ID_CURR', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 
        'NAME_PRODUCT_TYPE'
    ]
    df_prev = pd.read_csv('previous_application.csv', usecols=prev_cols)

    # Optimization: Filter previous data to only relevant clients
    df_prev = df_prev[df_prev['SK_ID_CURR'].isin(df_curr['SK_ID_CURR'])]

    # --- AGGREGATION FOR TAB 2 ---
    df_prev['is_refused'] = (df_prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    # Get Last Product Type
    df_prev_sorted = df_prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False])
    last_prod = df_prev_sorted.drop_duplicates(subset=['SK_ID_CURR'], keep='first')[['SK_ID_CURR', 'NAME_PRODUCT_TYPE']]
    last_prod = last_prod.rename(columns={'NAME_PRODUCT_TYPE': 'LAST_PRODUCT_TYPE'})

    # Aggregates
    prev_agg = df_prev.groupby('SK_ID_CURR').agg(
        prev_app_count=('NAME_CONTRACT_STATUS', 'count'),
        prev_refused_count=('is_refused', 'sum')
    ).reset_index()
    prev_agg['prev_refusal_rate'] = prev_agg['prev_refused_count'] / prev_agg['prev_app_count']

    # --- MERGE ---
    df = df_curr.merge(prev_agg, on='SK_ID_CURR', how='left')
    df = df.merge(last_prod, on='SK_ID_CURR', how='left')
    
    # Fill NAs
    df[['prev_app_count', 'prev_refusal_rate']] = df[['prev_app_count', 'prev_refusal_rate']].fillna(0)
    df['LAST_PRODUCT_TYPE'] = df['LAST_PRODUCT_TYPE'].fillna('No History')

    print("Data loaded successfully.")

except FileNotFoundError:
    print("WARNING: CSV Files not found. Using Dummy Data.")
    df = pd.DataFrame() # Dummy placeholder

# ----------------------------------------------------------------------------------
# 2. LOAD HTML FILE CONTENT
# ----------------------------------------------------------------------------------
timeline_html_content = ""
try:
    with open('stacked_boxes_timeline_auto.html', 'r', encoding='utf-8') as f:
        timeline_html_content = f.read()
    print("HTML Timeline file loaded successfully.")
except FileNotFoundError:
    timeline_html_content = "<h3>Error: stacked_boxes_timeline_auto.html not found in folder.</h3>"

# ----------------------------------------------------------------------------------
# 3. INITIALIZE DASH APP
# ----------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# ----------------------------------------------------------------------------------
# 4. LAYOUT
# ----------------------------------------------------------------------------------
sidebar = html.Div(
    [
        html.H3("Risk Analyst", className="display-6"),
        html.Hr(),
        html.P("Filter the Portfolio", className="lead"),
        html.Label("Contract Type"),
        dcc.Dropdown(
            id='filter-contract',
            options=[{'label': i, 'value': i} for i in df['NAME_CONTRACT_TYPE'].unique()] if not df.empty else [],
            value=None, placeholder="All Types", className="mb-3"
        ),
        html.Label("Gender"),
        dcc.Dropdown(
            id='filter-gender',
            options=[{'label': i, 'value': i} for i in df['CODE_GENDER'].unique()] if not df.empty else [],
            value=None, placeholder="All Genders", className="mb-3"
        ),
        html.Hr(),
        html.P("Project V4.0 (HTML Embed)", className="text-muted small"),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "18rem", "padding": "2rem 1rem", "background-color": "#f8f9fa"},
)

# Tab 1 Content
tab1_content = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Total Apps"), dbc.CardBody(html.H4(id='kpi-total'))], color="primary", inverse=True)),
        dbc.Col(dbc.Card([dbc.CardHeader("Default Rate"), dbc.CardBody(html.H4(id='kpi-rate'))], color="danger", inverse=True)),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='graph-target-dist'), width=6),
        dbc.Col(dcc.Graph(id='graph-education-risk'), width=6),
    ]),
     dbc.Row([dbc.Col(dcc.Graph(id='graph-income-dist'), width=12)])
]), className="mt-3")

# Tab 2 Content
tab2_content = dbc.Card(dbc.CardBody([
    html.H4("History Patterns", className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='graph-refusal-impact'), width=6),
        dbc.Col(dcc.Graph(id='graph-count-impact'), width=6),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='graph-product-type'), width=12)])
]), className="mt-3")

# --- Tab 3: Timeline Inspector (DIRECT HTML EMBED) ---
tab3_content = dbc.Card(dbc.CardBody([
    html.H4("Application History Timeline", className="mb-3"),
    html.P("Loading external visualization from 'stacked_boxes_timeline_auto.html'...", className="text-muted"),
    
    # We use an Iframe to render the standalone HTML file exactly as it is
    html.Div(
        html.Iframe(
            srcDoc=timeline_html_content,
            style={"width": "100%", "height": "800px", "border": "none"}
        ),
        style={"border": "1px solid #ddd", "padding": "5px"}
    )
]), className="mt-3")

content = html.Div(
    [
        html.H2("Credit Risk Dashboard", className="mb-4"),
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label="Current Profile", tab_id="tab-1"),
                dbc.Tab(tab2_content, label="History Analysis", tab_id="tab-2"),
                dbc.Tab(tab3_content, label="Timeline Inspector (Static)", tab_id="tab-3"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ],
    style={"margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem"},
)

app.layout = html.Div([sidebar, content])

# ----------------------------------------------------------------------------------
# 5. CALLBACKS
# ----------------------------------------------------------------------------------
@app.callback(
    [
        Output('kpi-total', 'children'),
        Output('kpi-rate', 'children'),
        Output('graph-target-dist', 'figure'),
        Output('graph-education-risk', 'figure'),
        Output('graph-income-dist', 'figure'),
        Output('graph-refusal-impact', 'figure'),
        Output('graph-count-impact', 'figure'),
        Output('graph-product-type', 'figure')
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

    # Tab 1 Charts
    fig_target = px.pie(dff, names='Risk_Label', title='Risk Distribution', color='Risk_Label',
                        color_discrete_map={'Safe (0)': '#2ecc71', 'High Risk (1)': '#e74c3c'}, hole=0.4)
    
    edu_risk = dff.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index().sort_values('TARGET')
    fig_edu = px.bar(edu_risk, x='TARGET', y='NAME_EDUCATION_TYPE', orientation='h',
                     title='Default Rate by Education', color='TARGET', color_continuous_scale='Reds')
    
    cutoff = dff['AMT_INCOME_TOTAL'].quantile(0.95)
    fig_inc = px.box(dff[dff['AMT_INCOME_TOTAL'] < cutoff], x='AMT_INCOME_TOTAL', y='Risk_Label', 
                     color='Risk_Label', title='Income vs Risk', 
                     color_discrete_map={'Safe (0)': '#2ecc71', 'High Risk (1)': '#e74c3c'})

    # Tab 2 Charts
    dff['Refusal_Bin'] = pd.cut(dff['prev_refusal_rate'], bins=[-0.1, 0, 0.5, 1.0], labels=['No Refusals', 'Some', 'Mostly Refused'])
    refusal_risk = dff.groupby('Refusal_Bin', observed=False)['TARGET'].mean().reset_index()
    fig_refusal = px.bar(refusal_risk, x='Refusal_Bin', y='TARGET', title='Risk by Past Refusal Rate',
                         color='TARGET', color_continuous_scale='Reds')

    dff['Count_Bin'] = pd.cut(dff['prev_app_count'], bins=[-1, 0, 3, 7, 100], labels=['0', '1-3', '4-7', '8+'])
    count_risk = dff.groupby('Count_Bin', observed=False)['TARGET'].mean().reset_index()
    fig_count = px.bar(count_risk, x='Count_Bin', y='TARGET', title='Risk by Loan Count',
                       color='TARGET', color_continuous_scale='Blues')

    top_prods = dff['LAST_PRODUCT_TYPE'].value_counts().head(5).index
    prod_risk = dff[dff['LAST_PRODUCT_TYPE'].isin(top_prods)].groupby('LAST_PRODUCT_TYPE')['TARGET'].mean().reset_index().sort_values('TARGET', ascending=False)
    fig_prod = px.bar(prod_risk, x='LAST_PRODUCT_TYPE', y='TARGET', title='Risk by Last Product Type',
                      color='TARGET', color_continuous_scale='Viridis')

    return kpi_total, kpi_rate, fig_target, fig_edu, fig_inc, fig_refusal, fig_count, fig_prod

# ----------------------------------------------------------------------------------
# 6. RUN SERVER
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)