import dash
from dash import dcc, html
from dash_extensions.enrich import Output, Input, State, ALL, MATCH, DashProxy, MultiplexerTransform  # conda install
from analyze_portfolio import PortfolioAnalyzer
import dash_bootstrap_components as dbc  # conda install
import yfinance as yf
import pandas as pd
import datetime
import json
import os
import joblib
from pathlib import Path
import xgboost as xgb
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR.parent / "models"

app = DashProxy(transforms=[MultiplexerTransform()])  # initializes app with multiplexer transforms
config = {
    'displayModeBar': False,
    'displaylogo': False,
    'scrollZoom': False,
}  # sets application scroll/display configs to False, want to work as a dashboard not a massive webpage

# Load insider trade dataset once for tooltips; fall back gracefully if missing.
try:
    insider_df = pd.read_csv(
        DATA_DIR / 'insider_trades_with_returns.csv',
        parse_dates=['transaction_date', 'trade_date'],
        low_memory=False
    )
    insider_df['ticker'] = insider_df['ticker'].astype(str).str.upper()
except Exception as e:  # pragma: no cover - diagnostic logging only
    insider_df = None
    print(f"Could not load insider trade data: {e}")

# Load precomputed backtest results for SPY comparison.
try:
    backtest_df = pd.read_csv(
        DATA_DIR / 'backtest_results.csv',
        parse_dates=['date'],
        low_memory=False
    )
except Exception as e:  # pragma: no cover - diagnostic logging only
    backtest_df = None
    print(f"Could not load backtest results: {e}")

#autocomplete_ticker: TickerAutofill = TickerAutofill()

# The layout of the dashboard is an arbitrary number of input fields,
# each takes the stock ticker as input and the dollar amount to invest
app.layout = html.Div([
    html.Div([
        html.Div(style={'height': '5px'}),
        html.Div([
            html.Div([
                html.H1('Stock Portfolio Simulator', style={'margin-left': '10px', 'color': 'navy'}),  # main app header
                html.H4('Please specify your stock portfolio below', style={'margin-left': '20px', 'color': 'black'}),
            ], style={'width': '80%', 'display': 'inline-block'}),
            html.Div([
                html.Div(style={'height': '10px'}),
                html.Div([
                    html.Div('Number of stocks: ', className='stock-number-label'),
                    # label and toggle for controlling the number of stocks in a portfolio
                    dcc.Input(id='stock-count', type='number', value=1,
                              min='1', max='100', className='stock-number-input'),
                ], className='submit-input-container'),
                html.Div([
                    html.Div('Start date: ', className='start-date-label'),
                    # calendar dropdown for selecting portfolio date of beginning
                    dbc.Input(id='start-date', type='date',
                              value=(datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1)).strftime(
                                  '%Y-%m-%d'), className='start-date-input'),
                ], className='submit-input-container'),  # final submit button to run analytics
                html.Div([
                    html.Button('Submit', id='submit-button', n_clicks=0, className='submit-button'),
                                # style={'background': 'darkgrey', 'border-radius': '16px', 'border': '1px'}),
                ], className='submit-input-container'),
                html.Div([
                    html.Button('Try the model', id='try-model-button', n_clicks=0, className='submit-button')
                ])
            ], className='submit-container'),
        ], style={'width': '100%', 'display': 'flex'}),
        html.Div(style={'height': '15px'}),
        # div for each portfolio element, containing a unique ticker, amount purchased, and error display
        html.Div([
            html.Div([
                html.H3('Stock Ticker'),  # ticker input
                html.Div([], id='tickers')
            ], style={'width': '15%'}, className='input-column'),
            html.Div([
                html.H3('Dollar Amount ($)'),  # amount toggle
                html.Div([], id='amounts')
            ], style={'width': '20%'}, className='input-column'),
            html.Div([
                html.H3('Error Status'),  # error handler display
                html.Div([], id='errors')
            ], style={'width': '15%'}, className='input-column'),
            html.Div([
                html.H3('Info'),  # info display
                html.Div([], id='info_buttons')]
                , style={'width': '25%'}, className='input-column'),
            html.Div([
                html.H3('​'),
                html.Div([], id='delete_buttons')  # delete button for each stock in case of mistake
            ], style={'width': '3%', 'padding': '0px 20px 20px 0px'}, className='input-column'),
        ], style={'width': '100%', 'display': 'flex'}),
        html.Div(style={'height': '20px'}),
        html.Div(style={'height': '200px'}),
        html.Div([html.H5('Powered by Yahoo! Finance', style={'height': '10px', 'padding': '300px 800px 5px 800px'}),
                  html.Img(src=app.get_asset_url('yahoo.png'),
                           style={'height': '25px', 'display': 'block', 'margin-left': 'auto',
                                  'margin-right': 'auto'}), ],
                 # 'padding': '1px 750px 50px 750px'
                 style={'height': '200px', 'padding': '0px 0px 250px 0px'})
    ], style={'width': '100%', 'margin': '1px 1px 1px 1px', 'background-color': 'white', 'color': 'black'},
        id='portfolio-input'),
    html.Div([  # analysis portion of the dashboard, able to be toggeled on/off with submit button
        html.Div([
            dcc.Graph(id='portfolio-plot',
                      style={'height': '100%', 'width': '50%', 'background-color': 'white', 'color': 'black'},
                      config=config),  # portfolio plot element for overal portfolio performance
            dcc.Graph(id='individual-stock-plot',
                      style={'height': '100%', 'width': '50%', 'background-color': 'white', 'color': 'black'},
                      config=config),  # overlaid plot element for comparison of individual stock performance
        ], style={'width': '100%', 'height': '54vh', 'display': 'flex', 'padding-top': '4vh',
                  'background-color': 'white'}),
        html.Div([
            html.Div(style={'float': 'left', 'background-color': 'white', 'color': 'black'},
                     className='stats-container', id='portfolio-stats'),  # portfolio overall stats
            html.Div(style={'float': 'right', 'background-color': 'white', 'color': 'black'},
                     className='stats-container', id='stock-stats'),  # individual stock stats
        ], style={'width': '100%', 'height': '30vh', 'margin': '2vh 0'}),
        html.Div([
            html.H3("XGBoost Trade Signal Backtest V. SPY", style={'margin-left': '10px', 'color': 'navy'}),
            dcc.Graph(
                id='backtest-plot',
                style={'height': '50vh', 'width': '100%', 'background-color': 'white', 'color': 'black'},
                config=config
            )
        ], style={'width': '100%', 'padding': '10px 0', 'background-color': 'white'}),
        html.Center([
            html.Button('Edit portfolio', id='edit-button', n_clicks=0,
                        className='submit-button', style={'width': '200px', 'height': '4vh'})  # re-route to return and edit portfolio contents
        ])
    ], style={'width': '100%', 'height': '100vh', 'background-color': 'white', 'display': 'none'}, id='analysis'),
    html.Div([
        html.H2("Model Playground", style={'margin-left': '10px', 'color': 'navy'}),
        html.Div("Enter insider trade details and run the model.", style={'margin-left': '10px'}),
        html.Div(style={'height': '15px'}),
        html.Div([
            dbc.Input(id='model-ticker', type='text', placeholder='ticker',
                      value='AAPL', style={'text-transform': 'uppercase'}),
            dbc.Input(id='model-company-name', type='text', placeholder='company_name',
                      value='Apple Inc'),
            dbc.Input(id='model-owner-name', type='text', placeholder='owner_name',
                      value='Cook Timothy D'),
            dbc.Input(id='model-title', type='text', placeholder='Title',
                      value='CEO'),
            dbc.Input(id='model-transaction-type', type='text', placeholder='transaction_type',
                      value='P - Purchase'),
            dbc.Input(id='model-last-price', type='text', placeholder='last_price',
                      value='175.0'),
            dbc.Input(id='model-qty', type='text', placeholder='Qty',
                      value='1000'),
            dbc.Input(id='model-shares-held', type='text', placeholder='shares_held',
                      value='5000'),
            dbc.Input(id='model-owned', type='text', placeholder='Owned',
                      value='1.0'),
            dbc.Input(id='model-value', type='text', placeholder='Value',
                      value='175000'),
        ], style={'display': 'grid',
                  'gridTemplateColumns': 'repeat(4, minmax(200px, 1fr))',
                  'gap': '10px',
                  'padding': '0 10px'}),
        html.Div(style={'height': '15px'}),
        html.Div([
            html.Button('See results', id='model-see-results', n_clicks=0, className='submit-button', style={'width': '200px'}),
            html.Button('Back to portfolio', id='back-to-portfolio-button', n_clicks=0, className='submit-button', style={'width': '200px', 'margin-left': '20px'})
        ], style={'display': 'flex', 'gap': '10px', 'margin-left': '10px'}),
        html.Div(id='model-results', style={'margin': '15px 10px', 'fontSize': '16px', 'color': 'black'})
    ], style={'width': '100%', 'height': '100vh', 'background-color': 'white', 'display': 'none'}, id='model-page')]
        #html.Div([
            #dcc.Graph(id='prediction-stock-plot', style={'height': '100%', 'width': '50%'}, config=config),
            #html.P(" Enter the prediction years:"),
            # dcc.Input(id='prediction-year', type='number',value=2018, min=1000, max=2022, step=1, className='prediction-year-input')
            #html.P("Select prediction mode:"),
            #dcc.Dropdown(id='prediction-mode', options=['Exponential prediction','Holt prediction'],
                         #value='Exponential prediction', clearable = False, className='prediction-mode-input')])
         )


# The callback function is called the number of stocks is changed
@app.callback(
    Output('tickers', 'children'),
    Output('amounts', 'children'),
    Output('errors', 'children'),
    Output('info_buttons', 'children'),
    Output('delete_buttons', 'children'),
    Input('stock-count', 'value'),
    State('tickers', 'children'),
    State('amounts', 'children'),
    State('errors', 'children'),
    State('info_buttons', 'children'),
    State('delete_buttons', 'children'), prevent_initial_call=False)
def add_stock(count, tickers, amounts, errors, infos, deletes):
    """
        Callback function for adding a stock to the given portfolio

        count: number of stocks to add to the portfolio
        tickers: given tickers of portfolio
        amounts: amount purchased of the given tickers
        errors: any error presiding over given tickers (invalid ticker)
        deleted: removed stocked from portfolio

    """
    if count is None:
        count = 1

    tickers = tickers[:count]
    amounts = amounts[:count]
    errors = errors[:count]
    infos = infos[:count]
    deletes = deletes[:count]

    for idx in range(len(tickers), count):
        tickers.append(html.Div([
            dcc.Input(id={
                'type': 'ticker',
                'index': idx,
            }, type='text', placeholder='Ticker or company', autoComplete='off', style={'text-transform': 'uppercase'}),
        ], style={'width': '20%'}, className='stock-input-div'))
        amounts.append(html.Div([
            dcc.Input(id={
                'type': 'amount',
                'index': idx,
            }, type='number', placeholder='Amount', step="0.01", min="0.01", autoComplete='off', value="1"),
        ], style={'width': '20%'}, className='stock-input-div'))
        errors.append(html.Div([
            html.Div(id={
                'type': 'error',
                'index': idx,
            })], style={'width': '75%', 'font-size': '15px'}, className='stock-input-div'))
        infos.append(html.Div([
            dbc.Button('i', id={
                'type': 'info-button',
                'index': idx
            }, n_clicks=0, className='info-button'),]+
            [dbc.Tooltip('Company or Crypto Information',
                        id={'type': 'info-tooltip', 'index':idx},
                        target={'type': 'info-button',
                                'index': idx},
                        placement='right',
                        style={
                        'display': 'inline-block',
                        'border': '1px dotted black',
                        'margin-left': '50px',
                        'margin-top': '50px',
                        'margin-bottom': '150px',
                        'border-radius':'15px',
                        'font-size': '14px',
                        'width':'300px',
                        'font-family': 'system-ui',
                        'background-color': 'lightgray',
                        'text-align': 'center',
                        'opacity':'0.7'}
                        )]
        , style={'width': '100%'}, className='stock-input-div')),
        deletes.append(html.Div([html.Div(style={'height': '8px'}),
                                 html.Button('×', id={
                                     'type': 'delete-button',
                                     'index': idx,
                                 }, n_clicks=0, className='delete-button')
                                 ], style={'width': '100%'}, className='stock-input-div'))

    return tickers, amounts, errors, infos, deletes


# The callback function is called when a stock is deleted
@app.callback(
    Output('tickers', 'children'),
    Output('amounts', 'children'),
    Output('errors', 'children'),
    Output('info_buttons', 'children'),
    Output('delete_buttons', 'children'),
    Output('stock-count', 'value'),
    Output({'type': 'delete-button', 'index': ALL}, 'n_clicks'),
    Input({'type': 'delete-button', 'index': ALL}, 'n_clicks'),
    State('tickers', 'children'),
    State('amounts', 'children'),
    State('errors', 'children'),
    State('info_buttons', 'children'),
    State('delete_buttons', 'children'),
    State('stock-count', 'value'), prevent_initial_call=True)
def delete_stock(clicks, tickers, amounts, errors, infos, deletes, count):
    """
        Callback function to remove a stock from the portfolio upon the action of the delete button

        clicks: number of stock to be deleted
        tickers: tickers for remaining stocks
        amounts: amounts for remaining stocks
        errors: errors presiding on the stocks
        deleted: delete buttons
        count: total stock count
    """
    for i in range(len(clicks)):
        if clicks[i] > 0:
            del tickers[i]
            del amounts[i]
            del errors[i]
            del infos[i]
            del deletes[i]
            count -= 1

    return tickers, amounts, errors, infos, deletes, count, [0] * len(clicks)


# The callback function is called when a stock is modified
@app.callback(Output({'type': 'error', 'index': MATCH}, 'children'),
              [Input({'type': 'ticker', 'index': MATCH}, 'value'),
               Input({'type': 'amount', 'index': MATCH}, 'value')], prevent_initial_call=False)
def update_error(ticker, amount):
    if ticker is None:
        return 'Missing or invalid ticker symbol.'

    if amount is None:
        return 'Dollar amount must have no more than 2 decimal places.'

    assert type(ticker) == str
    assert type(amount) in [int, float] or (type(amount) == str and amount.isnumeric()), type(amount)
    assert float(amount) > 0

    # Check if the ticker is valid
    try:
        # Use fast_info for a quick check, or history
        # fast_info is generally faster and reliable for existence
        if yf.Ticker(ticker).fast_info['last_price'] is None:
            return 'Ticker not present in our database.'
    except:
        # Fallback if fast_info fails or ticker is genuinely invalid
        return 'Ticker not present in our database.'

    return 'OK'


# Callback function associated with checking if a given company name is available
@app.callback(
    Output({'type': 'ticker', 'index': MATCH}, 'value'),
    Input({'type': 'ticker', 'index': MATCH}, 'value'), prevent_initial_call=True)
def check_name(company):
    """
        Callback function for adding auto-generated ticker names to the portfolio

        ticker: given tickers of portfolio
    """
    print(f'checking for companies named {company}')
    try:
        # Use history as a robust check if simple info fails, or just proceed
        # The original code logic was trying to check if 'company' is already a valid ticker
        # before searching for it as a name.
        if yf.Ticker(company).history(period='1d').empty:
             # Autocomplete helper disabled; previously called autocomplete_ticker.autocomplete_ticker(company)
             return company
    except IndexError:
        pass

    return company


# callback for company hover information
@app.callback(
    Output({'type': 'info-tooltip', 'index': MATCH}, 'children'),
    Input({'type': 'ticker', 'index': MATCH}, 'value'), prevent_initial_call=True)
def tooltip_info(ticker):
    """
        Callback function for updating the company information tooltip.

        ticker: given tickers of portfolio
    """
    if ticker is None or str(ticker).strip() == '':
        return "Company or Crypto Information"

    if insider_df is None:
        return "Insider trade data unavailable."

    ticker_upper = str(ticker).upper().strip()
    matches = insider_df[insider_df['ticker'] == ticker_upper]

    if matches.empty:
        return "No insider trade data available for this ticker."

    latest = matches.sort_values('transaction_date', ascending=False).iloc[0]

    def fmt_pct(val):
        return f"{val * 100:.2f}%" if pd.notnull(val) else "N/A"

    company_name = latest.get('company_name', ticker_upper)
    owner = latest.get('owner_name', 'N/A')
    title = latest.get('Title', 'N/A')
    txn_type = latest.get('transaction_type', 'N/A')
    last_price = latest.get('last_price', 'N/A')
    quantity = latest.get('Qty', 'N/A')
    tdate = latest.get('transaction_date', None)
    tdate_str = tdate.strftime('%Y-%m-%d') if pd.notnull(tdate) else 'N/A'

    # Previous Yahoo Finance-based implementation kept for reference:
    # try:
    #     ticker_obj = yf.Ticker(ticker)
    #     price = ticker_obj.fast_info['last_price']
    #     if price is not None:
    #         info = ticker_obj.info
    #         long_name = info.get('longName', ticker)
    #         sector = info.get('sector', 'N/A')
    #         industry = info.get('industry', 'N/A')
    #         employees = info.get('fullTimeEmployees', 'N/A')
    #         summary = info.get('longBusinessSummary', 'No summary available.')
    #         summary_short = summary[:summary.find('.', 100) + 1] if '.' in summary[100:] else summary[:200] + '...'
    #         return html.Div([
    #             html.H3(long_name),
    #             html.H4('Current Market Price:'), html.H5("$"+str(round(price, 2))),
    #             html.H4('Sector:'), html.H5(sector),
    #             html.H4('Industry:'), html.H5(industry),
    #             html.H4('Full-Time Employees:'), html.H5(employees),
    #             html.H4('Company Overview'), html.H5(summary_short)
    #         ])
    #     else:
    #         return "Invalid Company Name"
    # except Exception as e:
    #     print(f"Error fetching info for {ticker}: {e}")
    #     return "Company or Crypto Information"

    return html.Div([
        html.H3(company_name),
        html.H4('Ticker:'), html.H5(ticker_upper),
        html.H4('Last insider transaction:'), html.H5(f"{txn_type} on {tdate_str}"),
        html.H4('Transaction details:'), html.H5(f"Owner: {owner} ({title})"),
        html.H4('Quantity:'), html.H5(str(quantity)),
        html.H4('Forward returns after trade:'),
        html.H5(f"30d: {fmt_pct(latest.get('return_30d_close'))}"),
        html.H5(f"60d: {fmt_pct(latest.get('return_60d_close'))}"),
        html.H5(f"90d: {fmt_pct(latest.get('return_90d_close'))}"),
    ])

# The callback function is called when the submit button is clicked
# It sets the display of the portfolio input to none and the analysis to initial
@app.callback(
    Output('portfolio-input', 'style'),
    Output('analysis', 'style'),
    Output('portfolio-input', 'children'),
    Output('portfolio-plot', 'figure'),
    Output('individual-stock-plot', 'figure'),
    # Output('predicted-stock-plot', 'figure'),
    Output('portfolio-stats', 'children'),
    Output('stock-stats', 'children'),
    Output('backtest-plot', 'figure'),
    # Input('prediction-model','value'),
    # Input('prediction-year','value')
    Input('submit-button', 'n_clicks'),
    State('portfolio-input', 'style'),
    State('analysis', 'style'),
    State('errors', 'children'),
    State('portfolio-input', 'children'),
    State('tickers', 'children'),
    State('amounts', 'children'),
    State('start-date', 'value'), prevent_initial_call=True)
def submit_portfolio(_, portfolio_style, analysis_style, statuses, children, tickers, amounts, start_date):
    error = False
    for status in statuses:
        if status['props']['children'][0]['props']['children'] != 'OK':
            error = True
            break

    if error or len(statuses) == 0:
        children = [dbc.Alert("Please fix all errors before submitting",
                              duration=4000, fade=True, className='error-alert')] + children
        return portfolio_style, analysis_style, children, {}, {}, {}, {}, {}

    # collect the data from the input fields
    positions = {}
    for i in range(len(tickers)):
        ticker = tickers[i]['props']['children'][0]['props']['value'].upper().strip()
        amount = amounts[i]['props']['children'][0]['props']['value']
        if ticker not in positions:
            positions[ticker] = float(amount)
        else:
            positions[ticker] += float(amount)

    # set the display of the portfolio input to none and the analysis to initial
    portfolio_style['display'] = 'none'
    analysis_style['display'] = 'initial'
    analysis_style['background-color'] = 'white'
    analysis_style['color'] = 'black'

    # run the analysis
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    stock_analyser = PortfolioAnalyzer(positions, start_date)

    portfolio_graph = stock_analyser.graph_portfolio()
    individual_graph = stock_analyser.graph_individual_stocks()
    portfolio_stats = stock_analyser.get_portfolio_stats()
    stock_stats = stock_analyser.get_stock_stats()
    if 'SPY' in positions:
        backtest_graph = _build_backtest_fig(start_date, positions.get('SPY'))
        if not backtest_graph:
            backtest_graph = _empty_backtest_fig("Backtest data unavailable.")
    else:
        backtest_graph = _empty_backtest_fig()
    #if prediction_mode == 'Exponential prediction':
        #prediction_graph = stock_analyser.exponential_smoothing()
    #else:
        #prediction_graph = stock_analyser.holt_smoothing()
    return portfolio_style, analysis_style, children, portfolio_graph, individual_graph, portfolio_stats, stock_stats, backtest_graph


# The callback function is called when the edit button is clicked
# It sets the display of the portfolio input to initial and the analysis to none
@app.callback(
    Output('portfolio-input', 'style'),
    Output('analysis', 'style'),
    Input('edit-button', 'n_clicks'),
    State('portfolio-input', 'style'),
    State('analysis', 'style'), prevent_initial_call=True)
def edit_portfolio(_, portfolio_style, analysis_style):
    portfolio_style['display'] = 'initial'
    portfolio_style['background-color'] = 'white'
    portfolio_style['color'] = 'black'
    analysis_style['display'] = 'none'

    return portfolio_style, analysis_style


# Navigation between portfolio and model page
@app.callback(
    Output('portfolio-input', 'style'),
    Output('analysis', 'style'),
    Output('model-page', 'style'),
    Input('try-model-button', 'n_clicks'),
    Input('back-to-portfolio-button', 'n_clicks'),
    Input('submit-button', 'n_clicks'),
    State('portfolio-input', 'style'),
    State('analysis', 'style'),
    State('model-page', 'style'),
    prevent_initial_call=True)
def toggle_pages(try_clicks, back_clicks, submit_clicks, p_style, a_style, m_style):
    trig = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trig == 'try-model-button':
        p_style = {**(p_style or {}), 'display': 'none'}
        a_style = {**(a_style or {}), 'display': 'none'}
        m_style = {**(m_style or {}), 'display': 'initial'}
    elif trig == 'back-to-portfolio-button':
        p_style = {**(p_style or {}), 'display': 'initial', 'background-color': 'white', 'color': 'black'}
        a_style = {**(a_style or {}), 'display': 'none'}
        m_style = {**(m_style or {}), 'display': 'none'}
    elif trig == 'submit-button':
        # When submitting the portfolio, ensure the model page is hidden; leave other styles unchanged.
        m_style = {**(m_style or {}), 'display': 'none'}

    return p_style, a_style, m_style


# ========= Model prediction wiring (30/60/90) =========

def _load_artifacts(suffix: str):
    base = MODELS_DIR
    try:
        model_path = base / f"xgb_model_{suffix}.json"
        scaler_path = base / f"scaler_{suffix}.pkl"
        means_path = base / f"ticker_means_{suffix}.json"
        artifacts_path = base / f"artifacts_{suffix}.json"

        if not (model_path.exists() and scaler_path.exists() and means_path.exists() and artifacts_path.exists()):
            return None

        booster = xgb.Booster()
        booster.load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(means_path) as f:
            means = json.load(f)
        with open(artifacts_path) as f:
            artifacts = json.load(f)
        return {"model": booster, "scaler": scaler, "means": means, "artifacts": artifacts}
    except Exception as e:  # pragma: no cover
        print(f"Could not load artifacts for {suffix}: {e}")
        return None


MODELS = {
    "30d": _load_artifacts("30d"),
    "60d": _load_artifacts("60d"),
    "90d": _load_artifacts("90d"),
}

# Validation MAE (approx) from training logs; used for simple confidence banding.
MAE_BY_SUFFIX = {
    "30d": 0.1112,
    "60d": 0.1531,
    "90d": 0.1866,
}


def _predict_suffix(payload: dict, suffix: str):
    bundle = MODELS.get(suffix)
    if not bundle:
        return None, f"Artifacts for {suffix} not available."
    try:
        model = bundle["model"]
        scaler = bundle["scaler"]
        means = bundle["means"]
        artifacts = bundle["artifacts"]

        df = pd.DataFrame([{
            "ticker": payload["ticker"].upper(),
            "transaction_type": payload["transaction_type"],
            "last_price": float(payload["last_price"]),
            "Qty": float(payload["Qty"]),
            "shares_held": float(payload["shares_held"]),
            "Owned": float(payload["Owned"]),
            "Value": float(payload["Value"]),
        }])

        ticker_means = means.get("ticker_means", {})
        global_mean = means.get("global_mean", 0.0)
        df["ticker_encoded"] = df["ticker"].map(ticker_means).fillna(global_mean)
        df = df.drop(columns=["ticker"])

        df = pd.get_dummies(df, columns=["transaction_type"], drop_first=True)
        trans_cols = artifacts.get("transaction_type_columns", [])
        for col in trans_cols:
            if col not in df.columns:
                df[col] = 0

        scaled_cols = artifacts.get("scaled_cols", [])
        df[scaled_cols] = scaler.transform(df[scaled_cols])

        feature_order = artifacts.get("feature_order", list(df.columns))
        df = df.reindex(columns=feature_order, fill_value=0)

        dmat = xgb.DMatrix(df, feature_names=feature_order)
        pred = float(model.predict(dmat)[0])
        return pred, None
    except Exception as e:
        return None, f"{suffix} prediction failed: {e}"


def _build_backtest_fig(start_date: datetime.datetime, spy_amount: float):
    """
    Build a plotly figure for the precomputed backtest curves (30/60/90d vs SPY).
    Filters to user-selected start_date and scales all curves to the user's SPY dollar input.
    Returns an empty dict if data is unavailable.
    """
    if backtest_df is None or backtest_df.empty:
        return {}

    df = backtest_df.copy()
    # Ensure the date column is datetime for comparison (CSV can load as strings)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if df.empty:
        return {}
    cutoff = pd.to_datetime(start_date)
    df = df[df['date'] >= cutoff]
    if df.empty:
        return {}

    # Normalize each series to start at the user's SPY dollar input (default 10k fallback).
    target_start = float(spy_amount) if (spy_amount and spy_amount > 0) else 10000.0
    for col in ['total_equity_30d', 'total_equity_60d', 'total_equity_90d', 'spy_equity']:
        if col in df.columns:
            base = df.iloc[0].get(col, 0)
            if base and base != 0:
                df[col] = (df[col] / float(base)) * target_start

    fig = go.Figure()
    curves = [
        ('total_equity_30d', '30d equity', '#1f77b4'),
        ('total_equity_60d', '60d equity', '#ff7f0e'),
        ('total_equity_90d', '90d equity', '#2ca02c'),
        ('spy_equity', 'SPY equity', '#d62728'),
    ]

    for col, name, color in curves:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[col],
                mode='lines',
                name=name,
                line=dict(color=color, shape='hv' if name != 'SPY equity' else 'linear')
            ))

    fig.update_layout(
        title="Backtest Equity Curves",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def _empty_backtest_fig(reason: str = "Include SPY to see backtest.") -> dict:
    fig = go.Figure()
    fig.update_layout(
        title=reason,
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=reason, showarrow=False, font=dict(color="gray", size=14))]
    )
    return fig


@app.callback(
    Output('model-results', 'children'),
    Input('model-see-results', 'n_clicks'),
    State('model-ticker', 'value'),
    State('model-transaction-type', 'value'),
    State('model-last-price', 'value'),
    State('model-qty', 'value'),
    State('model-shares-held', 'value'),
    State('model-owned', 'value'),
    State('model-value', 'value'),
    prevent_initial_call=True)
def run_model(_, ticker, txn_type, last_price, qty, shares_held, owned, value):
    required = [ticker, txn_type, last_price, qty, shares_held, owned, value]
    if any(v in [None, ""] for v in required):
        return "Please fill all fields to run the model."

    payload = {
        "ticker": ticker,
        "transaction_type": txn_type,
        "last_price": last_price,
        "Qty": qty,
        "shares_held": shares_held,
        "Owned": owned,
        "Value": value,
    }
    results = {}
    errors = []
    for suffix in ["30d", "60d", "90d"]:
        pred, err = _predict_suffix(payload, suffix)
        if err:
            errors.append(err)
        else:
            results[suffix] = pred

    if errors and not results:
        return "; ".join(errors)

    parts = []
    for suffix in ["30d", "60d", "90d"]:
        if suffix in results:
            pred = results[suffix]
            mae = MAE_BY_SUFFIX.get(suffix)
            if mae:
                conf = 1 / (1 + mae / (abs(pred) + 1e-9))
                conf_pct = conf * 100
                parts.append(
                    f"• {suffix}: {pred:.4f} (conf≈{conf_pct:.1f}%)"
                )
            else:
                parts.append(f"• {suffix}: {pred:.4f}")
    if errors:
        parts.append(f"(Warnings: {'; '.join(errors)})")
    return "Predicted returns:\n" + "\n".join(parts)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the portfolio dashboard.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8050)),
        help="Port to run the Dash app on (default: 8050 or PORT env var)."
    )
    args = parser.parse_args()

    app.run(debug=True, port=args.port)