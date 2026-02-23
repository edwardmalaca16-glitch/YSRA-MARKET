from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Expanded to 500 stocks (top companies by market cap)
TOP_500_STOCKS = [
    # Technology (100)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC',
    'CRM', 'ADBE', 'NFLX', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AVGO', 'MU',
    'NOW', 'SHOP', 'SQ', 'PYPL', 'UBER', 'LYFT', 'SNAP', 'PINS', 'SPOT', 'ZM',
    'DOCU', 'OKTA', 'CRWD', 'PANW', 'FTNT', 'ZS', 'NET', 'DDOG', 'MDB', 'SNOW',
    'PLTR', 'U', 'AFRM', 'COIN', 'ROKU', 'RBLX', 'HOOD', 'SOFI', 'DKNG', 'BIDU',
    'JD', 'BABA', 'NTES', 'TCEHY', 'SE', 'MELI', 'PDD', 'WIX', 'ETSY', 'VRSN',
    'AKAM', 'CDNS', 'SNPS', 'ANSS', 'KEYS', 'NXPI', 'MCHP', 'ADI', 'MPWR', 'ON',
    'SWKS', 'QRVO', 'LSCC', 'ALGN', 'ILMN', 'VEEV', 'CDW', 'FFIV', 'JNPR', 'NTAP',
    'PSTG', 'WDC', 'STX', 'HPQ', 'HPE', 'DELL', 'SMCI', 'ZBRA', 'PTC', 'TYL',
    'PAYC', 'PAYX', 'ADP', 'CTSH', 'INFY', 'WIT', 'HCLTECH', 'TCS', 'LTIM', 'TECHM',

    # Financials (80)
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'BLK', 'BX', 'KKR',
    'APO', 'ARES', 'OWL', 'TPG', 'CG', 'BEN', 'IVZ', 'AMP', 'PFG', 'AFL',
    'MET', 'PRU', 'AIG', 'ALL', 'CB', 'PGR', 'TRV', 'ACGL', 'HIG', 'CINF',
    'L', 'MKL', 'AJG', 'MMC', 'AON', 'WTW', 'BRO', 'WRB', 'ERIE', 'FDS',
    'MCO', 'SPGI', 'V', 'MA', 'AXP', 'COF', 'DFS', 'SYF', 'ALLY', 'MTB',
    'PNC', 'USB', 'TFC', 'FITB', 'RF', 'HBAN', 'KEY', 'CMA', 'ZION', 'CFG',
    'SCHW', 'IBKR', 'LPLA', 'HOOD', 'RJF', 'EVR', 'PIPR', 'PJT', 'MC', 'HLI',
    'LAZ', 'EBC', 'FHI', 'DHIL', 'GSHD', 'LMND', 'ROOT', 'HIPO', 'OSCR', 'PLMR',

    # Healthcare (80)
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'ABT', 'TMO', 'DHR', 'LLY', 'BMY',
    'AMGN', 'GILD', 'REGN', 'VRTX', 'MRNA', 'BIIB', 'INCY', 'ALNY', 'IONS', 'SRPT',
    'SNY', 'NVS', 'AZN', 'GSK', 'SAN', 'BAYN.DE', 'ROG.SW', 'NOVO-B.CO', 'CHTR', 'LHCG',
    'CVS', 'WBA', 'CAH', 'MCK', 'ABC', 'COR', 'HUM', 'CNC', 'ANTM', 'CI',
    'MOH', 'UHS', 'HCA', 'THC', 'CYH', 'EHC', 'DVA', 'FMS', 'Fresenius', 'BDX',
    'BSX', 'SYK', 'MDT', 'ZBH', 'STE', 'HOLX', 'COO', 'TFX', 'RMD', 'MASI',
    'ISRG', 'EW', 'ABMD', 'ATRI', 'PODD', 'TNDM', 'DXCM', 'DHR', 'PKI', 'WST',
    'AVTR', 'NVST', 'MTD', 'Mettler', 'WAT', 'BIO', 'TECH', 'BRKR', 'AZTA', 'MEDP',

    # Consumer (80)
    'WMT', 'COST', 'TGT', 'DG', 'DLTR', 'BJ', 'AMZN', 'EBAY', 'ETSY', 'CVNA',
    'MCD', 'SBUX', 'CMG', 'DPZ', 'YUM', 'QSR', 'WEN', 'JACK', 'SHAK', 'CAKE',
    'NKE', 'LULU', 'UA', 'UAA', 'VFC', 'PVH', 'RL', 'GIII', 'CROX', 'BOOT',
    'SBUX', 'PEP', 'KO', 'MNST', 'KDP', 'CELH', 'REED', 'BUD', 'TAP', 'STZ',
    'PG', 'CL', 'KMB', 'CHD', 'EL', 'COTY', 'IPAR', 'NUS', 'SKIN', 'OLAPLEX',
    'MO', 'PM', 'BTI', 'RLX', 'TPB', 'VGR', 'UVV', 'IMBBY', 'IMPRO', 'XXII',
    'GM', 'F', 'STLA', 'RIVN', 'LCID', 'TSLA', 'HMC', 'TM', 'VWAGY', 'BMWYY',
    'HD', 'LOW', 'SHW', 'MAS', 'TOL', 'PHM', 'LEN', 'DHI', 'NVR', 'KBH',

    # Energy (80)
    'XOM', 'CVX', 'SHEL', 'TTE', 'BP', 'COP', 'EOG', 'PXD', 'OXY', 'DVN',
    'HES', 'FANG', 'APA', 'MRO', 'CHK', 'RRC', 'SWN', 'AR', 'CNX', 'EQT',
    'SLB', 'HAL', 'BKR', 'FTI', 'NOV', 'CHX', 'WFRD', 'LBRT', 'PUMP', 'NBR',
    'PSX', 'VLO', 'MPC', 'PBF', 'DK', 'CVI', 'DINO', 'VAL', 'CLNE', 'GPRE',
    'KMI', 'WMB', 'OKE', 'ET', 'EPD', 'MMP', 'PAA', 'DCP', 'ENLC', 'WES',
    'SUN', 'NS', 'USAC', 'CAPL', 'GEL', 'DKL', 'HESM', 'MWE', 'SEP', 'SHLX',
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'PEG', 'ED', 'WEC', 'XEL',
    'ETR', 'FE', 'DTE', 'PPL', 'AEE', 'CMS', 'CNP', 'NJR', 'OGS', 'SWX',

    # Industrials (80)
    'UPS', 'FDX', 'DHL', 'CHRW', 'JBHT', 'KNX', 'ODFL', 'SAIA', 'WERN', 'ARCB',
    'BA', 'LMT', 'NOC', 'GD', 'RTX', 'HII', 'TXT', 'LHX', 'HEI', 'TDG',
    'CAT', 'DE', 'CMI', 'PCAR', 'OSK', 'TEX', 'ALSN', 'WAB', 'GWW', 'FAST',
    'HON', 'MMM', 'GE', 'EMR', 'ROK', 'ABB', 'IR', 'ETN', 'AME', 'PH',
    'UNP', 'NSC', 'CSX', 'CNI', 'CP', 'KSU', 'WAB', 'TRN', 'GBX', 'RAIL',
    'RSG', 'WM', 'WCN', 'CLH', 'SRCL', 'GFL', 'PESI', 'VEON', 'MEG', 'ECOL',
    'J', 'R', 'ALK', 'DAL', 'UAL', 'LUV', 'AAL', 'SAVE', 'JBLU', 'SKYW',
    'EXPD', 'XPO', 'UPS', 'FWRD', 'MATX', 'GRPN', 'GXO', 'RLGT', 'AIR', 'TGH'
]

def generate_mock_stock(symbol, index):
    """Generate mock stock data (since we don't want to hit Yahoo Finance 500 times)"""
    base_price = random.uniform(20, 500)
    daily_change = random.uniform(-5, 5)
    daily_pct = (daily_change / base_price) * 100
    
    # Generate company name
    names = {
        'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.', 'NVDA': 'NVIDIA Corp.', 'META': 'Meta Platforms Inc.',
        'TSLA': 'Tesla Inc.', 'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America',
        'WMT': 'Walmart Inc.', 'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth',
        'V': 'Visa Inc.', 'MA': 'Mastercard Inc.', 'HD': 'Home Depot Inc.',
        'DIS': 'Walt Disney Co.', 'NFLX': 'Netflix Inc.', 'PYPL': 'PayPal Holdings',
        'ADBE': 'Adobe Inc.', 'CRM': 'Salesforce Inc.', 'INTC': 'Intel Corp.',
        'AMD': 'Advanced Micro Devices', 'CSCO': 'Cisco Systems', 'TXN': 'Texas Instruments',
        'QCOM': 'Qualcomm Inc.', 'MU': 'Micron Technology', 'NKE': 'Nike Inc.',
        'SBUX': 'Starbucks Corp.', 'MCD': "McDonald's Corp.", 'PEP': 'PepsiCo Inc.',
        'KO': 'Coca-Cola Co.', 'PFE': 'Pfizer Inc.', 'MRK': 'Merck & Co.',
        'ABBV': 'AbbVie Inc.', 'TMO': 'Thermo Fisher', 'DHR': 'Danaher Corp.',
        'CVX': 'Chevron Corp.', 'XOM': 'Exxon Mobil', 'COP': 'ConocoPhillips',
        'BA': 'Boeing Co.', 'CAT': 'Caterpillar Inc.', 'GE': 'General Electric',
        'HON': 'Honeywell Intl', 'UPS': 'United Parcel Service', 'FDX': 'FedEx Corp.'
    }
    
    name = names.get(symbol, f"{symbol} Corporation")
    
    # RSI (0-100)
    rsi = random.uniform(20, 80)
    
    # Trend type based on RSI and daily change
    if rsi > 65 and daily_pct > 1:
        trend = "Strong Uptrend"
    elif rsi > 55 and daily_pct > 0:
        trend = "Uptrend"
    elif rsi < 35 and daily_pct < -1:
        trend = "Strong Downtrend"
    elif rsi < 45 and daily_pct < 0:
        trend = "Downtrend"
    else:
        trend = "Neutral"
    
    # Score (0-100)
    score = random.randint(30, 95)
    
    # Action based on score
    if score >= 70:
        action = "BUY"
        action_color = "buy"
    elif score >= 40:
        action = "HOLD"
        action_color = "hold"
    else:
        action = "SELL"
        action_color = "sell"
    
    # Market cap (random between 1B and 2T)
    market_cap = random.uniform(1e9, 2e12)
    
    # Sparkline data (20 days)
    sparkline = [base_price + random.uniform(-10, 10) for _ in range(20)]
    
    return {
        'symbol': symbol,
        'name': name,
        'price': round(base_price, 2),
        'change': round(daily_change, 2),
        'daily_percentage': round(daily_pct, 2),
        'weekly_percentage': round(daily_pct * random.uniform(0.5, 2), 2),
        'monthly_percentage': round(daily_pct * random.uniform(2, 5), 2),
        'rsi': round(rsi, 1),
        'trend_type': trend,
        'previous_close': round(base_price - daily_change, 2),
        'next_prediction': round(base_price * random.uniform(0.95, 1.05), 2),
        'pred_change': round(random.uniform(-3, 3), 2),
        'volume': random.randint(100000, 10000000),
        'avg_volume': random.randint(200000, 8000000),
        'market_cap': market_cap,
        'market_cap_str': format_market_cap(market_cap),
        'pe_ratio': round(random.uniform(10, 30), 1),
        '52w_high': round(base_price * random.uniform(1.1, 1.5), 2),
        '52w_low': round(base_price * random.uniform(0.5, 0.9), 2),
        'beta': round(random.uniform(0.5, 2), 2),
        'dividend_yield': round(random.uniform(0, 3), 2),
        'score': score,
        'action': action,
        'action_color': action_color,
        'sparkline': [round(x, 2) for x in sparkline]
    }

def format_market_cap(cap):
    """Format market cap to readable string"""
    if cap >= 1e12:
        return f"${cap/1e12:.2f}T"
    elif cap >= 1e9:
        return f"${cap/1e9:.2f}B"
    elif cap >= 1e6:
        return f"${cap/1e6:.2f}M"
    else:
        return f"${cap:.0f}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/all-stocks')
def all_stocks():
    return render_template('all_stocks.html')

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    return render_template('stock_detail.html', symbol=symbol)

# API Routes
@app.route('/api/stocks/top')
def api_top_stocks():
    """Return top scored stocks (for dashboard)"""
    # Generate all 500 stocks
    all_stocks_list = []
    for i, symbol in enumerate(TOP_500_STOCKS[:500]):  # Use all 500
        stock = generate_mock_stock(symbol, i)
        all_stocks_list.append(stock)
    
    # Sort by score and return top 10 (or however many you want)
    all_stocks_list.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': all_stocks_list[:20]})  # Return top 20 for dashboard

@app.route('/api/stocks/all')
def api_all_stocks():
    """Return ALL 500 stocks"""
    all_stocks_list = []
    for i, symbol in enumerate(TOP_500_STOCKS[:500]):  # Use all 500
        stock = generate_mock_stock(symbol, i)
        all_stocks_list.append(stock)
    
    # Sort by score for consistency
    all_stocks_list.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': all_stocks_list})  # Return ALL 500

@app.route('/api/stock/<symbol>')
def api_stock_detail(symbol):
    """Return detailed data for a specific stock"""
    try:
        index = TOP_500_STOCKS.index(symbol.upper())
        stock_data = generate_mock_stock(symbol.upper(), index)
        return jsonify(stock_data)
    except ValueError:
        return jsonify({'error': 'Stock not found'}), 404

@app.route('/api/stock/<symbol>/history')
def api_stock_history(symbol):
    """Return price history for chart"""
    # Generate mock history data
    base_price = random.uniform(50, 500)
    history = []
    
    for i in range(180):  # 6 months
        date = (datetime.now() - timedelta(days=180-i)).strftime('%Y-%m-%d')
        price = base_price + random.uniform(-20, 20)
        history.append({
            'date': date,
            'close': round(price, 2),
            'volume': random.randint(100000, 10000000)
        })
    
    return jsonify({'history': history})

if __name__ == '__main__':
