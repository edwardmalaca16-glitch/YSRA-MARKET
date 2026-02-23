from flask import Flask, render_template, jsonify, request as freq
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

TOP_500_STOCKS = [
    # Technology
    'AAPL','MSFT','NVDA','AVGO','ORCL','ADBE','CRM','AMD','INTC','CSCO',
    'QCOM','TXN','MU','AMAT','LRCX','KLAC','SNPS','CDNS','MRVL','NXPI',
    'ON','STX','WDC','HPQ','HPE','DELL','NTAP','PSTG','IBM','ACN',
    'NOW','PANW','FTNT','CRWD','ZS','OKTA','DDOG','NET','MDB','SNOW',
    'PLTR','PATH','GTLB','HUBS','CFLT','TEAM','WDAY','VEEV','APPF','ZM',
    'DOCU','DOCN','TWLO','ESTC','PD','TENB','S','QLYS','RPD','FROG',
    # Big Tech
    'GOOGL','GOOG','AMZN','META','NFLX','TSLA',
    # Finance
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','COF',
    'USB','PNC','TFC','MTB','RF','CFG','HBAN','KEY','FITB','ZION',
    'BK','STT','NTRS','IVZ','TROW','RJF','ICE','CME','NDAQ','CBOE',
    'MCO','SPGI','V','MA','PYPL','FISV','FIS','GPN','WEX','JKHY',
    'AFL','AIG','MET','PRU','ALL','TRV','CB','HIG','LNC','PGR',
    'CINF','ERIE','RLI','WRB','AXS','RNR','MKL','ACGL','REN','FAF',
    # Healthcare
    'JNJ','UNH','LLY','ABBV','MRK','PFE','TMO','ABT','DHR','SYK',
    'BSX','MDT','EW','ISRG','ZTS','VRTX','REGN','GILD','BIIB','MRNA',
    'BDX','BAX','HOLX','DXCM','PODD','INCY','ALNY','HCA','HUM','ELV',
    'CI','CVS','MCK','ABC','CAH','VEEV','IQV','CRL','MEDP','ICLR',
    'AMGN','ILMN','IONS','SGEN','BMRN','EXEL','RARE','SRPT','ACAD','JAZZ',
    'FOLD','ARVN','ARWR','NTLA','BEAM','EDIT','CRSP','BLUE','FATE','KYMR',
    # Consumer Staples
    'PG','KO','PEP','MDLZ','GIS','K','CPB','SJM','CAG','MKC',
    'PM','MO','STZ','BUD','TAP','SAM','WMT','COST','TGT','KR',
    'SYY','PFGC','CHEF','USFD','CASY','SFM','WEIS','GO','IMKTA','VLGEA',
    # Consumer Discretionary
    'HD','LOW','TJX','ROST','BURL','FIVE','NKE','SBUX','MCD','YUM',
    'QSR','DPZ','CMG','TXRH','DRI','EAT','BJRI','CAKE','PLAY','RRGB',
    'DIS','CHTR','CMCSA','PARA','WBD','LYV','SEAS','FUN','SIX','PRKS',
    'F','GM','APTV','LEA','BWA','DAN','DORM','THRM','MTOR','ADNT',
    'EBAY','ETSY','W','CHWY','BBY','KSS','M','JWN','GPS','ANF',
    'AEO','URBN','PVH','RL','TPR','CPRI','VFC','HBI','SKX','CROX',
    # Industrials
    'GE','HON','MMM','CAT','DE','BA','RTX','LMT','NOC','GD',
    'UPS','FDX','XPO','ODFL','SAIA','KNX','WERN','JBHT','CHRW','EXPD',
    'EMR','ETN','PH','ROK','AME','GNRC','XYL','IR','TT','CARR',
    'OTIS','TDG','HEI','AXON','KTOS','LDOS','SAIC','BAH','CACI','KEYW',
    'WM','RSG','WCN','CLH','SRCL','CWST','GFL','ARIS','HCCI','NWFL',
    'UNP','CSX','NSC','CP','CN','WAB','TRN','GATX','RAIL','GBX',
    # Energy
    'XOM','CVX','COP','EOG','PXD','DVN','MPC','PSX','VLO','HES',
    'OXY','FANG','APA','MRO','SWN','RRC','AR','CNX','EQT','SM',
    'SLB','HAL','BKR','NOV','HP','PTEN','NE','RIG','VAL','DO',
    'OKE','WMB','KMI','EPD','ET','MMP','PAA','TRGP','LNG','CTRA',
    'NEE','SO','DUK','AEP','EXC','SRE','PCG','ED','XEL','WEC',
    'ETR','ES','PEG','EIX','PPL','AEE','DTE','CMS','NI','OGE',
    # Materials
    'LIN','APD','SHW','ECL','PPG','EMN','CE','DD','DOW','LYB',
    'NEM','GOLD','AEM','WPM','FNV','FCX','SCCO','AA','NUE','STLD',
    'RS','CMC','X','CLF','MT','ATI','CRS','HWM','ARNC','KALU',
    # Real Estate
    'PLD','AMT','CCI','EQIX','PSA','O','DLR','SPG','EQR','AVB',
    'ESS','MAA','UDR','CPT','NNN','VICI','GLPI','WPC','STOR','ADC',
    'HST','RHP','PK','SHO','APLE','IRM','COLD','CUBE','EXR','LSI',
    # Communication Services
    'T','VZ','TMUS','CHTR','LBRDA','WBD','PARA','FOX','FOXA','NYT',
    'NWSA','SPOT','PINS','SNAP','MTCH','IAC','ZI','ANGI','EXPE','BKNG',
    'ABNB','TRIP','LYFT','UBER','DASH','RBLX','TTWO','EA','ATVI','U',
    # Airlines & Travel
    'DAL','UAL','AAL','LUV','ALK','HA','JBLU','SAVE','CCL','RCL',
    'NCLH','VAC','HGV','TNL','HLT','MAR','H','IHG','CHH','WH',
    # Autos & EV
    'RIVN','LCID','NIO','LI','XPEV','HOG','RACE','TM','HMC','STLA',
    # Biotech Extended
    'RCUS','IMVT','PRTA','KROS','PRTK','FGEN','ADMA','ACCD','ARDX','ADAP',
    # Semiconductors Extended
    'ADI','MCHP','SWKS','QRVO','MPWR','ENTG','ACLS','ONTO','FORM','COHU',
    'ICHR','KLIC','UCTT','AXTI','AMBA','SITM','ALGM','DIOD','SLAB','SMTC',
    # Fintech Extended
    'SQ','AFRM','UPST','SOFI','LC','OPFI','CURO','WRLD','QFIN','CACC',
    # Misc S&P 500
    'BRK-B','MMC','AON','WTW','VRSK','CSGP','ANSS','TYL','EPAM','CTSH',
    'INFY','WIT','GLOB','LSPD','TASK','RELY','FLYW','FOUR','PAYO','DLO',
]

# Company names mapping for better display
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc.', 'GOOG': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms Inc.', 'NVDA': 'NVIDIA Corp.', 'TSLA': 'Tesla Inc.',
    'JPM': 'JPMorgan Chase & Co.', 'BAC': 'Bank of America Corp.', 'WFC': 'Wells Fargo & Co.',
    'GS': 'Goldman Sachs Group Inc.', 'MS': 'Morgan Stanley', 'V': 'Visa Inc.', 'MA': 'Mastercard Inc.',
    'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth Group Inc.', 'PFE': 'Pfizer Inc.', 'MRK': 'Merck & Co.',
    'WMT': 'Walmart Inc.', 'COST': 'Costco Wholesale Corp.', 'HD': 'Home Depot Inc.', 'MCD': "McDonald's Corp.",
    'NKE': 'Nike Inc.', 'SBUX': 'Starbucks Corp.', 'DIS': 'Walt Disney Co.', 'NFLX': 'Netflix Inc.',
    'INTC': 'Intel Corp.', 'AMD': 'Advanced Micro Devices Inc.', 'CSCO': 'Cisco Systems Inc.',
    'IBM': 'International Business Machines', 'ORCL': 'Oracle Corp.', 'CRM': 'Salesforce Inc.',
    'PYPL': 'PayPal Holdings Inc.', 'UBER': 'Uber Technologies Inc.', 'LYFT': 'Lyft Inc.',
    'XOM': 'Exxon Mobil Corp.', 'CVX': 'Chevron Corp.', 'BA': 'Boeing Co.', 'CAT': 'Caterpillar Inc.',
    'GE': 'General Electric Co.', 'HON': 'Honeywell International Inc.', 'UPS': 'United Parcel Service Inc.',
    'FDX': 'FedEx Corp.', 'MMM': '3M Co.', 'PEP': 'PepsiCo Inc.', 'KO': 'Coca-Cola Co.',
}

# Sector mapping for variety
SECTORS = ['Technology', 'Healthcare', 'Financial', 'Consumer Cyclical', 
           'Consumer Defensive', 'Industrials', 'Energy', 'Basic Materials', 
           'Real Estate', 'Communication Services', 'Utilities']

# Cache for generated stocks to avoid regenerating on every request
stock_cache = {}
cache_timestamp = {}

def generate_mock_stock(symbol, index):
    """Generate realistic mock stock data"""
    
    # Check if we have a cached version less than 5 minutes old
    cache_key = f"{symbol}"
    if cache_key in stock_cache:
        cache_age = datetime.now() - cache_timestamp.get(cache_key, datetime.now())
        if cache_age.seconds < 300:  # 5 minute cache
            return stock_cache[cache_key]
    
    # Base price varies by sector/type
    if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']:  # Tech giants
        base_price = random.uniform(150, 500)
    elif symbol in ['BRK-B', 'JPM', 'V', 'MA']:  # Financial giants
        base_price = random.uniform(150, 600)
    elif symbol in ['TSLA']:  # Volatile stocks
        base_price = random.uniform(180, 400)
    else:
        base_price = random.uniform(30, 250)
    
    # Daily change with realistic distribution
    daily_change_pct = random.gauss(0.1, 2.5)  # Mean 0.1%, std dev 2.5%
    daily_change = base_price * (daily_change_pct / 100)
    daily_pct = daily_change_pct
    
    # Weekly and monthly changes (more variation)
    weekly_pct = daily_pct + random.gauss(0, 1.5)
    monthly_pct = daily_pct + random.gauss(0, 3)
    
    # Get company name
    name = COMPANY_NAMES.get(symbol, f"{symbol} Corporation")
    
    # Sector based on stock
    if symbol in ['AAPL','MSFT','NVDA','AVGO','ORCL','ADBE','CRM','AMD','INTC','CSCO']:
        sector = 'Technology'
    elif symbol in ['JPM','BAC','WFC','GS','MS','C','V','MA','PYPL']:
        sector = 'Financial'
    elif symbol in ['JNJ','UNH','PFE','MRK','ABBV','TMO','LLY','ABT']:
        sector = 'Healthcare'
    else:
        sector = random.choice(SECTORS)
    
    # RSI (0-100)
    rsi = random.uniform(25, 75)
    
    # Trend type based on RSI and daily change
    if rsi > 65 and daily_pct > 1:
        trend_type = "Strong Uptrend"
    elif rsi > 55 and daily_pct > 0:
        trend_type = "Uptrend"
    elif rsi < 35 and daily_pct < -1:
        trend_type = "Strong Downtrend"
    elif rsi < 45 and daily_pct < 0:
        trend_type = "Downtrend"
    else:
        trend_type = "Neutral"
    
    # Score calculation (0-100) with some logic
    score = 50
    if rsi < 30:
        score += 20
    elif rsi > 70:
        score -= 20
    if "Strong Uptrend" in trend_type:
        score += 30
    elif "Uptrend" in trend_type:
        score += 15
    elif "Strong Downtrend" in trend_type:
        score -= 30
    elif "Downtrend" in trend_type:
        score -= 15
    if daily_pct > 2:
        score += 20
    elif daily_pct > 0:
        score += 10
    elif daily_pct < -2:
        score -= 20
    elif daily_pct < 0:
        score -= 10
    
    score = max(0, min(100, int(score)))
    
    # Action based on score
    if score >= 70:
        action = "BUY"
        action_color = "green"
    elif score >= 40:
        action = "HOLD"
        action_color = "orange"
    else:
        action = "SELL"
        action_color = "red"
    
    # Market cap based on stock type
    if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']:  # Mega-cap
        market_cap = random.uniform(800e9, 3e12)
    elif symbol in ['JPM', 'V', 'MA', 'WMT', 'JNJ', 'UNH']:  # Large-cap
        market_cap = random.uniform(200e9, 800e9)
    else:
        market_cap = random.uniform(10e9, 200e9)
    
    # Format market cap
    if market_cap >= 1e12:
        mc_str = f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        mc_str = f"${market_cap/1e9:.2f}B"
    else:
        mc_str = f"${market_cap/1e6:.2f}M"
    
    # Sparkline data (30 days) with trend
    sparkline = []
    trend_factor = random.uniform(-0.5, 0.5)
    for i in range(30):
        if trend_factor > 0.2:  # Uptrend
            price = base_price * (0.95 + (i/30)*0.1 + random.uniform(-0.02, 0.02))
        elif trend_factor < -0.2:  # Downtrend
            price = base_price * (1.05 - (i/30)*0.1 + random.uniform(-0.02, 0.02))
        else:  # Sideways
            price = base_price * (1 + random.uniform(-0.05, 0.05))
        sparkline.append(round(price, 2))
    
    # Next day prediction
    next_prediction = base_price * (1 + random.gauss(0.1, 1.5)/100)
    pred_change = ((next_prediction - base_price) / base_price) * 100
    
    # Volume
    volume = random.randint(1000000, 50000000)
    avg_volume = int(volume * random.uniform(0.7, 1.3))
    
    # 52-week range
    week_high = base_price * random.uniform(1.1, 1.5)
    week_low = base_price * random.uniform(0.6, 0.9)
    
    # Beta
    if sector == 'Technology':
        beta = random.uniform(0.9, 1.5)
    elif sector == 'Utilities':
        beta = random.uniform(0.3, 0.7)
    else:
        beta = random.uniform(0.6, 1.3)
    
    # Dividend yield
    if sector in ['Utilities', 'Consumer Defensive']:
        dividend = random.uniform(2, 5)
    elif sector == 'Technology':
        dividend = random.uniform(0, 1.5)
    else:
        dividend = random.uniform(0, 3)
    
    # PE Ratio
    pe_ratio = random.uniform(10, 35)
    
    stock_data = {
        'symbol': symbol,
        'name': name,
        'sector': sector,
        'price': round(base_price, 2),
        'previous_close': round(base_price - daily_change, 2),
        'change': round(daily_change, 2),
        'daily_percentage': round(daily_pct, 2),
        'weekly_percentage': round(weekly_pct, 2),
        'monthly_percentage': round(monthly_pct, 2),
        'rsi': round(rsi, 1),
        'trend_type': trend_type,
        'score': score,
        'action': action,
        'action_color': action_color,
        'next_prediction': round(next_prediction, 2),
        'pred_change': round(pred_change, 2),
        'sparkline': sparkline[-10:],  # Last 10 points for sparkline
        'volume': volume,
        'market_cap': market_cap,
        'market_cap_str': mc_str,
        'pe_ratio': round(pe_ratio, 2),
        'dividend_yield': round(dividend, 2),
        '52w_high': round(week_high, 2),
        '52w_low': round(week_low, 2),
        'avg_volume': avg_volume,
        'beta': round(beta, 2),
    }
    
    # Cache the result
    stock_cache[cache_key] = stock_data
    cache_timestamp[cache_key] = datetime.now()
    
    return stock_data

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    return render_template('stock_detail.html', symbol=symbol.upper())

@app.route('/all-stocks')
def all_stocks():
    return render_template('all_stocks.html', total_stocks=len(TOP_500_STOCKS))

@app.route('/api/stocks/top')
def api_top_stocks():
    """Return top scored stocks for dashboard"""
    all_stocks = []
    # Generate only first 30 for top list (lighter load)
    for i, symbol in enumerate(TOP_500_STOCKS[:30]):
        stock = generate_mock_stock(symbol, i)
        all_stocks.append(stock)
    
    all_stocks.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': all_stocks[:10]})  # Return top 10

@app.route('/api/stocks/page')
def api_stocks_page():
    """Return paginated stocks - only fetches the requested page"""
    page = int(freq.args.get('page', 1))
    per_page = int(freq.args.get('per_page', 20))
    
    # Calculate start and end indices
    start = (page - 1) * per_page
    end = min(start + per_page, len(TOP_500_STOCKS))
    
    # Get the batch of symbols for this page
    batch_symbols = TOP_500_STOCKS[start:end]
    
    # Generate data only for this page
    results = []
    for i, symbol in enumerate(batch_symbols):
        stock = generate_mock_stock(symbol, start + i)
        results.append(stock)
    
    # Sort by score for this page
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify({
        'stocks': results,
        'page': page,
        'per_page': per_page,
        'total': len(TOP_500_STOCKS),
        'total_pages': (len(TOP_500_STOCKS) + per_page - 1) // per_page,
        'has_next': end < len(TOP_500_STOCKS),
        'has_prev': page > 1
    })

@app.route('/api/stock/<symbol>')
def api_stock_detail(symbol):
    """Return detailed data for a specific stock"""
    symbol = symbol.upper()
    if symbol not in TOP_500_STOCKS:
        return jsonify({'error': 'Stock not found'}), 404
    
    index = TOP_500_STOCKS.index(symbol)
    stock_data = generate_mock_stock(symbol, index)
    return jsonify(stock_data)

@app.route('/api/stock/<symbol>/history')
def api_stock_history(symbol):
    """Return price history for chart"""
    symbol = symbol.upper()
    if symbol not in TOP_500_STOCKS:
        return jsonify({'error': 'Stock not found'}), 404
    
    # Generate 6 months of historical data
    history = []
    base_price = random.uniform(50, 500)
    
    # Create a trend
    trend = random.choice(['up', 'down', 'sideways', 'volatile'])
    
    for i in range(180):  # 6 months
        date = (datetime.now() - timedelta(days=180-i)).strftime('%Y-%m-%d')
        
        if trend == 'up':
            price_factor = 0.8 + (i/180) * 0.4
        elif trend == 'down':
            price_factor = 1.2 - (i/180) * 0.4
        elif trend == 'volatile':
            price_factor = 1 + 0.15 * (i % 20 - 10) / 10
        else:  # sideways
            price_factor = 1 + random.uniform(-0.1, 0.1)
        
        price = base_price * price_factor + random.uniform(-5, 5)
        
        # Generate OHLC data
        open_price = price + random.uniform(-2, 2)
        high_price = max(open_price, price) + random.uniform(0, 3)
        low_price = min(open_price, price) - random.uniform(0, 3)
        
        history.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(price, 2),
            'volume': random.randint(1000000, 50000000)
        })
    
    return jsonify({'history': history})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
