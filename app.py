from flask import Flask, render_template, jsonify, request as freq
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Deduplicate
seen = set()
unique = []
for s in TOP_500_STOCKS:
    if s not in seen:
        seen.add(s)
        unique.append(s)
TOP_500_STOCKS = unique


def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return round(float(val), 1) if not pd.isna(val) else 50.0
    except:
        return 50.0

def calculate_trend_type(prices):
    try:
        if len(prices) < 50:
            return "Neutral"
        ma20 = prices.rolling(window=20).mean().iloc[-1]
        ma50 = prices.rolling(window=50).mean().iloc[-1]
        current = prices.iloc[-1]
        if current > ma20 > ma50:
            return "Strong Uptrend"
        elif current > ma20:
            return "Uptrend"
        elif current < ma20 < ma50:
            return "Strong Downtrend"
        elif current < ma20:
            return "Downtrend"
        return "Neutral"
    except:
        return "Neutral"

def calculate_score(rsi, trend_type, daily_change):
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
    if daily_change > 2:
        score += 20
    elif daily_change > 0:
        score += 10
    elif daily_change < -2:
        score -= 20
    elif daily_change < 0:
        score -= 10
    return max(0, min(100, int(score)))

def predict_next_value(prices):
    try:
        prices = prices.dropna()
        if len(prices) < 5:
            return float(prices.iloc[-1])
        X = np.array(range(len(prices))).reshape(-1, 1)
        y = prices.values
        if np.isnan(y).any():
            return float(prices.iloc[-1])
        model = LinearRegression()
        model.fit(X, y)
        result = float(model.predict([[len(prices)]])[0])
        return result if not np.isnan(result) else float(prices.iloc[-1])
    except:
        return float(prices.iloc[-1])

def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="3mo")
        if hist.empty or len(hist) < 2:
            return None

        closes = hist['Close'].dropna()
        if len(closes) < 2:
            return None

        current_price = float(closes.iloc[-1])
        previous_close = float(closes.iloc[-2])
        daily_change = ((current_price - previous_close) / previous_close) * 100
        change = current_price - previous_close
        rsi = calculate_rsi(closes)
        trend_type = calculate_trend_type(closes)
        next_prediction = predict_next_value(closes)
        score = calculate_score(rsi, trend_type, daily_change)

        if score >= 70:
            action, action_color = "BUY", "green"
        elif score >= 40:
            action, action_color = "HOLD", "orange"
        else:
            action, action_color = "SELL", "red"

        week_ago = float(closes.iloc[-6]) if len(closes) >= 6 else previous_close
        month_ago = float(closes.iloc[-22]) if len(closes) >= 22 else previous_close
        weekly_change = ((current_price - week_ago) / week_ago) * 100
        monthly_change = ((current_price - month_ago) / month_ago) * 100
        pred_change = ((next_prediction - current_price) / current_price) * 100
        sparkline = [round(float(v), 2) for v in closes.tail(30).tolist()]

        market_cap = info.get('marketCap', 0) or 0
        if market_cap >= 1e12:
            mc_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            mc_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 0:
            mc_str = f"${market_cap/1e6:.2f}M"
        else:
            mc_str = "N/A"

        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'price': round(current_price, 2),
            'previous_close': round(previous_close, 2),
            'change': round(change, 2),
            'daily_percentage': round(daily_change, 2),
            'weekly_percentage': round(weekly_change, 2),
            'monthly_percentage': round(monthly_change, 2),
            'rsi': rsi,
            'trend_type': trend_type,
            'score': score,
            'action': action,
            'action_color': action_color,
            'next_prediction': round(next_prediction, 2),
            'pred_change': round(pred_change, 2),
            'sparkline': sparkline,
            'volume': info.get('volume', 0) or 0,
            'market_cap': market_cap,
            'market_cap_str': mc_str,
            'pe_ratio': round(info.get('trailingPE', 0) or 0, 2),
            'dividend_yield': round((info.get('dividendYield', 0) or 0) * 100, 2),
            '52w_high': round(info.get('fiftyTwoWeekHigh', 0) or 0, 2),
            '52w_low': round(info.get('fiftyTwoWeekLow', 0) or 0, 2),
            'avg_volume': info.get('averageVolume', 0) or 0,
            'beta': round(info.get('beta', 0) or 0, 2),
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_stocks_parallel(symbols, max_workers=20):
    """Fetch multiple stocks at the same time using threads"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data, sym): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result:
                results.append(result)
    return results


@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    return render_template('stock_detail.html', symbol=symbol.upper())

@app.route('/all-stocks')
def all_stocks():
    return render_template('all_stocks.html')

@app.route('/api/stocks/top')
def api_top_stocks():
    # Fetch first 50 in parallel, return top 10 by score
    results = fetch_stocks_parallel(TOP_500_STOCKS[:50], max_workers=20)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results[:10]})

@app.route('/api/stocks/all')
def api_all_stocks():
    # Fetch all in parallel with 30 workers
    results = fetch_stocks_parallel(TOP_500_STOCKS, max_workers=30)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results})

# ── NEW: Batch endpoint so All Stocks page loads progressively ──
@app.route('/api/stocks/batch')
def api_stocks_batch():
    offset = int(freq.args.get('offset', 0))
    limit  = int(freq.args.get('limit', 50))
    batch  = TOP_500_STOCKS[offset:offset + limit]
    results = fetch_stocks_parallel(batch, max_workers=20)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({
        'stocks':    results,
        'total':     len(TOP_500_STOCKS),
        'offset':    offset,
        'has_more':  offset + limit < len(TOP_500_STOCKS)
    })

@app.route('/api/stock/<symbol>')
def api_stock_detail(symbol):
    data = fetch_stock_data(symbol.upper())
    if not data:
        return jsonify({'error': 'Stock not found'}), 404
    return jsonify(data)

@app.route('/api/stock/<symbol>/history')
def api_stock_history(symbol):
    try:
        stock = yf.Ticker(symbol.upper())
        hist = stock.history(period="6mo")
        if hist.empty:
            return jsonify({'error': 'No data'}), 404
        data = []
        for date, row in hist.iterrows():
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        return jsonify({'history': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)