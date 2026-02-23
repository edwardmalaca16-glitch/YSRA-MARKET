from flask import Flask, render_template, jsonify, request as freq
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import threading
import time
import random
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Yahoo Finance session with browser-like headers ──────────────────────────
_session_lock = threading.Lock()
_yf_session   = None
_crumb        = None
_session_ts   = 0
SESSION_TTL   = 55 * 60   # refresh every 55 minutes

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

def _build_session():
    """Create a requests session that looks like a real browser and fetch a Yahoo crumb."""
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    # Hit the consent page first (needed outside US)
    try:
        s.get("https://finance.yahoo.com", timeout=10)
    except Exception:
        pass
    # Fetch crumb
    try:
        r = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        crumb = r.text.strip() if r.status_code == 200 and r.text.strip() else None
    except Exception:
        crumb = None
    return s, crumb

def get_yf_session():
    """Return a valid (session, crumb) pair, refreshing if stale."""
    global _yf_session, _crumb, _session_ts
    with _session_lock:
        if _yf_session is None or (time.time() - _session_ts) > SESSION_TTL:
            _yf_session, _crumb = _build_session()
            _session_ts = time.time()
        return _yf_session, _crumb


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


# ── Rate-limit throttle: max N concurrent Yahoo requests ─────────────────────
_yahoo_semaphore = threading.Semaphore(5)   # only 5 simultaneous Yahoo calls

def fetch_stock_data(symbol, retries=3):
    """Fetch data for one symbol, reusing the shared browser session."""
    for attempt in range(retries):
        try:
            with _yahoo_semaphore:
                # Small random delay to avoid thundering-herd rate limits
                time.sleep(random.uniform(0.1, 0.5))

                session, crumb = get_yf_session()
                stock = yf.Ticker(symbol, session=session)

                # Pass crumb if we have one
                hist_kwargs = {"period": "3mo"}
                hist = stock.history(**hist_kwargs)

                if hist.empty or len(hist) < 2:
                    return None

                info = stock.info  # fetch after history to reuse cookies

            closes = hist['Close'].dropna()
            if len(closes) < 2:
                return None

            current_price   = float(closes.iloc[-1])
            previous_close  = float(closes.iloc[-2])
            daily_change    = ((current_price - previous_close) / previous_close) * 100
            change          = current_price - previous_close
            rsi             = calculate_rsi(closes)
            trend_type      = calculate_trend_type(closes)
            next_prediction = predict_next_value(closes)
            score           = calculate_score(rsi, trend_type, daily_change)

            if score >= 70:
                action, action_color = "BUY",  "green"
            elif score >= 40:
                action, action_color = "HOLD", "orange"
            else:
                action, action_color = "SELL", "red"

            week_ago       = float(closes.iloc[-6])  if len(closes) >= 6  else previous_close
            month_ago      = float(closes.iloc[-22]) if len(closes) >= 22 else previous_close
            weekly_change  = ((current_price - week_ago)  / week_ago)  * 100
            monthly_change = ((current_price - month_ago) / month_ago) * 100
            pred_change    = ((next_prediction - current_price) / current_price) * 100
            sparkline      = [round(float(v), 2) for v in closes.tail(30).tolist()]

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
                'symbol':          symbol,
                'name':            info.get('longName', symbol),
                'sector':          info.get('sector', 'N/A'),
                'price':           round(current_price, 2),
                'previous_close':  round(previous_close, 2),
                'change':          round(change, 2),
                'daily_percentage':  round(daily_change,    2),
                'weekly_percentage': round(weekly_change,   2),
                'monthly_percentage':round(monthly_change,  2),
                'rsi':             rsi,
                'trend_type':      trend_type,
                'score':           score,
                'action':          action,
                'action_color':    action_color,
                'next_prediction': round(next_prediction, 2),
                'pred_change':     round(pred_change,     2),
                'sparkline':       sparkline,
                'volume':          info.get('volume',          0) or 0,
                'market_cap':      market_cap,
                'market_cap_str':  mc_str,
                'pe_ratio':        round(info.get('trailingPE',       0) or 0, 2),
                'dividend_yield':  round((info.get('dividendYield',   0) or 0) * 100, 2),
                '52w_high':        round(info.get('fiftyTwoWeekHigh', 0) or 0, 2),
                '52w_low':         round(info.get('fiftyTwoWeekLow',  0) or 0, 2),
                'avg_volume':      info.get('averageVolume', 0) or 0,
                'beta':            round(info.get('beta', 0) or 0, 2),
            }

        except Exception as e:
            err = str(e)
            print(f"Error fetching {symbol} (attempt {attempt+1}): {err}")
            # On rate-limit, wait longer before retrying
            if "429" in err or "Too Many" in err or "Rate" in err:
                time.sleep(2 ** attempt + random.uniform(0, 1))  # exponential back-off
            elif "401" in err or "Unauthorized" in err or "Crumb" in err:
                # Force session refresh and retry
                with _session_lock:
                    global _yf_session, _crumb, _session_ts
                    _yf_session = None
                time.sleep(1)
            else:
                break   # non-retryable error

    return None


def fetch_stocks_parallel(symbols, max_workers=8):
    """
    Fetch stocks in parallel.
    Keep max_workers LOW (≤8) to avoid Yahoo rate-limits on Azure.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data, sym): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result:
                results.append(result)
    return results


# ── Routes ────────────────────────────────────────────────────────────────────

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
    results = fetch_stocks_parallel(TOP_500_STOCKS[:50], max_workers=8)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results[:10]})

@app.route('/api/stocks/all')
def api_all_stocks():
    results = fetch_stocks_parallel(TOP_500_STOCKS, max_workers=8)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results})

@app.route('/api/stocks/batch')
def api_stocks_batch():
    offset  = int(freq.args.get('offset', 0))
    limit   = int(freq.args.get('limit',  50))
    batch   = TOP_500_STOCKS[offset:offset + limit]
    results = fetch_stocks_parallel(batch, max_workers=8)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({
        'stocks':   results,
        'total':    len(TOP_500_STOCKS),
        'offset':   offset,
        'has_more': offset + limit < len(TOP_500_STOCKS)
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
        session, _ = get_yf_session()
        stock = yf.Ticker(symbol.upper(), session=session)
        hist  = stock.history(period="6mo")
        if hist.empty:
            return jsonify({'error': 'No data'}), 404
        data = []
        for date, row in hist.iterrows():
            data.append({
                'date':   date.strftime('%Y-%m-%d'),
                'open':   round(float(row['Open']),  2),
                'high':   round(float(row['High']),  2),
                'low':    round(float(row['Low']),   2),
                'close':  round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        return jsonify({'history': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
