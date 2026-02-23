from flask import Flask, render_template, jsonify, request as freq
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

# ── Twelve Data API ───────────────────────────────────────────────────────────
# Get your free key at https://twelvedata.com/register
# Add it to Azure App Settings as:  TWELVE_DATA_KEY = your_key_here
TWELVE_API_KEY = os.environ.get('TWELVE_DATA_KEY', 'demo')
TWELVE_BASE    = "https://api.twelvedata.com"

# Rate-limit: free tier = 8 requests/minute, 800/day
# Keep semaphore at 4 to stay safely under the per-minute limit
_semaphore = threading.Semaphore(4)

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
    'CINF','ERIE','RLI','WRB','AXS','RNR','MKL','ACGL','FAF',
    # Healthcare
    'JNJ','UNH','LLY','ABBV','MRK','PFE','TMO','ABT','DHR','SYK',
    'BSX','MDT','EW','ISRG','ZTS','VRTX','REGN','GILD','BIIB','MRNA',
    'BDX','BAX','HOLX','DXCM','PODD','INCY','ALNY','HCA','HUM','ELV',
    'CI','CVS','MCK','ABC','CAH','IQV','CRL','MEDP','ICLR',
    'AMGN','ILMN','IONS','BMRN','EXEL','SRPT','ACAD','JAZZ',
    # Consumer Staples
    'PG','KO','PEP','MDLZ','GIS','K','CPB','SJM','CAG','MKC',
    'PM','MO','STZ','TAP','WMT','COST','TGT','KR',
    'SYY','PFGC','USFD','CASY','SFM',
    # Consumer Discretionary
    'HD','LOW','TJX','ROST','BURL','FIVE','NKE','SBUX','MCD','YUM',
    'QSR','DPZ','CMG','TXRH','DRI','EAT',
    'DIS','CMCSA','LYV',
    'F','GM','APTV','LEA','BWA',
    'EBAY','ETSY','BBY','KSS','M','JWN','GPS','ANF',
    'AEO','URBN','PVH','RL','TPR','VFC','SKX','CROX',
    # Industrials
    'GE','HON','MMM','CAT','DE','BA','RTX','LMT','NOC','GD',
    'UPS','FDX','XPO','ODFL','SAIA','KNX','WERN','JBHT','CHRW','EXPD',
    'EMR','ETN','PH','ROK','AME','GNRC','XYL','IR','TT','CARR',
    'OTIS','TDG','HEI','AXON','LDOS','SAIC','BAH',
    'WM','RSG','WCN','CLH','SRCL',
    'UNP','CSX','NSC','CP','WAB',
    # Energy
    'XOM','CVX','COP','EOG','DVN','MPC','PSX','VLO','HES',
    'OXY','APA','MRO','SWN','RRC','AR','EQT',
    'SLB','HAL','BKR',
    'OKE','WMB','KMI','EPD','ET','TRGP','LNG',
    'NEE','SO','DUK','AEP','EXC','SRE','PCG','ED','XEL','WEC',
    'ETR','ES','PEG','EIX','PPL','AEE','DTE','CMS',
    # Materials
    'LIN','APD','SHW','ECL','PPG','EMN','CE','DD','DOW','LYB',
    'NEM','FCX','SCCO','NUE','STLD',
    # Real Estate
    'PLD','AMT','CCI','EQIX','PSA','O','DLR','SPG','EQR','AVB',
    'ESS','MAA','UDR','CPT','NNN','VICI','GLPI','WPC','ADC',
    'HST','IRM','CUBE','EXR',
    # Communication Services
    'T','VZ','TMUS','CHTR','WBD','PARA','FOX','FOXA',
    'SPOT','PINS','SNAP','MTCH','EXPE','BKNG',
    'ABNB','LYFT','UBER','DASH','RBLX','TTWO','EA',
    # Airlines & Travel
    'DAL','UAL','AAL','LUV','ALK','JBLU','CCL','RCL',
    'NCLH','HLT','MAR',
    # Autos & EV
    'RIVN','LCID','NIO','HOG','RACE','TM','HMC',
    # Semiconductors Extended
    'ADI','MCHP','SWKS','QRVO','MPWR','ENTG','AMBA',
    # Fintech Extended
    'SQ','AFRM','UPST','SOFI',
    # Misc
    'BRK-B','MMC','AON','WTW','VRSK','CSGP','ANSS','TYL','CTSH',
    'INFY','GLOB',
]

# Deduplicate
seen  = set()
unique = []
for s in TOP_500_STOCKS:
    if s not in seen:
        seen.add(s)
        unique.append(s)
TOP_500_STOCKS = unique


# ── Helper calculations ───────────────────────────────────────────────────────

def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / loss
        rsi   = 100 - (100 / (1 + rs))
        val   = rsi.iloc[-1]
        return round(float(val), 1) if not pd.isna(val) else 50.0
    except:
        return 50.0

def calculate_trend_type(prices):
    try:
        if len(prices) < 50:
            return "Neutral"
        ma20    = prices.rolling(window=20).mean().iloc[-1]
        ma50    = prices.rolling(window=50).mean().iloc[-1]
        current = prices.iloc[-1]
        if current > ma20 > ma50:   return "Strong Uptrend"
        elif current > ma20:        return "Uptrend"
        elif current < ma20 < ma50: return "Strong Downtrend"
        elif current < ma20:        return "Downtrend"
        return "Neutral"
    except:
        return "Neutral"

def calculate_score(rsi, trend_type, daily_change):
    score = 50
    if rsi < 30:   score += 20
    elif rsi > 70: score -= 20
    if   "Strong Uptrend"   in trend_type: score += 30
    elif "Uptrend"          in trend_type: score += 15
    elif "Strong Downtrend" in trend_type: score -= 30
    elif "Downtrend"        in trend_type: score -= 15
    if   daily_change >  2: score += 20
    elif daily_change >  0: score += 10
    elif daily_change < -2: score -= 20
    elif daily_change <  0: score -= 10
    return max(0, min(100, int(score)))

def predict_next_value(prices):
    try:
        prices = prices.dropna()
        if len(prices) < 5:
            return float(prices.iloc[-1])
        X = np.array(range(len(prices))).reshape(-1, 1)
        y = prices.values
        model = LinearRegression()
        model.fit(X, y)
        result = float(model.predict([[len(prices)]])[0])
        return result if not np.isnan(result) else float(prices.iloc[-1])
    except:
        return float(prices.iloc[-1])

def fmt_market_cap(mc):
    if mc >= 1e12: return f"${mc/1e12:.2f}T"
    if mc >= 1e9:  return f"${mc/1e9:.2f}B"
    if mc >  0:    return f"${mc/1e6:.2f}M"
    return "N/A"


# ── Twelve Data API caller ────────────────────────────────────────────────────

def td_get(endpoint, params):
    """Make a Twelve Data API call with retries and rate-limit handling."""
    params['apikey'] = TWELVE_API_KEY
    for attempt in range(3):
        try:
            with _semaphore:
                time.sleep(random.uniform(0.1, 0.3))
                r = requests.get(f"{TWELVE_BASE}{endpoint}", params=params, timeout=15)
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"Rate limited on {endpoint}, waiting {wait}s...")
                time.sleep(wait)
                continue
            return r.json()
        except Exception as e:
            print(f"TD API error ({endpoint}): {e}")
            time.sleep(2 ** attempt)
    return {}


# ── Core stock fetcher ────────────────────────────────────────────────────────

def fetch_stock_data(symbol):
    """Fetch all data for one symbol using Twelve Data."""
    try:
        # 1. Time series — 90 days of daily closes for RSI, trend, sparkline
        ts_resp = td_get("/time_series", {
            "symbol":     symbol,
            "interval":   "1day",
            "outputsize": 90,
            "type":       "stock",
        })

        if ts_resp.get("status") == "error" or "values" not in ts_resp:
            print(f"No time series for {symbol}: {ts_resp.get('message', '')}")
            return None

        values = ts_resp["values"]   # newest first from API
        if len(values) < 2:
            return None

        # Reverse so index 0 = oldest, last = most recent
        closes_raw = [float(v["close"]) for v in reversed(values)]
        closes     = pd.Series(closes_raw)

        current_price  = closes.iloc[-1]
        previous_close = closes.iloc[-2]
        daily_change   = ((current_price - previous_close) / previous_close) * 100
        change         = current_price - previous_close

        rsi             = calculate_rsi(closes)
        trend_type      = calculate_trend_type(closes)
        next_prediction = predict_next_value(closes)
        score           = calculate_score(rsi, trend_type, daily_change)

        if score >= 70:   action, action_color = "BUY",  "green"
        elif score >= 40: action, action_color = "HOLD", "orange"
        else:             action, action_color = "SELL", "red"

        week_ago       = float(closes.iloc[-6])  if len(closes) >= 6  else previous_close
        month_ago      = float(closes.iloc[-22]) if len(closes) >= 22 else previous_close
        weekly_change  = ((current_price - week_ago)  / week_ago)  * 100
        monthly_change = ((current_price - month_ago) / month_ago) * 100
        pred_change    = ((next_prediction - current_price) / current_price) * 100
        sparkline      = [round(float(v), 2) for v in closes.tail(30).tolist()]

        # 2. Quote — volume, 52w high/low, name
        quote      = td_get("/quote", {"symbol": symbol})
        name       = quote.get("name", symbol)
        volume     = int(float(quote.get("volume", 0) or 0))
        avg_volume = int(float(quote.get("average_volume", 0) or 0))
        w52        = quote.get("fifty_two_week", {}) or {}
        week52_high = round(float(w52.get("high", 0) or 0), 2)
        week52_low  = round(float(w52.get("low",  0) or 0), 2)

        # 3. Statistics — market cap, PE, beta, dividend yield
        stats      = td_get("/statistics", {"symbol": symbol})
        stat_vals  = stats.get("statistics", {}) or {}
        val_metrics = stat_vals.get("valuations_metrics", {}) or {}
        stk_stats   = stat_vals.get("stock_statistics", {}) or {}
        div_splits  = stat_vals.get("dividends_and_splits", {}) or {}

        market_cap    = float(val_metrics.get("market_capitalization", 0) or 0)
        pe_ratio      = round(float(val_metrics.get("trailing_pe", 0) or 0), 2)
        beta          = round(float(stk_stats.get("beta", 0) or 0), 2)
        div_yield_raw = div_splits.get("forward_annual_dividend_yield", 0)
        div_yield     = round(float(div_yield_raw or 0) * 100, 2)

        # TD returns market cap in full dollars
        mc_str = fmt_market_cap(market_cap)

        return {
            'symbol':             symbol,
            'name':               name,
            'sector':             quote.get("exchange", "N/A"),
            'price':              round(current_price,  2),
            'previous_close':     round(previous_close, 2),
            'change':             round(change,         2),
            'daily_percentage':   round(daily_change,   2),
            'weekly_percentage':  round(weekly_change,  2),
            'monthly_percentage': round(monthly_change, 2),
            'rsi':                rsi,
            'trend_type':         trend_type,
            'score':              score,
            'action':             action,
            'action_color':       action_color,
            'next_prediction':    round(next_prediction, 2),
            'pred_change':        round(pred_change,     2),
            'sparkline':          sparkline,
            'volume':             volume,
            'market_cap':         market_cap,
            'market_cap_str':     mc_str,
            'pe_ratio':           pe_ratio,
            'dividend_yield':     div_yield,
            '52w_high':           week52_high,
            '52w_low':            week52_low,
            'avg_volume':         avg_volume,
            'beta':               beta,
        }

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_stocks_parallel(symbols, max_workers=4):
    """
    Fetch stocks in parallel.
    max_workers=4 keeps us safely within Twelve Data free tier (8 req/min).
    Each stock makes 3 API calls (time_series + quote + statistics).
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
    # Only fetch first 20 for the dashboard top-10 to save API credits
    results = fetch_stocks_parallel(TOP_500_STOCKS[:20], max_workers=4)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results[:10]})

@app.route('/api/stocks/all')
def api_all_stocks():
    results = fetch_stocks_parallel(TOP_500_STOCKS, max_workers=4)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results})

@app.route('/api/stocks/batch')
def api_stocks_batch():
    offset  = int(freq.args.get('offset', 0))
    limit   = int(freq.args.get('limit',  20))   # 20 per batch on free tier
    batch   = TOP_500_STOCKS[offset:offset + limit]
    results = fetch_stocks_parallel(batch, max_workers=4)
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
        resp = td_get("/time_series", {
            "symbol":     symbol.upper(),
            "interval":   "1day",
            "outputsize": 130,
            "type":       "stock",
        })
        if "values" not in resp:
            return jsonify({'error': 'No data'}), 404
        data = []
        for v in reversed(resp["values"]):
            data.append({
                'date':   v["datetime"],
                'open':   round(float(v["open"]),  2),
                'high':   round(float(v["high"]),  2),
                'low':    round(float(v["low"]),   2),
                'close':  round(float(v["close"]), 2),
                'volume': int(float(v.get("volume", 0)))
            })
        return jsonify({'history': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
