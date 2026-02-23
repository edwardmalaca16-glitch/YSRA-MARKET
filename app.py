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
import yfinance as yf
from functools import lru_cache
warnings.filterwarnings('ignore')

app = Flask(name)

# ── Twelve Data API ───────────────────────────────────────────────────────────
# Get your free key at https://twelvedata.com/register
# Add it to Azure App Settings as:  TWELVE_DATA_KEY = your_key_here
TWELVE_API_KEY = os.environ.get('TWELVE_DATA_KEY', 'demo')
TWELVE_BASE    = "https://api.twelvedata.com"

# Rate-limit: free tier = 8 requests/minute, 800/day
# Keep semaphore at 2 to stay safely under the per-minute limit
_semaphore = threading.Semaphore(2)

# Simple cache for Twelve Data calls
_cache = {}
_cache_ttl = 300  # 5 minutes

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


# ── Twelve Data API caller (with caching) ────────────────────────────────────

def td_get_cached(endpoint, params):
    """Cached Twelve Data API call with rate-limit handling."""
    # Create cache key
    cache_key = f"{endpoint}:{params.get('symbol', '')}:{params.get('interval', '')}"
    
    # Check cache
    cached = _cache.get(cache_key)
    if cached and time.time() - cached['time'] < _cache_ttl:
        print(f"Cache hit for {cache_key}")
        return cached['data']
    
    # Make API call
    params['apikey'] = TWELVE_API_KEY
    for attempt in range(3):
        try:
            with _semaphore:
                time.sleep(random.uniform(2, 4))  # Delay between requests
                r = requests.get(f"{TWELVE_BASE}{endpoint}", params=params, timeout=15)
            
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"Rate limited on {endpoint}, waiting {wait}s...")
                time.sleep(wait)
                continue
                
            if r.status_code == 200:
                data = r.json()
                # Cache the result
                _cache[cache_key] = {'data': data, 'time': time.time()}
                return data
            else:
                print(f"Error {r.status_code} from Twelve Data: {r.text[:100]}")
                return {}
                
        except Exception as e:
            print(f"TD API error ({endpoint}): {e}")
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
            else:
                return {}
    return {}


# ── yFinance stock fetcher (primary) ─────────────────────────────────────────

def fetch_stock_data_yfinance(symbol):
    """Fetch stock data using yfinance as primary source."""
    try:
        print(f"Fetching {symbol} with yfinance...")
        stock = yf.Ticker(symbol)
        
        # Get historical data (3 months of daily data)
        hist = stock.history(period="3mo")
        if hist.empty:
            print(f"No yfinance data for {symbol}, will try Twelve Data")
            return None
        
        # Get info
        info = stock.info
        if not info:
            print(f"No info for {symbol}")
            return None
        
        # Calculate metrics from historical data
        closes = hist['Close']
        current_price = closes.iloc[-1]
        previous_close = closes.iloc[-2] if len(closes) > 1 else current_price
        daily_change = ((current_price - previous_close) / previous_close) * 100
        
        # Weekly and monthly changes
        week_ago = closes.iloc[-6] if len(closes) >= 6 else previous_close
        month_ago = closes.iloc[-22] if len(closes) >= 22 else previous_close
        weekly_change = ((current_price - week_ago) / week_ago) * 100
        monthly_change = ((current_price - month_ago) / month_ago) * 100
        
        # Technical indicators
        rsi = calculate_rsi(closes)
        trend_type = calculate_trend_type(closes)
        next_prediction = predict_next_value(closes)
        pred_change = ((next_prediction - current_price) / current_price) * 100
        
        # Score and action
        score = calculate_score(rsi, trend_type, daily_change)
        if score >= 70:
            action, action_color = "BUY", "green"
        elif score >= 40:
            action, action_color = "HOLD", "orange"
        else:
            action, action_color = "SELL", "red"
        
        # Sparkline (last 30 days)
        sparkline = [round(float(x), 2) for x in closes.tail(30).tolist()]
        
        # Get metrics from info
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio:
            pe_ratio = round(pe_ratio, 2)
        
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield:
            dividend_yield = round(dividend_yield * 100, 2)
        else:
            dividend_yield = 0
        
        beta = info.get('beta', 0)
        if beta:
            beta = round(beta, 2)
        
        volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 0)
        week52_high = info.get('fiftyTwoWeekHigh', 0)
        week52_low = info.get('fiftyTwoWeekLow', 0)
        
        # Format market cap
        mc_str = fmt_market_cap(market_cap)
        
        # Get company name
        name = info.get('longName', symbol)
        sector = info.get('sector', 'N/A')
        
        return {
            'symbol': symbol,
            'name': name,
            'sector': sector,
            'price': round(current_price, 2),
            'previous_close': round(previous_close, 2),
            'change': round(current_price - previous_close, 2),
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
            'volume': volume,
            'market_cap': market_cap,
            'market_cap_str': mc_str,
            'pe_ratio': pe_ratio if pe_ratio else 0,
            'dividend_yield': dividend_yield,
            '52w_high': round(week52_high, 2) if week52_high else 0,
            '52w_low': round(week52_low, 2) if week52_low else 0,
            'avg_volume': avg_volume,
            'beta': beta if beta else 0,
            'data_source': 'yfinance'
        }
        
    except Exception as e:
        print(f"Error in yfinance fetch for {symbol}: {e}")
        return None


# ── Twelve Data stock fetcher (fallback) ─────────────────────────────────────

def fetch_stock_data_twelvedata(symbol):
    """Fetch stock data using Twelve Data as fallback."""
    try:
        print(f"Fetching {symbol} with Twelve Data (fallback)...")
        
        # 1. Time series — 90 days of daily closes
        ts_resp = td_get_cached("/time_series", {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 90,
            "type": "stock",
        })

        if not ts_resp or ts_resp.get("status") == "error" or "values" not in ts_resp:
            print(f"No time series for {symbol}")
            return None

        values = ts_resp["values"]
        if len(values) < 2:
            return None

        # Reverse so index 0 = oldest, last = most recent
        closes_raw = [float(v["close"]) for v in reversed(values)]
        closes = pd.Series(closes_raw)

        current_price = closes.iloc[-1]
        previous_close = closes.iloc[-2]
        daily_change = ((current_price - previous_close) / previous_close) * 100
        change = current_price - previous_close

        # Technical indicators
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

        #Weekly and monthly changes
        week_ago = float(closes.iloc[-6]) if len(closes) >= 6 else previous_close
        month_ago = float(closes.iloc[-22]) if len(closes) >= 22 else previous_close
        weekly_change = ((current_price - week_ago) / week_ago) * 100
        monthly_change = ((current_price - month_ago) / month_ago) * 100
        pred_change = ((next_prediction - current_price) / current_price) * 100
        sparkline = [round(float(v), 2) for v in closes.tail(30).tolist()]

        # 2. Quote — volume, 52w high/low, name
        quote = td_get_cached("/quote", {"symbol": symbol})
        name = quote.get("name", symbol)
        volume = int(float(quote.get("volume", 0) or 0))
        avg_volume = int(float(quote.get("average_volume", 0) or 0))
        w52 = quote.get("fifty_two_week", {}) or {}
        week52_high = round(float(w52.get("high", 0) or 0), 2)
        week52_low = round(float(w52.get("low", 0) or 0), 2)

        # 3. Statistics — market cap, PE, beta, dividend yield
        stats = td_get_cached("/statistics", {"symbol": symbol})
        stat_vals = stats.get("statistics", {}) or {}
        val_metrics = stat_vals.get("valuations_metrics", {}) or {}
        stk_stats = stat_vals.get("stock_statistics", {}) or {}
        div_splits = stat_vals.get("dividends_and_splits", {}) or {}

        market_cap = float(val_metrics.get("market_capitalization", 0) or 0)
        pe_ratio = round(float(val_metrics.get("trailing_pe", 0) or 0), 2)
        beta = round(float(stk_stats.get("beta", 0) or 0), 2)
        div_yield_raw = div_splits.get("forward_annual_dividend_yield", 0)
        div_yield = round(float(div_yield_raw or 0) * 100, 2)

        mc_str = fmt_market_cap(market_cap)

        return {
            'symbol': symbol,
            'name': name,
            'sector': quote.get("exchange", "N/A"),
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
            'volume': volume,
            'market_cap': market_cap,
            'market_cap_str': mc_str,
            'pe_ratio': pe_ratio,
            'dividend_yield': div_yield,
            '52w_high': week52_high,
            '52w_low': week52_low,
            'avg_volume': avg_volume,
            'beta': beta,
            'data_source': 'twelvedata'
        }

    except Exception as e:
        print(f"Error fetching {symbol} with Twelve Data: {e}")
        return None


# ── Main stock fetcher (tries yfinance first, then Twelve Data) ─────────────

def fetch_stock_data(symbol):
    """Fetch stock data trying yfinance first, then falling back to Twelve Data."""
    # Try yfinance first
    result = fetch_stock_data_yfinance(symbol)
    if result:
        return result
    
    # If yfinance fails, try Twelve Data
    print(f"yfinance failed for {symbol}, trying Twelve Data...")
    time.sleep(1)  # Small delay before fallback
    return fetch_stock_data_twelvedata(symbol)


def fetch_stocks_parallel(symbols, max_workers=2):
    """
    Fetch stocks in parallel with reduced workers to avoid rate limiting.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data, sym): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
                    print(f"Successfully fetched {result['symbol']} from {result.get('data_source', 'unknown')}")
            except Exception as e:
                symbol = future_to_symbol[future]
                print(f"Failed to fetch {symbol}: {e}")
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
    # Fetch only 5 stocks for the dashboard to save API calls
    results = fetch_stocks_parallel(TOP_500_STOCKS[:5], max_workers=2)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results[:10]})  # Still return top 10 even if we only fetched 5

@app.route('/api/stocks/all')
def api_all_stocks():
    # Be careful with this - it will make many API calls
    # Consider implementing pagination instead
    results = fetch_stocks_parallel(TOP_500_STOCKS[:20], max_workers=2)  # Only fetch first 20
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results})

@app.route('/api/stocks/batch')
def api_stocks_batch():
    offset = int(freq.args.get('offset', 0))
    limit = int(freq.args.get('limit', 5))  # Small batches to avoid rate limiting
    batch = TOP_500_STOCKS[offset:offset + limit]
    results = fetch_stocks_parallel(batch, max_workers=2)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({
        'stocks': results,
        'total': len(TOP_500_STOCKS),
        'offset': offset,
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
        # Try yfinance first for history
        stock = yf.Ticker(symbol.upper())
        hist = stock.history(period="6mo")
        
        if not hist.empty:
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
            return jsonify({'history': data, 'source': 'yfinance'})
        
        # Fallback to Twelve Data
        resp = td_get_cached("/time_series", {
            "symbol": symbol.upper(),
            "interval": "1day",
            "outputsize": 130,
            "type": "stock",
        })
        
        if "values" not in resp:
            return jsonify({'error': 'No data'}), 404
            
        data = []
        for v in reversed(resp["values"]):
            data.append({
                'date': v["datetime"],
                'open': round(float(v["open"]), 2),
                'high': round(float(v["high"]), 2),
                'low': round(float(v["low"]), 2),
                'close': round(float(v["close"]), 2),
                'volume': int(float(v.get("volume", 0)))
            })
        return jsonify({'history': data, 'source': 'twelvedata'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint to verify API connectivity."""
    # Test yfinance
    yf_working = False
    td_working = False
    
    try:
        test_stock = yf.Ticker('AAPL')
        test_hist = test_stock.history(period="1d")
        yf_working = not test_hist.empty
    except:
        pass
    
    try:
        test_resp = td_get_cached("/quote", {"symbol": "AAPL"})
        td_working = bool(test_resp and test_resp.get('name'))
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'yfinance_working': yf_working,
        'twelvedata_working': td_working,
        'twelvedata_key': 'configured' if TWELVE_API_KEY != 'demo' else 'demo'
    })

if name == 'main':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting stock dashboard on port {port}")
    print(f"Twelve Data API key: {'Configured' if TWELVE_API_KEY != 'demo' else 'Using DEMO key (limited)'}")
    print(f"Health check available at /api/health")
    app.run(host='0.0.0.0', port=port, debug=False)
