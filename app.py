from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

TOP_500_STOCKS = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','JPM','JNJ',
    'V','PG','UNH','HD','DIS','MA','BAC','NFLX','ADBE','CRM',
    'CMCSA','XOM','VZ','KO','PEP','INTC','CSCO','WMT','PFE','TMO',
    'ABT','NKE','MRK','ORCL','ACN','DHR','MCD','TXN','NEE','WFC',
    'AMD','BA','QCOM','COST','UPS','MS','IBM','CAT','GE','CVX',
    'LLY','ABBV','RTX','HON','AMGN','SBUX','GS','BLK','MMM','SPGI',
    'AXP','ISRG','DE','NOW','LRCX','MU','ADI','PANW','KLAC','AMAT',
    'REGN','SYK','BSX','VRTX','MDT','ZTS','CI','HCA','ELV','HUM',
    'MDLZ','GILD','MCK','AIG','AFL','TRV','ALL','MET','PRU','PGR',
    'SO','DUK','AEP','EXC','SRE','PCG','ED','XEL','WEC','ETR',
    'PLD','AMT','CCI','EQIX','PSA','O','DLR','SPG','EQR','AVB',
    'LIN','APD','SHW','ECL','PPG','EMN','CE','DD','DOW','LYB',
    'F','GM','TM','HMC','STLA','HOG','RACE','LCID','RIVN','NIO',
    'DAL','UAL','AAL','LUV','ALK','HA','JBLU','SAVE','SKYW','MESA'
]

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if not pd.isna(val) else 50.0

def calculate_trend_type(prices):
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
    if len(prices) < 5:
        return float(prices.iloc[-1])
    X = np.array(range(len(prices))).reshape(-1, 1)
    y = prices.values
    model = LinearRegression()
    model.fit(X, y)
    return float(model.predict([[len(prices)]])[0])

def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="3mo")
        if hist.empty or len(hist) < 2:
            return None

        closes = hist['Close']
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

        # Weekly/monthly changes
        week_ago = float(closes.iloc[-6]) if len(closes) >= 6 else previous_close
        month_ago = float(closes.iloc[-22]) if len(closes) >= 22 else previous_close
        weekly_change = ((current_price - week_ago) / week_ago) * 100
        monthly_change = ((current_price - month_ago) / month_ago) * 100

        # Prediction change
        pred_change = ((next_prediction - current_price) / current_price) * 100

        # 30-day history for sparkline
        sparkline = [round(float(v), 2) for v in closes.tail(30).tolist()]

        market_cap = info.get('marketCap', 0) or 0
        mc_str = ""
        if market_cap >= 1e12:
            mc_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            mc_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 0:
            mc_str = f"${market_cap/1e6:.2f}M"

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
    results = []
    for sym in TOP_500_STOCKS[:30]:  # fetch first 30, return top 10
        d = fetch_stock_data(sym)
        if d:
            results.append(d)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results[:10]})

@app.route('/api/stocks/all')
def api_all_stocks():
    results = []
    for sym in TOP_500_STOCKS:
        d = fetch_stock_data(sym)
        if d:
            results.append(d)
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'stocks': results})

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