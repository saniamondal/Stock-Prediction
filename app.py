import feedparser
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import plotly.graph_objs as go
from transformers import pipeline
from newspaper import Article
load_dotenv()

# ---------- VALID PERIOD/INTERVAL COMBOS ----------
VALID_COMBOS = {
    "1mo":  ["1m","2m","5m","15m","30m","60m","90m","1d"],
    "3mo":  ["1m","2m","5m","15m","30m","60m","90m","1d","1wk"],
    "6mo":  ["1d","1wk"],
    "1y":   ["1d","1wk","1mo"],
    "2y":   ["1d","1wk","1mo"],
    "5y":   ["1d","1wk","1mo"],
}

# ---------- LOAD FINBERT ----------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

# ---------- FULL ARTICLE FETCH ----------
def get_full_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:2000]
    except:
        return ""

# ---------- Fetch Data ----------
def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    ticker = ticker.strip().upper()
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    except Exception as e:
        return None, None, None, str(e)

    if data is None or data.empty:
        return None, None, None, "No data returned. The ticker may be too new for this period, or the period/interval combo is invalid."

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.astype(float)
    data.index = pd.to_datetime(data.index)
    data = data.dropna()

    if len(data) < 60:
        return None, None, None, f"Only {len(data)} rows returned. Need at least 60. Try a longer period like '1y' with '1d' interval."

    close = data['Close']

    try:
        data['rsi']         = ta.momentum.RSIIndicator(close=close).rsi()
        data['roc']         = ta.momentum.ROCIndicator(close=close).roc()
        stoch               = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=close)
        data['stoch_k']     = stoch.stoch()
        data['stoch_d']     = stoch.stoch_signal()
        macd                = ta.trend.MACD(close=close)
        data['macd']        = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff']   = macd.macd_diff()
        data['ma20']        = close.rolling(20).mean()
        data['ma50']        = close.rolling(50).mean()
        data['cci']         = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=close, window=14).cci()
        data['dpo']         = ta.trend.DPOIndicator(close=close).dpo()
        dmi                 = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=close, window=14)
        data['adx']         = dmi.adx()
        data['dmi_pos']     = dmi.adx_pos()
        data['dmi_neg']     = dmi.adx_neg()
        bb                  = ta.volatility.BollingerBands(close=close)
        data['bb_upper']    = bb.bollinger_hband()
        data['bb_mid']      = bb.bollinger_mavg()
        data['bb_lower']    = bb.bollinger_lband()
        data['returns']     = close.pct_change()
    except Exception as e:
        return None, None, None, f"Indicator calculation failed: {e}"

    data = data.dropna()
    if data.empty:
        return None, None, None, "Data became empty after indicator calculation. Try a longer period."

    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    latest_price = float(data['Close'].iloc[-1])
    summary = {
        "latest_price": latest_price,
        "high":         float(data['High'].max()),
        "low":          float(data['Low'].min()),
        "avg_volume":   float(data['Volume'].mean()),
    }
    filtered_info = {
        "name":      info.get("longName", ticker),
        "sector":    info.get("sector", "N/A"),
        "marketCap": info.get("marketCap", 0),
        "peRatio":   info.get("trailingPE", 0),
        "currency":  info.get("currency", "INR"),
    }
    return data, summary, filtered_info, None

# ---------- Prediction (LSTM) ----------
def predict_price(df):
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    seq_len = 30
    if len(scaled) <= seq_len:
        x = np.arange(len(close_prices))
        slope, intercept = np.polyfit(x, close_prices.flatten(), 1)
        return float(slope * len(close_prices) + intercept)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    pred_scaled = model.predict(last_seq, verbose=0)
    return float(scaler.inverse_transform(pred_scaled)[0][0])

# ---------- Signal ----------
def generate_signal(rsi, macd, ma20, ma50, prev_ma20, prev_ma50):
    score = 0
    if rsi < 35:    score += 1
    elif rsi > 65:  score -= 1
    if macd > 0:    score += 1
    elif macd < 0:  score -= 1
    if ma20 > ma50:   score += 1
    elif ma20 < ma50: score -= 1
    if prev_ma20 < prev_ma50 and ma20 > ma50:   score += 2
    elif prev_ma20 > prev_ma50 and ma20 < ma50: score -= 2
    if score >= 2:    return "BUY"
    elif score <= -2: return "SELL"
    return "HOLD"

# ---------- News ----------
def fetch_news(ticker):
    try:
        query = ticker.replace(".NS", "").replace(".BO", "")
        url   = f"https://news.google.com/rss/search?q={query}+stock+India"
        feed  = feedparser.parse(url)
        return [{"title": e.title, "link": e.link, "summary": e.get("summary", "")}
                for e in feed.entries[:10]]
    except:
        return []

def summarize(title, raw_summary, llm):
    try:
        prompt = f"Summarize this news in 50-60 words.\nTitle: {title}\nDetails: {raw_summary}"
        result = llm.invoke(prompt)
        return result.content.strip() if hasattr(result, "content") else str(result)
    except:
        return title

# ---------- FINBERT Sentiment ----------
def analyze_sentiment(news):
    if not news:
        return 0
    model  = load_sentiment_model()
    scores = []
    for item in news:
        full_text = get_full_article(item.get("link", ""))
        if not full_text or len(full_text.strip()) < 50:
            full_text = f"{item.get('title','')} {item.get('summary','')}"
        text = full_text[:512]
        try:
            result = model(text)[0]
        except:
            continue
        label = result['label'].lower()
        score = result['score']
        if label == "positive":   scores.append(score)
        elif label == "negative": scores.append(-score)
        else:                     scores.append(0)
    return float(np.mean(scores)) if scores else 0

def sentiment_label(score):
    if score > 0.05:  return "🟢 Positive"
    if score < -0.05: return "🔴 Negative"
    return "🟡 Neutral"

# ---------- Currency Symbol ----------
def get_currency_symbol(currency: str) -> str:
    symbols = {
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CNY": "¥",
        "CAD": "C$",
        "AUD": "A$",
        "SGD": "S$",
        "HKD": "HK$",
    }
    return symbols.get(currency.upper(), currency + " ")

# ---------- LLM ----------
@st.cache_resource
def load_llm():
    llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", temperature=0.7, max_new_tokens=512)
    return ChatHuggingFace(llm=llm)

def generate_response(llm, query, summary, info, prediction, rsi, macd, signal, sentiment):
    template = load_prompt("stock_prompt_template.json")
    prompt   = template.invoke({
        "query": query, "name": info.get("name"), "sector": info.get("sector"),
        "price": summary.get("latest_price"), "prediction": prediction,
        "rsi": rsi, "macd": macd, "signal": signal, "sentiment": sentiment,
        "high": summary.get("high"), "low": summary.get("low"),
        "volume": summary.get("avg_volume"), "pe": info.get("peRatio")
    })
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)

# ---------- UI ----------
st.set_page_config(page_title="Indian Stock Dashboard", layout="wide")
st.title("📈 Indian Stock Trading Dashboard")

ticker   = st.text_input("Enter stock ticker (e.g. WAAREEENER.NS)")
query    = st.text_input("Ask your question")
period   = st.selectbox("Select period",   ['1mo','3mo','6mo','1y','2y','5y'])
interval = st.selectbox("Select interval", ['1d','1wk','1mo'])

# warn user about invalid combos before they click
if period in VALID_COMBOS and interval not in VALID_COMBOS[period]:
    st.warning(f"⚠️ '{interval}' interval is not valid for '{period}' period. Valid options: {VALID_COMBOS[period]}")

if st.button("Analyze"):
    if not ticker or not query:
        st.warning("Enter both ticker and question.")
    elif period in VALID_COMBOS and interval not in VALID_COMBOS[period]:
        st.error(f"Invalid period/interval combo. For '{period}', use one of: {VALID_COMBOS[period]}")
    else:
        with st.spinner("Running analysis..."):
            df, summary, info, err = fetch_stock_data(ticker, period, interval)
            if df is None:
                st.error(f"Data fetch failed: {err}")
                st.info("For newly listed stocks like WAAREEENER.NS, use *1y* period with *1d* interval.")
                st.stop()

            prediction = predict_price(df)
            latest     = df.iloc[-1]
            prev       = df.iloc[-2] if len(df) >= 2 else latest
            signal     = generate_signal(
                latest['rsi'], latest['macd'],
                latest['ma20'], latest['ma50'],
                prev['ma20'],  prev['ma50']
            )
            news      = fetch_news(ticker)
            sentiment = analyze_sentiment(news)
            llm       = load_llm()
            response  = generate_response(
                llm, query, summary, info,
                round(prediction, 2), round(float(latest['rsi']), 2),
                round(float(latest['macd']), 2), signal, round(sentiment, 2)
            )

            sym = get_currency_symbol(info.get("currency", "INR"))

            # ---- Metrics ----
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Price",      f"{sym}{round(summary['latest_price'], 2)}")
            col2.metric("RSI",        round(float(latest['rsi']), 2))
            col3.metric("Signal",     signal)
            col4.metric("ADX",        round(float(latest['adx']), 2))
            col5.metric("Prediction", f"{sym}{round(prediction, 2)}")

            # ---- Main Chart ----
            st.subheader("Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df['ma20'],     name='MA20',     line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['ma50'],     name='MA50',     line=dict(color='blue',   width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray',   width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray',   width=1, dash='dash'),
                                     fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.3,
                                 yaxis='y2', marker_color='lightblue'))
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                yaxis_title=f"Price ({sym})",
                yaxis2=dict(overlaying='y', side='right', title='Volume', showgrid=False),
                legend=dict(orientation='h', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Sub Charts ----
            c1, c2 = st.columns(2)
            with c1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash='dash', line_color='red')
                fig_rsi.add_hline(y=30, line_dash='dash', line_color='green')
                fig_rsi.update_layout(title='RSI', height=250, margin=dict(t=30, b=20))
                st.plotly_chart(fig_rsi, use_container_width=True)

                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd'],        name='MACD',   line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='orange')))
                fig_macd.add_trace(go.Bar(x=df.index,     y=df['macd_diff'],   name='Hist',   marker_color='gray', opacity=0.5))
                fig_macd.update_layout(title='MACD', height=250, margin=dict(t=30, b=20))
                st.plotly_chart(fig_macd, use_container_width=True)

            with c2:
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], name='%K', line=dict(color='blue')))
                fig_stoch.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], name='%D', line=dict(color='orange')))
                fig_stoch.add_hline(y=80, line_dash='dash', line_color='red')
                fig_stoch.add_hline(y=20, line_dash='dash', line_color='green')
                fig_stoch.update_layout(title='Stochastic', height=250, margin=dict(t=30, b=20))
                st.plotly_chart(fig_stoch, use_container_width=True)

                fig_adx = go.Figure()
                fig_adx.add_trace(go.Scatter(x=df.index, y=df['adx'],     name='ADX', line=dict(color='black')))
                fig_adx.add_trace(go.Scatter(x=df.index, y=df['dmi_pos'], name='+DI', line=dict(color='green')))
                fig_adx.add_trace(go.Scatter(x=df.index, y=df['dmi_neg'], name='-DI', line=dict(color='red')))
                fig_adx.update_layout(title='ADX / DMI', height=250, margin=dict(t=30, b=20))
                st.plotly_chart(fig_adx, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig_cci = go.Figure()
                fig_cci.add_trace(go.Scatter(x=df.index, y=df['cci'], name='CCI', line=dict(color='teal')))
                fig_cci.add_hline(y=100,  line_dash='dash', line_color='red')
                fig_cci.add_hline(y=-100, line_dash='dash', line_color='green')
                fig_cci.update_layout(title='CCI', height=250, margin=dict(t=30, b=20))
                st.plotly_chart(fig_cci, use_container_width=True)

            with c4:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=df.index, y=df['roc'], name='ROC', line=dict(color='brown')))
                fig_roc.add_hline(y=0, line_dash='dash', line_color='gray')
                fig_roc.update_layout(title='ROC', height=250, margin=dict(t=30, b=20))
                st.plotly_chart(fig_roc, use_container_width=True)

            # ---- Raw Data ----
            with st.expander("Raw Data (last 10 rows)"):
                st.dataframe(df.tail(10))

            # ---- News ----
            st.subheader("📰 Latest News")
            for item in news:
                news_summary = summarize(item["title"], item.get("summary", ""), llm)
                st.markdown(f"*[{item['title']}]({item['link']})*")
                st.write(news_summary)
                st.divider()

            # ---- Sentiment ----
            st.subheader("📊 Market Sentiment")
            st.markdown(f"### {sentiment_label(sentiment)}")

            # ---- AI Analysis ----
            st.subheader("🤖 AI Analysis")
            st.write(response)