📈 Indian Stock Trading Dashboard
A Streamlit-based dashboard for analyzing Indian stocks using technical indicators, deep learning prediction, financial sentiment analysis, and LLM-based insights.

The app combines market data, machine learning, and news analysis to provide interpretable trading insights in one interface.

🚀 Features
Fetches real-time and historical stock data
Computes advanced technical indicators (RSI, MACD, Bollinger Bands, etc.)
Predicts future prices using an LSTM model
Analyzes financial news sentiment using FinBERT
Generates BUY / HOLD / SELL signals
Uses an LLM (LangChain + HuggingFace) to explain insights
Displays results in an interactive Streamlit dashboard

🧠 System Workflow


<img width="333" height="465" alt="image" src="https://github.com/user-attachments/assets/361257ba-3b77-4962-a678-0d12c2e09a71" />


⚙️ Core Components
1. Data Layer - Uses yfinance to fetch OHLCV stock data.
2. Technical Indicators - RSI, MACD, Bollinger Bands, Stochastic, ADX, CCI, ROC, MA20, MA50.
3. Prediction Engine - 2-layer LSTM model predicting the next price using 30-day sequences.
                     - Falls back to linear regression if the data is insufficient.
4. Trading Signals - Rule-based scoring using RSI + MACD + Moving Average crossovers.
5. News Sentiment - Google News RSS → Article extraction (newspaper3k) → FinBERT sentiment.
6. LLM Insights - Explains predictions, summarizes news, and answers stock-related questions.

📁 Project Structure

Stock_prediction/
│
├── app.py
├── stock_prompt_template.json
├── .env
└── requirements.txt

🛠️ Setup
1. Install dependencies:
pip install -r requirements.txt

2. Add environment variable:
HUGGINGFACEHUB_API_TOKEN=your_token_here

3. Run the app:
streamlit run app.py

📦 Tech Stack
Data: yfinance
Indicators: ta
ML: TensorFlow (LSTM)
NLP: FinBERT
LLM: HuggingFace + LangChain
News: feedparser, newspaper3k
UI: Streamlit, Plotly

⚠️ Notes
1. Requires ≥ 60 data points for predictions
2. Supports Indian tickers (.NS, .BO)
3. First run loads the FinBERT model, which may take time
