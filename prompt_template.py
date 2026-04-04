from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template="""You are a stock market assistant.
User question: {query}

Stock Details:
- Name: {name}
- Sector: {sector}
- Current Price: ₹{price}
- Predicted Price: ₹{prediction}
- 52 High: ₹{high}
- 52 Low: ₹{low}
- Average Volume: {volume}
- PE Ratio: {pe}

Technical Indicators:
- RSI: {rsi}
- MACD: {macd}
- Signal: {signal}

Market Sentiment Score: {sentiment}

Instructions:
- Answer the user question directly
- Use the data above to justify your answer
- Keep it short and clear (max 120 words)
- Mention if the stock looks bullish, bearish, or neutral
- Do not give financial advice disclaimers

Response:""",
    
    input_variables=[
        "query",
        "name",
        "sector",
        "price",
        "prediction",
        "rsi",
        "macd",
        "signal",
        "sentiment",
        "high",
        "low",
        "volume",
        "pe"
    ]
)
template.save("stock_prompt_template.json")