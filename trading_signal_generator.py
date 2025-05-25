import ta

def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['Stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    return df
