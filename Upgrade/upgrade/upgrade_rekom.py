import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scripts.data_preprocessing import load_data, add_moving_averages
from scripts.technical_analysis import add_technical_indicators
from scripts.trading_signal_generator import generate_signals

st.title("Analyst Trading Rekomendasi App")

uploaded_file = st.file_uploader("Upload data harga (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df = add_moving_averages(df)
    df = add_technical_indicators(df)
    df = generate_signals(df)

    st.subheader("Data Lengkap + Sinyal Trading")
    st.write(df.tail())

    # Chart Harga & Moving Average
    st.subheader("Chart Harga & Moving Averages")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], label='Close Price')
    ax.plot(df['Date'], df['MA5'], label='MA5')
    ax.plot(df['Date'], df['MA10'], label='MA10')
    ax.legend()
    st.pyplot(fig)

    # Tampilkan sinyal trading terakhir
    latest_signal = df.iloc[-1]['Signal']
    st.subheader(f"Sinyal Trading Terbaru: **{latest_signal}**")

    # Export sinyal ke file CSV
    st.download_button(
        label="Download Sinyal",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='sinyal_trading.csv',
        mime='text/csv'
    )import tadef generate_signals(df):
    df['Signal'] = 'Hold'

    buy_condition = (
        (df['RSI'] < 30) &
        (df['Close'] < df['BB_low']) &
        (df['Stoch_k'] < 20)
    )
    sell_condition = (
        (df['RSI'] > 70) &
        (df['Close'] > df['BB_high']) &
        (df['Stoch_k'] > 80)
    )

    df.loc[buy_condition, 'Signal'] = 'Buy'
    df.loc[sell_condition, 'Signal'] = 'Sell'

    return dffrom sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_classifier(df):
    features = ['RSI', 'EMA20', 'MACD', 'BB_high', 'BB_low', 'Stoch_k', 'ATR']
    df = df.dropna(subset=features + ['Signal'])

    X = df[features]
    y = df['Signal'].map({'Buy': 1, 'Sell': -1, 'Hold': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, 'models/rf_classifier.pkl')

def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['Stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    return df8import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scripts.data_preprocessing import load_data, add_moving_averages
from scripts.technical_analysis import add_technical_indicators
from scripts.trading_signal_generator import generate_signals

st.title("Analyst Trading Rekomendasi Upgrade")

uploaded_file = st.file_uploader("Upload data harga (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df = add_moving_averages(df)
    df = add_technical_indicators(df)

    # Generate sinyal sederhana
    df = generate_signals(df)

    # Load model klasifikasi
    clf = joblib.load('models/rf_classifier.pkl')

    features = ['RSI', 'EMA20', 'MACD', 'BB_high', 'BB_low', 'Stoch_k', 'ATR']
    df['Model_Signal'] = clf.predict(df[features].fillna(0))

    # Map hasil klasifikasi ke label
    signal_map = {1: 'Buy', 0: 'Hold', -1: 'Sell'}
    df['Model_Signal_Label'] = df['Model_Signal'].map(signal_map)

    st.subheader("Data Lengkap dengan Sinyal dan Model Prediksi")
    st.write(df.tail())

    # Chart harga dan indikator
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], label='Close Price')
    ax.plot(df['Date'], df['BB_high'], label='Bollinger High', linestyle='--')
    ax.plot(df['Date'], df['BB_low'], label='Bollinger Low', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    # Tampilkan sinyal terbaru dari model
    latest_signal = df.iloc[-1]['Model_Signal_Label']
    st.subheader(f"Sinyal Trading Terbaru (Model): **{latest_signal}**")

    # Download sinyal lengkap
    st.download_button(
        label="Download Sinyal Lengkap",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='sinyal_trading_accurate.csv',
        mime='text/csv'
    )
