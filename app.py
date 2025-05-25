import streamlit as st
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
