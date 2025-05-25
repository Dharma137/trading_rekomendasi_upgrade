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
    )
