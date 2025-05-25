def generate_signals(df):
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

    return df
