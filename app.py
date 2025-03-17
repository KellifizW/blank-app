import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 自訂 Attention 層（適配 TensorFlow 2.19.0）
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # 計算注意力權重
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        output = K.sum(context, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# 構建模型
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=1)(x)  # pool_size=1 不改變序列長度
    x = Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True))(x)
    x = Attention()(x)
    x = Dropout(0.01)(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 數據預處理
def preprocess_data(data, timesteps):
    data['Yesterday_Close'] = data['Close'].shift(1)
    data['Average'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna()

    features = ['Yesterday_Close', 'Open', 'High', 'Low', 'Average']
    target = 'Close'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(data[features])
    scaled_target = scaler_target.fit_transform(data[[target]])

    X, y = [], []
    for i in range(len(scaled_features) - timesteps):
        X.append(scaled_features[i:i + timesteps])
        y.append(scaled_target[i + timesteps])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler_target, data.index[train_size + timesteps:], data

# 預測函數
@tf.function(reduce_retracing=True)
def predict_step(model, x):
    return model(x, training=False)

# 回測與交易策略
def backtest(data, predictions, test_dates, initial_capital=100000):
    data = data.copy()
    test_size = len(predictions)
    data['Predicted'] = np.nan
    data.iloc[-test_size:, data.columns.get_loc('Predicted')] = predictions.flatten()

    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    position = 0
    capital = initial_capital
    shares = 0
    capital_values = []
    buy_signals = []
    sell_signals = []

    test_start_idx = data.index.get_loc(test_dates[0])
    capital_values = [initial_capital] * test_start_idx

    for i in range(test_start_idx, len(data)):
        close_price = data['Close'].iloc[i].item()  # 使用 .item() 避免 FutureWarning
        if data['MACD'].iloc[i] > data['Signal'].iloc[i] and data['MACD'].iloc[i - 1] <= data['Signal'].iloc[i - 1]:
            if position == 0:
                shares = capital // close_price
                capital -= shares * close_price
                position = 1
                buy_signals.append((data.index[i], close_price))
        elif data['MACD'].iloc[i] < data['Signal'].iloc[i] and data['MACD'].iloc[i - 1] >= data['Signal'].iloc[i - 1]:
            if position == 1:
                capital += shares * close_price
                position = 0
                shares = 0
                sell_signals.append((data.index[i], close_price))

        total_value = capital + (shares * close_price if position > 0 else 0)
        capital_values.append(total_value)

    capital_values = np.array(capital_values)
    total_return = (capital_values[-1] / capital_values[0] - 1) * 100
    max_return = (max(capital_values) / capital_values[0] - 1) * 100
    min_return = (min(capital_values) / capital_values[0] - 1) * 100

    return capital_values, total_return, max_return, min_return, buy_signals, sell_signals

# 主程式
def main():
    st.title("股票價格預測與回測系統 BETA")
    
    st.markdown("""
    ### 功能與限制
    本程式使用深度學習模型（CNN-BiLSTM-Attention）預測股票價格，並基於MACD策略進行模擬交易回測。
    - **功能**：輸入股票代碼以下載2020-2022年的歷史數據，預測未來價格並查看回測結果。
    - **限制**：預測結果僅供參考，不構成投資建議。模型訓練需要大量計算，可能需要3-5分鐘。
    - **注意**：由於雲端環境限制，計算可能需要更長時間。請避免頻繁刷新頁面，並耐心等待結果。
    """)
    
    stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL，依據Yahoo Finance）", value="TSLA")
    timesteps = 60
    
    if st.button("運行分析"):
        with st.spinner("正在下載數據並訓練模型，請等待1-2分鐘..."):
            data = yf.download(stock_symbol, start="2020-01-01", end="2022-12-31")
            if data.empty:
                st.error("無法獲取此代碼的數據。請檢查股票代碼！")
                return

            X_train, X_test, y_train, y_test, scaler_target, test_dates, full_data = preprocess_data(data, timesteps)
            
            model = build_model(input_shape=(timesteps, X_train.shape[2]))
            history = model.fit(X_train, y_train, epochs=200, batch_size=256, validation_split=0.1, verbose=0)
            
            predictions = predict_step(model, X_test)
            predictions = scaler_target.inverse_transform(predictions)
            y_test = scaler_target.inverse_transform(y_test)
            
            capital_values, total_return, max_return, min_return, buy_signals, sell_signals = backtest(
                full_data, predictions, test_dates)
            
            st.subheader(f"{stock_symbol} 分析結果")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(test_dates, y_test, label='Actual Price')
            ax.plot(test_dates, predictions, label='Predicted Price')
            buy_x, buy_y = zip(*[(d, p) for d, p in buy_signals if d in test_dates])
            sell_x, sell_y = zip(*[(d, p) for d, p in sell_signals if d in test_dates])
            ax.scatter(buy_x, buy_y, color='green', label='Buy Signal', marker='^', s=100)
            ax.scatter(sell_x, sell_y, color='red', label='Sell Signal', marker='v', s=100)
            ax.set_title(f'{stock_symbol} Actual vs Predicted Prices (2022)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("回測結果")
            st.write(f"初始資金: $100,000")
            st.write(f"最終資金: ${capital_values[-1]:.2f}")
            st.write(f"總回報率: {total_return:.2f}%")
            st.write(f"最大回報率: {max_return:.2f}%")
            st.write(f"最小回報率: {min_return:.2f}%")
            st.write(f"買入交易次數: {len(buy_signals)}")
            st.write(f"賣出交易次數: {len(sell_signals)}")
            
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            st.subheader("模型評估指標")
            st.write(f"平均絕對誤差 (MAE): {mae:.4f}")
            st.write(f"均方根誤差 (RMSE): {rmse:.4f}")
            st.write(f"決定係數 (R²): {r2:.4f}")
            st.write(f"平均絕對百分比誤差 (MAPE): {mape:.2f}%")

if __name__ == "__main__":
    main()
