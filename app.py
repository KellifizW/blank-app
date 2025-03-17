import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras.backend import tanh, dot, sigmoid, softmax, sum as K_sum
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 自訂 Attention 層
class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W_h = self.add_weight(name='attention_W_h', shape=(input_shape[-1], input_shape[-1]),
                                   initializer='random_normal', trainable=True)
        self.W_a = self.add_weight(name='attention_W_a', shape=(input_shape[-1], 1),
                                   initializer='random_normal', trainable=True)
        self.b_h = self.add_weight(name='attention_b_h', shape=(input_shape[-1], 1),
                                   initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        intermediate = tanh(dot(inputs, self.W_h) + self.b_h)
        e = dot(intermediate, self.W_a)
        e = tf.squeeze(e, axis=-1)
        alpha = softmax(e, axis=1)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        output = K_sum(context, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# 構建模型
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=128, kernel_size=1, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=1)(x)
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
        close_price = float(data['Close'].iloc[i])
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
    st.title("股票價格預測與回測系統BETA(Backtesting Stage)")
    
    # 添加說明文字
    st.markdown("""
    ### 程式功能與限制
    本程式使用深度學習模型（CNN-BiLSTM-Attention）預測股票價格，並基於 MACD 策略進行模擬交易回測。
    - **功能**：輸入股票代碼後，程式將下載 2020-2022 年的歷史數據，預測未來價格並顯示回測結果。
    - **限制**：預測結果僅供參考，不構成投資建議。模型訓練需大量計算，可能需要 3-5 分鐘，請耐心等候。
    - **提示**：由於雲端環境限制，計算時間較長，請勿頻繁刷新頁面，耐心等待結果顯示。
    """)
    
    # 用戶輸入股票代碼
    stock_symbol = st.text_input("請輸入股票代碼（例如 TSLA, AAPL）", value="TSLA")
    timesteps = 60  # 固定參數
    
    if st.button("運行分析"):
        with st.spinner("正在下載數據並訓練模型，請耐心等候 1-2 分鐘..."):
            # 下載數據
            data = yf.download(stock_symbol, start="2020-01-01", end="2022-12-31")
            if data.empty:
                st.error("無法獲取該股票數據，請檢查股票代碼是否正確！")
                return

            # 數據預處理
            X_train, X_test, y_train, y_test, scaler_target, test_dates, full_data = preprocess_data(data, timesteps)
            
            # 構建並訓練模型
            model = build_model(input_shape=(timesteps, X_train.shape[2]))
            history = model.fit(X_train, y_train, epochs=200, batch_size=256, validation_split=0.1, verbose=0)
            
            # 預測
            predictions = predict_step(model, X_test)
            predictions = scaler_target.inverse_transform(predictions)
            y_test = scaler_target.inverse_transform(y_test)
            
            # 回測
            capital_values, total_return, max_return, min_return, buy_signals, sell_signals = backtest(
                full_data, predictions, test_dates)
            
            # 顯示結果
            st.subheader(f"{stock_symbol} 分析結果")
            
            # 繪製價格圖表（英文標籤）
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(test_dates, y_test, label='Actual Price')
            ax.plot(test_dates, predictions, label='Predicted Price')
            buy_x, buy_y = zip(*[(d, p) for d, p in buy_signals if d in test_dates])
            sell_x, sell_y = zip(*[(d, p) for d, p in sell_signals if d in test_dates])
            ax.scatter(buy_x, buy_y, color='green', label='Buy Signal', marker='^', s=100)
            ax.scatter(sell_x, sell_y, color='red', label='Sell Signal', marker='v', s=100)
            ax.set_title(f'{stock_symbol} Actual vs Predicted Price (2022)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
            
            # 顯示回測結果（中文）
            st.subheader("回測結果")
            st.write(f"初始本金: $100,000")
            st.write(f"最終本金: ${capital_values[-1]:.2f}")
            st.write(f"總收益率: {total_return:.2f}%")
            st.write(f"最大收益率: {max_return:.2f}%")
            st.write(f"最小收益率: {min_return:.2f}%")
            st.write(f"買入次數: {len(buy_signals)}")
            st.write(f"賣出次數: {len(sell_signals)}")
            
            # 顯示評估指標（中文）
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            st.subheader("模型評估指標")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"R²: {r2:.4f}")
            st.write(f"MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()
