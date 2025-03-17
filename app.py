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

# 數據預處理函數（這裡補回缺失的部分）
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

# 主程式
def main():
    st.title("股票價格預測與回測系統")
    
    # 用戶輸入股票代碼
    stock_symbol = st.text_input("請輸入股票代碼（例如 TSLA, AAPL）", value="TSLA")
    timesteps = 60  # 固定參數
    
    if st.button("運行分析"):
        with st.spinner("正在下載數據並訓練模型..."):
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
            
            # 顯示結果
            st.subheader(f"{stock_symbol} 分析結果")
            
            # 繪製價格圖表
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(test_dates, y_test, label='實際價格')
            ax.plot(test_dates, predictions, label='預測價格')
            ax.set_title(f'{stock_symbol} 實際與預測價格 (2022)')
            ax.set_xlabel('日期')
            ax.set_ylabel('價格')
            ax.legend()
            st.pyplot(fig)
            
            # 顯示評估指標
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
