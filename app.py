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

# 自訂 Attention 層（保持不變）
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

# 其他函數（build_model, preprocess_data, evaluate_predictions, predict_step, backtest）保持不變
# 這裡省略這些函數的代碼，直接沿用你的原始程式

# 主程式改為 Streamlit 版本
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
            
            # 回測
            capital_values, total_return, max_return, min_return, buy_signals, sell_signals = backtest(
                full_data, predictions, test_dates)
            
            # 評估
            mae, rmse, r2, mape = evaluate_predictions(y_test, predictions)
            
            # 顯示結果
            st.subheader(f"{stock_symbol} 分析結果")
            
            # 繪製價格圖表
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(test_dates, y_test, label='實際價格')
            ax.plot(test_dates, predictions, label='預測價格')
            buy_x, buy_y = zip(*[(d, p) for d, p in buy_signals if d in test_dates])
            sell_x, sell_y = zip(*[(d, p) for d, p in sell_signals if d in test_dates])
            ax.scatter(buy_x, buy_y, color='green', label='買入信號', marker='^', s=100)
            ax.scatter(sell_x, sell_y, color='red', label='賣出信號', marker='v', s=100)
            ax.set_title(f'{stock_symbol} 實際與預測價格 (2022)')
            ax.set_xlabel('日期')
            ax.set_ylabel('價格')
            ax.legend()
            st.pyplot(fig)
            
            # 顯示回測結果
            st.write(f"初始本金: $100,000")
            st.write(f"最終本金: ${capital_values[-1]:.2f}")
            st.write(f"總收益率: {total_return:.2f}%")
            st.write(f"最大收益率: {max_return:.2f}%")
            st.write(f"最小收益率: {min_return:.2f}%")
            st.write(f"買入次數: {len(buy_signals)}")
            st.write(f"賣出次數: {len(sell_signals)}")
            
            # 顯示評估指標
            st.subheader("模型評估指標")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"R²: {r2:.4f}")
            st.write(f"MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()