import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from datetime import datetime, timedelta

# 自訂 Attention 層
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_h = self.add_weight(name='W_h', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b_h = self.add_weight(name='b_h', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.W_a = self.add_weight(name='W_a', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        h_transformed = K.tanh(K.dot(inputs, self.W_h) + self.b_h)
        e = K.dot(h_transformed, self.W_a)
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
    x = Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(inputs)
    x = Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True))(x)
    x = Dropout(0.01)(x)
    x = Attention()(x)
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

    data_index = pd.to_datetime(data.index)
    st.write(f"數據範圍：{data_index[0]} 至 {data_index[-1]}，總共 {len(data_index)} 個交易日")

    # 按論文建議分割：前 70% 訓練，後 30% 測試
    total_samples = len(scaled_features) - timesteps
    train_size = int(total_samples * 0.7)
    test_size = total_samples - train_size

    st.write(f"總樣本數：{total_samples}，訓練樣本數：{train_size}，測試樣本數：{test_size}")

    X, y = [], []
    for i in range(total_samples):
        X.append(scaled_features[i:i + timesteps])
        y.append(scaled_target[i + timesteps])

    X = np.array(X)
    y = np.array(y)

    # 分割訓練和測試數據
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    test_dates = data_index[timesteps + train_size:timesteps + train_size + test_size]

    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"訓練數據範圍：{data_index[0]} 至 {data_index[train_size + timesteps - 1]}")
    st.write(f"測試數據範圍：{data_index[train_size + timesteps]} 至 {data_index[-1]}")

    # 顯示訓練數據（前 5 條記錄）
    if len(X_train) > 0:
        st.subheader("訓練數據記錄（前 5 條）")
        train_records = []
        for i in range(min(5, len(X_train))):
            record = {
                "日期": data_index[i + timesteps].strftime('%Y-%m-%d'),
                "特徵 (X_train)": X_train[i].flatten()[:10],  # 展平並顯示前 10 個值
                "目標 (y_train)": y_train[i][0]
            }
            train_records.append(record)
        st.write(pd.DataFrame(train_records))

    return X_train, X_test, y_train, y_test, scaler_target, test_dates, data

# 預測函數
@tf.function(reduce_retracing=True)
def predict_step(model, x):
    st.write(f"X_test shape before prediction: {x.shape}")
    return model(x, training=False)

# 回測與交易策略
def backtest(data, predictions, test_dates, period_start, period_end, initial_capital=100000):
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

    test_dates = pd.to_datetime(test_dates)
    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)

    mask = (test_dates >= period_start) & (test_dates <= period_end)
    filtered_dates = test_dates[mask]

    if len(filtered_dates) == 0:
        st.error("回測時段不在測試數據範圍內！")
        return None, None, None, None, None, None

    test_start_idx = data.index.get_loc(filtered_dates[0])
    test_end_idx = data.index.get_loc(filtered_dates[-1])

    capital_values = [initial_capital] * test_start_idx

    for i in range(test_start_idx, test_end_idx + 1):
        close_price = data['Close'].iloc[i].item()
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
    - **功能**：輸入股票代碼並選擇回測時段，預測未來價格並查看回測結果。
    - **限制**：預測結果僅供參考，不構成投資建議。模型訓練需要大量計算，可能需要3-5分鐘。
    - **注意**：程式根據選擇的回測時段動態下載前3年的歷史數據（約1095天），按 70%/30% 分割為訓練集和測試集。回測時段必須在測試集範圍內。
    """)

    stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL，依據Yahoo Finance）", value="TSLA")
    timesteps = st.slider("選擇時間步長（歷史數據窗口天數）", min_value=10, max_value=100, value=30, step=10)
    epochs = st.slider("選擇訓練次數（epochs）", min_value=50, max_value=200, value=200, step=50)
    st.markdown("**提示**：更高的訓練次數（epochs）可能提高模型準確度，但會增加訓練時間。")

    # 生成回測時段選項（從 2020 年至今，每 6 個月一個時段）
    current_date = datetime(2025, 3, 17)  # 當前日期
    start_date = datetime(2020, 1, 1)  # 開始日期
    periods = []
    temp_end_date = current_date

    while temp_end_date >= start_date:
        period_start = temp_end_date - timedelta(days=179)  # 6 個月約 180 天
        if period_start < start_date:
            period_start = start_date
        periods.append(f"{period_start.strftime('%Y-%m-%d')} to {temp_end_date.strftime('%Y-%m-%d')}")
        temp_end_date = period_start - timedelta(days=1)

    if not periods:
        st.error("無法生成回測時段選項！請檢查日期範圍設置。")
        return

    selected_period = st.selectbox("選擇回測時段（6個月）", periods[::-1])

    if st.button("運行分析"):
        start_time = time.time()

        with st.spinner("正在下載數據並訓練模型，請等待..."):
            # 解析回測時段
            period_start_str, period_end_str = selected_period.split(" to ")
            period_start = datetime.strptime(period_start_str, "%Y-%m-%d")
            period_end = datetime.strptime(period_end_str, "%Y-%m-%d")

            # 根據回測時段動態下載數據（前 3 年 + 回測時段）
            data_start = period_start - timedelta(days=1095)
            data_end = period_end + timedelta(days=1)  # 多加 1 天以確保包含 period_end

            # 檢查股票上市日期
            ticker = yf.Ticker(stock_symbol)
            try:
                first_trade_date = pd.to_datetime(ticker.info.get('firstTradeDateEpochUtc', 0), unit='s')
                if first_trade_date > pd.to_datetime(data_start):
                    st.error(f"股票 {stock_symbol} 上市日期為 {first_trade_date.strftime('%Y-%m-%d')}，無法提供 {data_start.strftime('%Y-%m-%d')} 之前的數據！請選擇更晚的回測時段。")
                    return
            except Exception as e:
                st.warning(f"無法獲取股票 {stock_symbol} 的上市日期，繼續下載數據...（錯誤：{e}）")

            # 下載數據
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("步驟 1/5: 下載數據...")
            st.write(f"正在下載 {stock_symbol} 的歷史數據：{data_start.strftime('%Y-%m-%d')} 至 {data_end.strftime('%Y-%m-%d')}...")
            data = yf.download(stock_symbol, start=data_start, end=data_end)

            # 檢查實際下載的數據範圍
            if data.empty:
                st.error("無法獲取此代碼的數據。請檢查股票代碼或時段！")
                return
            else:
                actual_start = data.index[0].strftime('%Y-%m-%d')
                actual_end = data.index[-1].strftime('%Y-%m-%d')
                st.write(f"實際下載的數據範圍：{actual_start} 至 {actual_end}，共 {len(data)} 個交易日。")

            progress_bar.progress(20)

            # 預處理數據
            status_text.text("步驟 2/5: 預處理數據...")
            X_train, X_test, y_train, y_test, scaler_target, test_dates, full_data = preprocess_data(data, timesteps)
            if X_train is None or X_test.size == 0:
                st.error("訓練或測試數據為空，無法繼續執行！")
                return

            # 檢查回測時段是否在測試集範圍內（允許小幅誤差）
            test_dates = pd.to_datetime(test_dates)
            st.write(f"回測時段：{period_start} 至 {period_end}")
            st.write(f"測試集日期範圍：{test_dates[0]} 至 {test_dates[-1]}")
            if test_dates[0] > period_start + timedelta(days=1):  # 允許 1 天誤差
                st.error(f"回測時段開始日期（{period_start}）早於測試集開始日期（{test_dates[0]}），相差 {(test_dates[0] - period_start).days} 天！")
                return
            if test_dates[-1] < period_end - timedelta(days=3):  # 允許 3 天誤差
                st.error(f"回測時段結束日期（{period_end}）晚於測試集結束日期（{test_dates[-1]}），相差 {(period_end - test_dates[-1]).days} 天！")
                return

            progress_bar.progress(40)

            # 構建並訓練模型
            status_text.text("步驟 3/5: 訓練模型（這可能需要幾分鐘）...")
            model = build_model(input_shape=(timesteps, X_train.shape[2]))
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_split=0.1, verbose=1)
            progress_bar.progress(60)

            # 繪製訓練損失曲線
            st.subheader("訓練損失曲線")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)

            # 進行預測
            status_text.text("步驟 4/5: 進行價格預測...")
            predictions = predict_step(model, X_test)
            predictions = scaler_target.inverse_transform(predictions)
            y_test = scaler_target.inverse_transform(y_test)
            progress_bar.progress(80)

            # 回測
            status_text.text("步驟 5/5: 執行回測...")
            result = backtest(full_data, predictions, test_dates, period_start, period_end)
            if result[0] is None:
                return
            capital_values, total_return, max_return, min_return, buy_signals, sell_signals = result
            progress_bar.progress(100)
            status_text.text("完成！正在生成結果...")

        # 計算運行時間
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 過濾測試數據以顯示回測時段
        test_dates = pd.to_datetime(test_dates)
        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)

        mask = (test_dates >= period_start) & (test_dates <= period_end)
        filtered_dates = test_dates[mask]
        filtered_y_test = y_test[mask]
        filtered_predictions = predictions[mask]

        # 繪製圖表
        st.subheader(f"{stock_symbol} 分析結果（{selected_period}）")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filtered_dates, filtered_y_test, label='Actual Price')
        ax.plot(filtered_dates, filtered_predictions, label='Predicted Price')
        buy_signals = [(d, p) for d, p in buy_signals if period_start <= d <= period_end]
        sell_signals = [(d, p) for d, p in sell_signals if period_start <= d <= period_end]
        buy_x, buy_y = zip(*buy_signals) if buy_signals else ([], [])
        sell_x, sell_y = zip(*sell_signals) if sell_signals else ([], [])
        ax.scatter(buy_x, buy_y, color='green', label='Buy Signal', marker='^', s=100)
        ax.scatter(sell_x, sell_y, color='red', label='Sell Signal', marker='v', s=100)
        ax.set_title(f'{stock_symbol} Actual vs Predicted Prices ({selected_period})')
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

        mae = mean_absolute_error(filtered_y_test, filtered_predictions)
        rmse = np.sqrt(mean_squared_error(filtered_y_test, filtered_predictions))
        r2 = r2_score(filtered_y_test, filtered_predictions)
        mape = np.mean(np.abs((filtered_y_test - filtered_predictions) / filtered_y_test)) * 100

        st.subheader("模型評估指標")
        st.write(f"平均絕對誤差 (MAE): {mae:.4f}")
        st.write(f"均方根誤差 (RMSE): {rmse:.4f}")
        st.write(f"決定係數 (R²): {r2:.4f}")
        st.write(f"平均絕對百分比誤差 (MAPE): {mape:.2f}%")

        st.subheader("運行時間")
        st.write(f"總耗時: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
