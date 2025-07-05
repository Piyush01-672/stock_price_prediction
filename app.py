import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model (make sure your model is in the correct path)
model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'POWERGRID.NS'  # Default stock if none is entered

        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 10, 1)

        # Download stock data
        df = yf.download(stock, start=start, end=end)
        data_desc = df.describe()

        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predictions
        y_predicted = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot 1
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time (in years)")
        ax1.set_ylabel("Price (in ₹)")
        ax1.legend()
        plt.tight_layout()  
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time (in years)")
        ax2.set_ylabel("Price (in ₹)")
        ax2.legend()
        plt.tight_layout()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time (predicted day)")
        ax3.set_ylabel("Price (in ₹)")
        ax3.legend()
        plt.tight_layout()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

                # AI Summary Logic
        predicted_price = float(y_predicted[-1][0])
        actual_price = float(y_test[-1])
        price_change = predicted_price - actual_price
        direction = "increase 📈" if price_change > 0 else "decrease 📉"

        # Trend analysis for EMA
        ema_trends = []
        for ema in [ema20, ema50, ema100, ema200]:
            try:
                if ema.iloc[-1] > ema.iloc[-5]:
                    ema_trends.append("rising 📈")
                else:
                    ema_trends.append("falling 📉")
            except:
                ema_trends.append("insufficient data")

        # Short-term EMA (20/50)
        if len(ema_trends) >= 2:
            short_term_trend = (
                "bullish momentum" if "falling" not in ema_trends[0] + ema_trends[1] 
                else "possible trend reversal"
            )
        else:
            short_term_trend = "insufficient data for short-term trend"

        # Long-term EMA (100/200)
        if len(ema_trends) >= 4:
            long_term_trend = (
                "strong uptrend" if "falling" not in ema_trends[2] + ema_trends[3] 
                else "possible weakening of long-term trend"
            )
        else:
            long_term_trend = "insufficient data for long-term trend"

        # Prediction analysis
        if len(y_test) > 10 and len(y_predicted) > 10:
            actual_trend = "upward 📈" if y_test[-1] > y_test[-10] else "downward 📉"
            model_trend = "upward 📈" if y_predicted[-1] > y_predicted[-10] else "downward 📉"
        else:
            actual_trend = model_trend = "insufficient data"

        ai_summary = f"""
        📊 The AI predicts a **{direction}** in the stock price.<br>
        💰 Final predicted price: <b>{predicted_price:.2f}</b> | Recent actual price: <b>{actual_price:.2f}</b><br><br>
        📈 Short-term EMA trend (20/50): {ema_trends[0]}, {ema_trends[1]} → <b>{short_term_trend}</b><br>
        📉 Long-term EMA trend (100/200): {ema_trends[2]}, {ema_trends[3]} → <b>{long_term_trend}</b><br><br>
        🧠 Original trend: <b>{actual_trend}</b> | Model prediction trend: <b>{model_trend}</b>
        """
        
        return render_template('index.html',
                               plot_path_ema_20_50=ema_chart_path,
                               plot_path_ema_100_200=ema_chart_path_100_200,
                               plot_path_prediction=prediction_chart_path,
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path,
                               ai_summary=ai_summary)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
