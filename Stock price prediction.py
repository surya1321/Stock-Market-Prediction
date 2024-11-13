# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# Adjust plot size
rcParams['figure.figsize'] = 20, 10

# 2. Read the Dataset
df = pd.read_csv("c:/Users/Meruva Surya Tej/Desktop/Stock Price Prediction/INFY.csv")  # Provide the correct path to your CSV file
df.head()

# 3. Analyze the Closing Prices
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price History')
plt.legend()
plt.show()

# 4. Sort and Filter the Dataset
data = df[['Date', 'Close']].sort_index(ascending=True)
data.set_index('Date', inplace=True)

# 5. Normalize the Dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

train_data = scaled_data[:987]
valid_data = scaled_data[987:]

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# 6. Build and Train the LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=10, batch_size=1, verbose=2)  # Increased epochs for better training

# 7. Make Predictions
inputs_data = data[-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# 8. Save the LSTM Model
lstm_model.save("saved_model.h5")

# 9. Visualize the Predictions
train_data = data[:987]
valid_data = data[987:].copy()
valid_data['Predictions'] = predicted_closing_price

plt.figure(figsize=(16, 8))
plt.plot(train_data.index, train_data['Close'], label='Training Data')
plt.plot(valid_data.index, valid_data['Close'], label='Validation Data')
plt.plot(valid_data.index, valid_data['Predictions'], label='Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
