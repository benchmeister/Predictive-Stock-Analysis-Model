import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Download stock price data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close'], stock_data.index  # Return both prices and dates

# Preprocess data and create input sequences
def preprocess_data(data, dates, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length].values
        target = data[i+sequence_length]
        date = dates[i+sequence_length]
        sequences.append((sequence, target, date))
    return sequences

# Split data into training and testing sets
def split_data(data, test_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    return train_data, test_data

# Train linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Main Execution - Download stock data
ticker = 'GOOGL'
start_date = '2020-01-01'
end_date = '2024-09-17'
stock_data, dates = download_stock_data(ticker, start_date, end_date)

# Preprocess data - Creates sequences of 10 days of stock prices, target on the 11th day
sequence_length = 10
data_sequences = preprocess_data(stock_data, dates, sequence_length)

# Split data into training and testing sets
train_data, test_data = split_data(data_sequences)

# Prepare training data
X_train = np.array([item[0] for item in train_data])
y_train = np.array([item[1] for item in train_data])

# Prepare testing data
X_test = np.array([item[0] for item in test_data])
y_test = np.array([item[1] for item in test_data])
test_dates = [item[2] for item in test_data]  # Extract dates for test data

# Train linear regression model
model = train_model(X_train, y_train)

# Example of using the trained model for prediction
last_sequence = X_test[-1].reshape(1, -1)
predicted_price = model.predict(last_sequence)[0]

# Output the last day’s stock price and predicted price with dates
last_date_actual = test_dates[-1]
predicted_date = test_dates[-1]  # Date of the last prediction, same as the actual date

print(f"Last Day's Stock Price (Actual) on {last_date_actual.date()}: ${stock_data[-1]:.2f}")
print(f"Predicted Stock Price for {predicted_date.date()}: ${predicted_price:.2f}")
