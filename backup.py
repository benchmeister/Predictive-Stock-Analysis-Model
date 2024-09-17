# remember to: pip install yfinance pandas scikit-learn matplotlib

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split

# Fetch historical data for Apple Inc.
ticker = 'AAPL'
data = yf.download(ticker)

# Display the first few rows of the dataset
print(data.head())

# Calculate moving averages
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# Drop NaN values
data = data.dropna()

# Defines the feature set, which includes the closing price and the moving averages.
X = data[['Close', 'MA_10', 'MA_50']]

# Defines the target variable, which is the closing price shifted by one day (to predict the next day’s price).
y = data['Close'].shift(-1).dropna()
X = X[:-1]

# Splits the data into training and testing sets, with 20% of the data reserved for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a predictive model
from sklearn.linear_model import LinearRegression
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
predictions = model.predict(X_test)
# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# Step 5: Visualising Results
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Price')
plt.plot(y_test.index, predictions, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()