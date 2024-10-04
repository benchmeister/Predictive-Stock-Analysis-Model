import pandas as pd
import yfinance as yf
from prophet import Prophet
from datetime import datetime

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()  # Reset the index to get 'Date' as a column
    data = data[['Date', 'Adj Close']]  # Use 'Date' and 'Adj Close' for prediction
    data.columns = ['ds', 'y']  # Rename columns to match Prophet's expected format
    data['ds'] = pd.to_datetime(data['ds']).dt.date  # Ensure 'ds' column is in date format
    return data

def fit_prophet_model(stock_data):
    """
    Fit the Prophet model to the historical stock data.
    """
    model = Prophet()
    model.fit(stock_data)
    return model

def make_prediction(model, stock_data, end_date, predict_date):
    """
    Make a prediction for the specified date using the fitted Prophet model.
    """
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    predict_date_dt = datetime.strptime(predict_date, "%Y-%m-%d").date()
    
    # # Debugging: Print the dates in the historical data
    # print("Dates in historical data:")
    # print(stock_data['ds'].to_string(index=False))
    
    # If the prediction date is within the historical data range
    if stock_data['ds'].min() <= predict_date_dt <= stock_data['ds'].max():
        # Check if the prediction date is in the historical data (i.e., a trading day)
        if predict_date_dt not in stock_data['ds'].values:
            print(f"Error: {predict_date} is not a trading day. No historic data available.")
            return None, None
        
        # Make a prediction using the Prophet model
        future = model.make_future_dataframe(periods=0)  # No need to extend the future dataframe
        forecast = model.predict(future)
        
        # Retrieve the actual historical price for comparison
        actual_price = stock_data[stock_data['ds'] == predict_date_dt]['y'].values[0]
        predicted_price = forecast[forecast['ds'] == predict_date]['yhat'].values[0]
        
        print(f"Actual price for {predict_date}: {actual_price}")
        print(f"Predicted price for {predict_date}: {predicted_price}")
        return predicted_price, actual_price
    
    # If the prediction date is beyond the end date, proceed with future prediction
    else:
        days_to_predict = (predict_date_dt - end_date_dt).days
        
        # Create future dataframe to predict up to the specified date
        future = model.make_future_dataframe(periods=days_to_predict + 10)  # Add 10 extra days to be safe
        forecast = model.predict(future)
        print("THIS IS THE FORECAST " + str(forecast))
        
        # Output the prediction for the specified date
        prediction = forecast[forecast['ds'] == predict_date_dt]
        print('THIS IS THE PREDICTION ' + str(prediction))

        if not prediction.empty:
            predicted_price = prediction['yhat'].values[0]
            return predicted_price, None
        else:
            print(f"No prediction available for {predict_date}.")
            return None, None

def main():
    # Input parameters
    ticker = input("Enter the stock symbol: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    predict_date = input("Enter the date to predict (YYYY-MM-DD): ")
    
    stock_data = get_stock_data(ticker, start_date, end_date)
    model = fit_prophet_model(stock_data)
    predicted_price, actual_price = make_prediction(model, stock_data, end_date, predict_date)
    
    if predicted_price is not None:
        if actual_price is not None:
            print(f"The predicted price for {predict_date} is {predicted_price}, and the actual price was {actual_price}.")
        else:
            print(f"The predicted price for {predict_date} is {predicted_price}.")
    else:
        print("Prediction could not be made.")

if __name__ == "__main__":
    main()