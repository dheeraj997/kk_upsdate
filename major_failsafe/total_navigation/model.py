import streamlit as st
import pandas as pd
from functions import StockPricePredictor

train_instance = None  # Define train_instance globally
st.title("Stock Price Prediction App")

tinker1 = st.sidebar.selectbox('Select Stock Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA','F'])
start_date1 = st.sidebar.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
end_date1 = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-01-01'))

class Train:
    def __init__(self):
        pass

    def train_model(self):
        predictor = StockPricePredictor()
        df = predictor.get_historical_data(tinker1, start_date1, end_date1)
        train_seq, train_label, test_seq, test_label, scaler, gstock_data, training_size = predictor.preprocess_data(df)
        model = predictor.create_lstm_model(train_seq)
        model.fit(train_seq, train_label, epochs=10, validation_data=(test_seq, test_label), verbose=1)
        return model, scaler, gstock_data, test_seq

    def forecasting(self, test_seq, gstock_data, scaler, model):
        if model is None:
            st.error("Please train the model first.")
            return

        test_predicted = model.predict(test_seq)
        # Model data analysis
        gs_slice_data = StockPricePredictor().predictions_data_analysis(test_predicted, gstock_data, scaler)
        # Forecasting
        lst_output = StockPricePredictor().forecasting(test_seq, model)
        # Plotting the forecast predictions
        StockPricePredictor().plot_predictions(gs_slice_data, scaler, test_predicted, lst_output)


def main():
    train_instance = Train()  # Use the global train_instance variable
    trained_model = None  # Define trained_model globally

    
    trained_model, scaler, gstock_data, test_seq = train_instance.train_model()  # Assign the trained model
    st.success("Model trained successfully!")

    if st.button('Get results'):
        predictor = StockPricePredictor()
        if trained_model is None:
            st.error("Please train the model first.")
        else:
            train_instance.forecasting(test_seq, gstock_data, scaler, trained_model)

if __name__ == "__main__":
    main()
