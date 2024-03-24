import streamlit as st
from functions import StockPricePredictor
from StockDashboard import StockDashboard  # Import the StockDashboard class

train_instance = None  # Define train_instance globally
dashboard_instance = None  # Define dashboard_instance globally

class Train:
    def __init__(self):
        pass


    def train_model(self, tinker, start_date, end_date):
        
        global dashboard_instance  # Access the global dashboard_instance
        predictor = StockPricePredictor()
        df = predictor.get_historical_data(tinker, start_date, end_date)
        train_seq, train_label, test_seq, test_label, scaler, gstock_data, training_size = predictor.preprocess_data(df)
        model = predictor.create_lstm_model(train_seq)
        model.fit(train_seq, train_label, epochs=10, validation_data=(test_seq, test_label), verbose=1)
        
        # Create an instance of StockDashboard and pass necessary parameters
        dashboard_instance = StockDashboard()
        
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

    def run(tinker, start_date, end_date):
        global train_instance  # Access the global train_instance variable
        train_instance = Train()  # Create an instance of Train
        
        trained_model = None  # Define trained_model globally
        trained_model, scaler, gstock_data, test_seq = train_instance.train_model(tinker, start_date, end_date)  # Assign the trained model
        st.success("Model trained successfully!")

        if st.button('Get results'):
            if trained_model is None:
                st.error("Please train the model first.")
            else:
                train_instance.forecasting(test_seq, gstock_data, scaler, trained_model)
def runn():
    dashboard_instance = StockDashboard()  # Create an instance of StockDashboard
    dashboard_instance.run()  # Run the StockDashboard to get the selected tinker, start_date, and end_date
    tinker = dashboard_instance.tinker  # Get the selected tinker
    start_date = dashboard_instance.start_date  # Get the selected start_date
    end_date = dashboard_instance.end_date  # Get the selected end_date

    Train.run(tinker, start_date, end_date)  # Run the model with the obtained tinker, start_date, and end_date