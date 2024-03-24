#Importing nessacary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import streamlit as st
#--------------------------------------------------------------------------------------------------------------------------------------------
# Function to fetch historical data for a given ticker
def get_historical_data(tinker, start_date, end_date):
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token 0b4623cf02c29229cfa1f8790ccba6d0bd04983c'
    }

    url = f"https://api.tiingo.com/tiingo/daily/{tinker}/prices"

    params = {
        'startDate': start_date,
        'endDate': end_date,
        'resampleFreq':'daily'
    }

    try:
        requestResponse = requests.get(url,
                                    headers=headers,
                                    params=params)
        df = pd.DataFrame(requestResponse.json())
        requestResponse.raise_for_status()  # Raise an error for bad responses
        print(requestResponse.json())
    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)

    return df
#----------------------------------------------------------------------------------------------------------------------------------------------
# Function to preprocess data for LSTM model
def preprocess_data(df):
    # making date into date-time object
    df['date']= pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # Slicing data to four main features open,close,high,low with date-time as index of dataframe
    gstock_data = df [['date','open','close','high','low']]
    gstock_data .set_index('date',drop=True,inplace=True)

    #Making scsasling values between 0 and 1
    scaler=MinMaxScaler()
    gstock_data [gstock_data .columns] = scaler.fit_transform(gstock_data )
    #creating trainig ad testing data
    training_size=round(len(gstock_data)*0.80)

    train_data = gstock_data [:training_size]
    test_data  = gstock_data [training_size:]

    #Creating trianing and test sequences with a past look back of 50 days
    def create_sequence(df:pd.DataFrame):
        sequences=[]
        labels=[]
        start_index=0
        for stop_index in range(50,len(df)):
            sequences.append(df.iloc[start_index:stop_index])
            labels.append(df.iloc[stop_index])
            start_index +=1
        return (np.array(sequences), np.array(labels))
    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)

    return train_seq, train_label,test_seq, test_label, scaler, gstock_data, training_size
#---------------------------------------------------------------------------------------------------------------------------------------------    
# Function to create and train an LSTM model
def create_lstm_model(train_seq):
    model=Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])


    return model
#------------------------------------------------------------------------------------------------------------------------------------------------
#creating dataframe of original and predicted data for accuracy check
def predictions_data_analysis(test_predicted,gstock_data):
    test_inverse_predicted = scaler.inverse_transform(test_predicted)

    gstock_subset = gstock_data.iloc[-test_predicted.shape[0]:].copy()

    # Creating a DataFrame from the predicted values with appropriate columns
    predicted_df = pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted','high predicted','low predictied'])

    # Aligning the index of predicted_df with the index of gstock_subset
    predicted_df.index = gstock_subset.index

    # Concatenating the two DataFrames along the columns axis
    gs_slic_data = pd.concat([gstock_subset, predicted_df], axis=1)

    gs_slic_data[['open','close','high','close']] = scaler.inverse_transform(gs_slic_data[['open','close','high','close']])

    #plotting the comparision btw data
    plt.figure(figsize=(10, 6))

    # Plot actual 'open' and 'open_predicted'
    plt.plot(gs_slic_data.index, gs_slic_data['open'], label='Actual Open')
    plt.plot(gs_slic_data.index, gs_slic_data['open_predicted'], linestyle='--', label='Predicted Open')

    # Plot actual 'close' and 'close_predicted'
    plt.plot(gs_slic_data.index, gs_slic_data['close'], label='Actual Close')
    plt.plot(gs_slic_data.index, gs_slic_data['close_predicted'], linestyle='--', label='Predicted Close')

    plt.xticks(rotation=45)
    plt.xlabel('Date', size=15)
    plt.ylabel('Stock Price', size=15)
    plt.title('Actual vs Predicted Stock Prices', size=15)
    plt.legend()
    st.pyplot()                   
    return gs_slic_data

#------------------------------------------------------------------------------------------------------------------------------------------------
#forecasting logic
def forecasting(temp_input):
    
    lst_output = []
    n_steps = 50  # Number of timesteps
    n_features = 4  # Number of features
    i = 0

    while i < 30:
        if len(temp_input) > n_steps:
            x_input = temp_input[-n_steps:] # Select the last n_steps elements
            yhat = model.predict(x_input, verbose=0)  # Predict next value
            temp_input = np.concatenate((temp_input, yhat.reshape(1, -1, n_features)), axis=0) # Append prediction to temp_input
            lst_output.append(yhat[0])  # Append prediction to lst_output
            i += 1
        else:
            x_input = temp_input  # Use all available elements
            yhat = model.predict(x_input, verbose=0)  # Predict next value
            temp_input = np.concatenate((temp_input, yhat.reshape(1, -1, n_features)), axis=0) # Append prediction to temp_input
            lst_output.append(yhat[0])  # Append prediction to lst_output
            i += 1
    return lst_output



#------------------------------------------------------------------------------------------------------------------------------------------------

# Function to plot historical data, indications, and predictions
def plot_predictions(gs_slice_data, scaler,test_predicted,lst_output):
        
    day_new=np.arange(1,test_predicted.shape[0])
    day_pred=np.arange(test_predicted.shape[0],test_predicted.shape[0]+30)

    plt.figure(figsize=(10, 6))
    plt.plot(day_new,scaler.inverse_transform(test_predicted[:test_predicted.shape[0]-1]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.xlabel('Date', size=15)
    plt.ylabel('Stock Price', size=15)
    plt.legend()

    #kav kav plot this in st

    #forecast plotting
    forecast_dates = pd.date_range(start=gs_slice_data.index[-1] + pd.Timedelta(days=1), periods=len(lst_output))

    # Creating a DataFrame for the forecasted values with the forecast_dates as index
    forecast_df = pd.DataFrame(lst_output, index=forecast_dates, columns=['future_close_predicted','future_open_predicted','future_high_predicted','future_low_predicted'])

    # Concatenating the forecast_df with gs_slic_data
    combined_data = pd.concat([gs_slice_data, forecast_df])

    # Plotting the data
    plt.figure(figsize=(15, 6))
    plt.plot(combined_data.index, combined_data[['future_close_predicted','future_open_predicted','future_high_predicted','future_low_predicted']])
    plt.xlabel('Date', size=15)
    plt.ylabel('Stock Price', size=15)
    plt.legend(['open_predicted',	'close_predicted',	'high predicted',	'low predictied'])
    plt.show()

    #kav kav plot this in st

    st.pyplot()
#------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit app
st.set_page_config(layout="wide")

# Sidebar CSS
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #333;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main content CSS
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        flex: 1;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stock Price Prediction")

# Sidebar for input parameters
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.selectbox('Select Stock Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
    start_date = st.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2024-01-01'))


# Main content Routing thorugh fuctions
if st.button('Get Results'):
    #data collection
    df = get_historical_data(ticker, start_date, end_date)
    #spliting data to train test
    train_seq, train_label,test_seq, test_label, scaler, gstock_data, training_size= preprocess_data(df)
    #model creation
    model = create_lstm_model(train_seq)
    #model training
    model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)
    #model prediction on test data
    test_predicted = model.predict(test_seq)
    #model data analysis
    gs_slice_data=predictions_data_analysis(test_predicted,gstock_data)
    #forecasting
    lst_output=forecasting(test_seq)
    #plotting the forecast predictions
    plot_predictions(gs_slice_data, scaler,test_predicted,lst_output)

#--------------------------------------------------------------------------------------------------------------------------