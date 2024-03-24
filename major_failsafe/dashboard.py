import streamlit as st,pandas as pd ,numpy as np
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import plotly.express as px
import requests

st.title("Stock DashBoard")
tinker = st.sidebar.selectbox('Select Stock Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA','F'])
start_date=st.sidebar.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
end_date=st.sidebar.date_input('End Date', value=pd.to_datetime('2024-01-01'))


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


# making date into date-time object
df['date']= pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ')
# Slicing data to four main features open,close,high,low with date-time as index of dataframe

gstock_data = df [['date','open','close','high','low']]
gstock_data .set_index('date',drop=True,inplace=True)


fig=px.line(gstock_data,x=gstock_data.index,y=gstock_data['close'], title =tinker)

st.plotly_chart(fig)

pricing_data,fundamental_data,news=st.tabs(["Pricing Data","Fundamental Data","Top 10 News"])

with pricing_data:
    st.header("Pricing Movements")
    data2=df
    data2['% Change']=df['adjClose'] / df['adjClose'].shift(1)-1
    data2.dropna(inplace=True)
    st.write(df)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is', annual_return,"%")
    stdev=np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation  is', stdev*100,"%")
    st.write('Risk ADj. return is',annual_return/(stdev*100))


    
with fundamental_data:

    
    key = 'WLVTA1QIATT02HEH'
    fd = FundamentalData(key, output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(tinker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(tinker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(tinker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)


with news:
    st.header(f'News of {tinker}')
    sn = StockNews(tinker, save_news=False)
    df_news= sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment=df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

