import streamlit as st
from prophet import Prophet
from plotly import graph_objs as go
import yfinance as yf
from datetime import date
from prophet.plot import plot_plotly

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Bitcoin-USD Prediction App")



@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data("BTC-USD")
data_load_state.text("Loading data...done")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name = 'Opening Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'Closing Price'))
    fig.layout.update(title="Bitcoin Open and Closing Prices", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# n_years = st.slider("Years of Prediction:", 1,4)
# period = n_years*365
num_days = st.number_input("Enter the number of days for prediction", min_value=1)
opening_price = st.number_input("Enter the opening price in $", min_value=10000)

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

 
ok = st.button('Predict')
if ok:
    m = Prophet()

    m.fit(df_train)
    future = m.make_future_dataframe(periods=num_days)
    forecast = m.predict(future)

    st.subheader('Forecast data')
    future_data = forecast[['ds', 'yhat']]
    future_data = future_data.rename(columns={'ds': 'Date', 'yhat':'Closing Price'})
    future_data = future_data[-num_days:]
    st.write(future_data)

    st.write("forecast data graph")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    fig2 = m.plot_components(forecast)
    st.write(fig2)

