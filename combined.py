import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import requests
import pandas as pd
import matplotlib.pyplot as plt

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data
def fetch_coins_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        coins = response.json()
        return {coin['name']: coin['id'] for coin in coins}, None
    except Exception as e:
        return {}, str(e)

@st.cache_data
def fetch_coin_history(coin_id, days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def plot_data(df1, df2, coin1, coin2):
    plt.figure(figsize=(10, 5))
    plt.plot(df1['date'], df1['price'], label=f'{coin1} Price')
    plt.plot(df2['date'], df2['price'], label=f'{coin2} Price')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.title('Cryptocurrency Price Comparison')
    plt.legend()
    st.pyplot(plt)

model = load_model('mnist_digit_classifier.keras')

st.sidebar.title("A00473427")
app_mode = st.sidebar.selectbox("Choose the section", ["question3", "question1", "question2"])

if app_mode == "question3":
    st.title('Question 3')
    uploaded_file = st.file_uploader("Choose an image of a digit...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            white_canvas = Image.new('RGB', image.size, '#aaa')
            white_canvas.paste(image, mask=image.split()[3])
            image = white_canvas
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        image = image.convert('L')
        image = ImageOps.invert(image)
        img_inv = image.resize((28, 28))
        image_array = np.array(img_inv) / 255.0
        image_array = image_array.reshape((1, 28, 28, 1))

        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        st.write(f'Predicted Digit: {predicted_class[0]}')

elif app_mode == "question1":
    st.title('Question 1')
    coins_dict, error = fetch_coins_list()
    if error:
        st.error(f"Failed to fetch cryptocurrencies list: {error}")
        st.stop()

    selected_coin_name = st.selectbox('Select a Cryptocurrency', options=list(coins_dict.keys()))
    selected_coin_id = coins_dict.get(selected_coin_name)

    if selected_coin_id:
        df, error = fetch_coin_history(selected_coin_id)
        if not df.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(df['date'], df['price'], label='Price')
            plt.xlabel('Date')
            plt.ylabel('Price in USD')
            plt.title(f'{selected_coin_name} Price Over the Last Year')
            plt.legend()
            st.pyplot(plt)
            max_price = df['price'].max()
            min_price = df['price'].min()
            max_price_date = df[df['price'] == max_price]['date'].dt.strftime('%Y-%m-%d').values[0]
            min_price_date = df[df['price'] == min_price]['date'].dt.strftime('%Y-%m-%d').values[0]

            st.write(f"Maximum Price: ${max_price:.2f} on {max_price_date}")
            st.write(f"Minimum Price: ${min_price:.2f} on {min_price_date}")

elif app_mode == "question2":
    st.title('Question 2')
    coins_dict, error = fetch_coins_list()
    if error:
        st.error(f"Failed to fetch cryptocurrencies list: {error}")
        st.stop()

    coin_names = list(coins_dict.keys())
    coin1 = st.selectbox('Select the first cryptocurrency', options=coin_names, index=0)
    coin2 = st.selectbox('Select the second cryptocurrency', options=coin_names, index=1)
    time_frames = {"1 week": 7, "1 month": 30, "1 year": 365, "5 years": 365 * 5}  #API failing for 5 years
    time_frame = st.selectbox('Time frame', options=list(time_frames.keys()))

    submit = st.button('Submit')

    if submit:
        df1, error1 = fetch_coin_history(coins_dict[coin1], time_frames[time_frame])
        df2, error2 = fetch_coin_history(coins_dict[coin2], time_frames[time_frame])

        if error1 or error2:
            st.error(f"Error fetching data: {error1 or error2}")
        else:
            plot_data(df1, df2, coin1, coin2)
