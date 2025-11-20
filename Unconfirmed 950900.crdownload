import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

st.header('💸 Fake Currency Prediction Using Machine Learning')

data='''### 🔍 Project Overview

Fake currency is a big problem that can affect the economy. Checking each note by hand takes time and can lead to mistakes. This project uses machine learning to help solve that problem by checking if a currency note is real or fake.

The data used in this project comes from images of currency notes. From these images, we get features like variance, skewness, curtosis, and entropy. These features help the model learn patterns and decide if a note is genuine or counterfeit.

This tool makes fake note detection faster, easier, and more accurate.'''
st.markdown(data)


st.image('https://img.freepik.com/premium-photo/money-background_670382-194062.jpg')

with open('fake_currency_pred.pkl','rb') as f:
    chatgpt = pickle.load(f)

# Load 
df = pd.read_csv('FakeCurrencyModel.csv')


st.sidebar.header('Select Features to Predict Fake Currency')
st.sidebar.image('https://c.tenor.com/24htQAU-aqIAAAAd/money-raining-money.gif')

all_values = []

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

final_value = [all_values]

ans = chatgpt.predict(final_value)[0]

import time
random.seed(132)
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Fake Currency')

place = st.empty()
place.image('https://cdnl.iconscout.com/lottie/premium/thumb/magnifying-glass-7174352-5843172.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Fake Currency Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'Fake Currency Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)
st.markdown('Designed by: **Anjali**')
