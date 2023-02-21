import streamlit as st
import pandas as pd
from predict_page import show_predict_page
from explore_page import show_explore_page

# This script launches a Streamlit web app capable of data exploration and prediction

# to view this Streamlit app on browser, run it with the following command:
# streamlit run .\f_app\app.py --server.port 9800
# In this case I changed the server port because the default one is locked on Windows

# create the web page
page = st.sidebar.selectbox("Explore or Predict", ("Predict","Explore"))
if page == "Predict":
    show_predict_page()
elif page == "Explore":
    show_explore_page()