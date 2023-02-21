import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def show_explore_page():
    st.title('Data Exploration')
    st.subheader('Training Set')
    # read the saved model
    mdlPath = 'f_app/mdl.pickle'
    dfPath = 'f_app/traincleaned.csv'
    mdl, df = load_data(mdlPath,dfPath)
    st.write(df)

@st.cache_data
def load_data(mdlPath,dfPath):
    mdl = pickle.load(open(mdlPath, "rb"))
    df = pd.read_csv(dfPath, index_col="INDEX")
    return mdl, df


        #df[target].value_counts()