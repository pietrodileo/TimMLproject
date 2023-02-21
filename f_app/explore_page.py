import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import pickle
import seaborn as sns

def show_explore_page():
    st.title('Data Exploration')
    st.subheader('Training Set')
    # read the saved model
    #mdlPath = 'f_app/mdl.pickle'
    dfPath = 'f_app/traincleaned.csv'
    df = load_data(dfPath)
    st.write(df)

    st.subheader('Explore data!')
    selectPlot = st.selectbox('Select the chart you want to see', ['Correlation Plot', 'Boxplot', 'Barplot'])
    doPlot(selectPlot,df)

def doPlot(selectPlot,df):
        fullVars = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL', 'TRAVTIME',
        'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE',
        'PARENT1', 'MSTATUS', 'RED_CAR', 'REVOKED', 'GENDER',
        'COMMERCIAL_CAR_USE', 'URBAN_CAR', 'BACHELORS', 'ELEMENTARY_EDUCATION',
        'MASTERS', 'PHD', 'HIGH_SCHOOL', 'CLERICAL', 'DOCTOR', 'HOME_MAKER',
        'LAWYER', 'MANAGER', 'PROFESSIONAL', 'STUDENT', 'BLUE_COLLAR',
        'MINIVAN', 'PANEL_TRUCK', 'PICKUP', 'SPORTS_CAR', 'VAN', 'SUV']

        if selectPlot == 'Correlation Plot':
            fig1 = plt.figure(figsize=(15,10))
            sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
        elif selectPlot == 'Boxplot':
            var1 = 'TARGET_FLAG'
            var2 = st.selectbox('Select a variable', fullVars)
            fig1 = plt.figure(figsize=(10,8))
            sns.boxplot(data=df, x=var1, y=var2)
        elif selectPlot == 'Barplot':
            fig1 = plt.figure(figsize=(10,8))
            df['TARGET_FLAG'].value_counts().plot(kind='bar', title='Unbalanced classes')

        plotButton = st.button('Plot!')
        if plotButton:
            st.pyplot(fig1)

@st.cache_data
def load_data(dfPath):
    #mdl = pickle.load(open(mdlPath, "rb"))
    df = pd.read_csv(dfPath, index_col="INDEX")
    return df
