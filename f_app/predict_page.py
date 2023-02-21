import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from preprocessClient import preprocessClient

# Predict the class of a customer who entered their information

def show_predict_page():
    st.title('Predict your class')
    # load data and model
    mdlPath = 'f_app/mdl.pickle'
    dfPath = 'f_app/traincleaned.csv'
    mdl, df = load_data(mdlPath,dfPath)
    
    # Define sliders and selectboxes
    st.subheader("Please, insert your data and click the button below to show your class")
    kidsdriv = st.slider('Kids Driven',0,10)
    age = st.slider('Input age',18,100)
    homekids = st.slider('Home Kids',0,10)
    yoj = st.slider('YOJ',0,20) 
    income = st.slider('Income',0,100000)
    parent1 = st.selectbox('Parent 1', ['No','Yes'])
    homeVal = st.slider('Home Value',0,1000000)
    married = st.selectbox('Marital Status', ['No','Yes'])
    gender = st.selectbox('Select Gender', ['M','F'])
    edu = st.selectbox('Education', ['Elementary_Education','High School','Bachelors','Masters','PhD'])
    job = st.selectbox('Job', ['Lawyer','Clerical','Professional','Blue_Collar','Manager','Student','Home_Maker','Doctor'])
    travtime = st.slider('Travel Time',0,100) 
    car_use = st.selectbox('Car Use', ['Private','Commercial'])
    bluebook = st.slider('Bluebook',0,50000)
    tif = st.slider('TIF',0,20) 
    car_type = st.selectbox('Car Type', ['Van','SUV','Sports_Car','Panel_Truck','Minivan','Pickup'])
    red_car = st.selectbox('Red Car', ['No','Yes'])
    oldclaim = st.slider('Old Claim',0,50000)
    claimFreq = st.slider('Claim Frequency',0,10) 
    revoked = st.selectbox('Revoked', ['No','Yes'])
    mvr_pts = st.slider('MVR Points',0,20) 
    car_age = st.slider('Car Age',0,50) 
    urbanicity = st.selectbox('Urbanicity', ['Highly Urban/ Urban','z_Highly Rural/ Rural'])
        
    # Define buttons
    selectButt = st.button('Your Selection')
    predButt = st.button('Show your class')

    # create a df for the client
    varNames = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME','PARENT1', 'HOME_VAL', 'MSTATUS', 'SEX', 'EDUCATION', 'JOB', 'TRAVTIME',
                'CAR_USE', 'BLUEBOOK', 'TIF', 'CAR_TYPE', 'RED_CAR', 'OLDCLAIM','CLM_FREQ', 'REVOKED', 'MVR_PTS', 'CAR_AGE', 'URBANICITY']
    newSubject = pd.DataFrame([[kidsdriv,age,homekids,yoj,income,parent1,homeVal,married,gender,edu,job,travtime,car_use,bluebook,tif,car_type,red_car,oldclaim,
               claimFreq,revoked,mvr_pts,car_age,urbanicity]],columns=varNames)

    if selectButt:
        # plot new df
        st.write(newSubject)
    elif predButt:
        # To make a prediction, we need to preprocess and convert the df file to the correct format
        catVars = ['PARENT1', 'MSTATUS', 'RED_CAR', 'REVOKED','SEX','CAR_USE','URBANICITY','JOB','CAR_TYPE','EDUCATION']              
        fullVars = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL', 'TRAVTIME',
        'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE',
        'PARENT1', 'MSTATUS', 'RED_CAR', 'REVOKED', 'GENDER',
        'COMMERCIAL_CAR_USE', 'URBAN_CAR', 'BACHELORS', 'ELEMENTARY_EDUCATION',
        'MASTERS', 'PHD', 'HIGH_SCHOOL', 'CLERICAL', 'DOCTOR', 'HOME_MAKER',
        'LAWYER', 'MANAGER', 'PROFESSIONAL', 'STUDENT', 'BLUE_COLLAR',
        'MINIVAN', 'PANEL_TRUCK', 'PICKUP', 'SPORTS_CAR', 'VAN', 'SUV']
        newSubject_s = preprocessClient(catVars,'TARGET_FLAG',fullVars,newSubject,df)
        # predict subject class
        y_pred=mdl.predict(newSubject_s)
        # show prediction
        st.subheader('Your estimated class is: ' + str(y_pred[0]))

@st.cache_data
def load_data(mdlPath,dfPath):
    mdl = pickle.load(open(mdlPath, "rb"))
    df = pd.read_csv(dfPath, index_col="INDEX")
    return mdl, df
