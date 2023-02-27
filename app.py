import streamlit as st
import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import LabelEncoder


st.write("# Rental bike count Prediction")


# Some functions used in the project
# 1. convert date in to season
def get_season(date):
    # convert the input date to a month-day tuple
    month_day = (date.month, date.day)

    # define the start and end dates for each season
    spring_start = (3, 20)
    summer_start = (6, 21)
    autumn_start = (9, 22)
    winter_start = (12, 21)

    # determine the season based on the month-day tuple
    if month_day >= spring_start and month_day < summer_start:
        season = 'Spring'
    elif month_day >= summer_start and month_day < autumn_start:
        season = 'Summer'
    elif month_day >= autumn_start and month_day < winter_start:
        season = 'Autumn'
    else:
        season = 'Winter'

    # return the season
    return season



# Data collection form user 
col1, col2, col3 = st.columns(3)

# url https://api.open-meteo.com/v1/forecast?latitude=37.57&longitude=126.98&hourly=temperature_2m&hourly=relativehumidity_2m&hourly=windspeed_10m&hourly=direct_radiation&hourly=snowfall&hourly=rain&hourly=visibility&start_date=2023-02-17&end_date=2023-02-17

max_date = date.today() + timedelta(days=7)
pridct_date = col1.date_input("Prediction Date",date.today(), max_value = max_date, min_value= date.today())  

# Collecting weather data from API

date_string = pridct_date.strftime('%Y-%m-%d')    # format the date as a string in the "YYYY-MM-DD" format

# print the date string
st.write(date_string)
response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude=37.57&longitude=126.98&hourly=temperature_2m&hourly=relativehumidity_2m&hourly=windspeed_10m&hourly=direct_radiation&hourly=snowfall&hourly=rain&hourly=visibility&start_date={date_string}&end_date={date_string}')
data = response.json()

# Storing the data of every hour as dict

Temperature	    = data['hourly']["temperature_2m"]
Humidity	    = data['hourly']["relativehumidity_2m"]
Wind_speed	    = data['hourly']["windspeed_10m"]
Visibility	    = data['hourly']["visibility"]
Solar_Radiation	= data['hourly']["direct_radiation"]
Rainfall	    = data['hourly']["rain"]
Snowfall        = data['hourly']["snowfall"]


Seasons = get_season(pridct_date)
Holiday	= col2.selectbox("Holiday or not",["Yes", "No"])
Functioning_Day	= col3.selectbox("Functioning_Day or not",["Yes", "No"])


Year	= pridct_date.year
Month	= pridct_date.month
Day	    = pridct_date.day
WeekDay	= pridct_date.weekday()
Weekend =  'yes' if WeekDay in [5,6] else 'No'

data_dict = {'Temperature': Temperature,
             'Humidity': Humidity,
             'Wind_speed': Wind_speed,
             'Visibility': Visibility,
             'Solar_Radiation': Solar_Radiation,
             'Rainfall': Rainfall,
             'Snowfall': Snowfall,
            }

df = pd.DataFrame(index=range(24))

# Assigning the values for each hour
for hour in range(24):
    df.loc[hour, 'Hour'] = hour
    df.loc[hour, 'Temperature'] = Temperature[hour]
    df.loc[hour, 'Humidity'] = Humidity[hour]
    df.loc[hour, 'Wind_speed'] = Wind_speed[hour]
    df.loc[hour, 'Visibility'] = Visibility[hour]
    df.loc[hour, 'Solar_Radiation'] = Solar_Radiation[hour]
    df.loc[hour, 'Rainfall'] = Rainfall[hour]
    df.loc[hour, 'Snowfall'] = Snowfall[hour]
    df.loc[hour, 'Seasons'] = Seasons
    df.loc[hour, 'Holiday'] = Holiday
    df.loc[hour, 'Functioning_Day'] = Functioning_Day
    df.loc[hour, 'Year'] = Year
    df.loc[hour, 'Month'] = Month
    df.loc[hour, 'Day'] = Day
    df.loc[hour, 'WeekDay'] = WeekDay
    df.loc[hour, 'Weekend'] = Weekend

# Displaying the DataFrame
st.write(df)

# lable encoding
le = LabelEncoder()
sel_col = ['Seasons','Holiday', 'Functioning_Day', 'Weekend']
for col in sel_col:
    df[col]= le.fit_transform(df[col])
    
#st.write(df)

# loading trained model

pickle_in = open('GBR_model.pkl', 'rb')
model = pickle.load(pickle_in)

pickle_in = open('scaler.pkl', 'rb')
scaler = pickle.load(pickle_in)

# Doing prediciton
df = scaler.transform(df)
prediction = model.predict(df)
prediction = np.square(prediction)


if st.button('Predict'):
    st.bar_chart(prediction)
