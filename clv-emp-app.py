import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# CLV Employee Turnover Prediction App

This app predicts the Employee Working Status!

""")

st.sidebar.header('User Input Features')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/phucbui/employee_turnover/main/dataset_example.csv)
# """)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        # sex = st.sidebar.selectbox('Sex',('male','female'))
        # bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        # bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        # flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        # body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        # data = {'island': island,
                # 'bill_length_mm': bill_length_mm,
                # 'bill_depth_mm': bill_depth_mm,
                # 'flipper_length_mm': flipper_length_mm,
                # 'body_mass_g': body_mass_g,
                # 'sex': sex}
        Year_of_Birth = st.sidebar.number_input(label='Year of Birth', value=1990)
        Gender = st.sidebar.selectbox('Gender',('Female', 'Male'))
        Marriage = st.sidebar.selectbox('Marriage',('Single','Married'))
        Child = year_of_birth = st.sidebar.number_input(label='Child', value=0)
        Family_Background = st.sidebar.selectbox('Family Background',('Live alone', 'Couple family', 'Big family'))
        HomeTown_Region = st.sidebar.selectbox('HomeTown Region',('South', 'Middle', 'North'))
        IT_Major = st.sidebar.selectbox('IT Major',('IT', 'Non-IT'))
        Specialty = st.sidebar.selectbox('Specialty',('Pure Tech', 'Mostly Tech', 'Neutral', 'Mostly Biz', 'Pure Biz'), index=2)
        Personality = st.sidebar.selectbox('Specialty',('Pure SI', 'Mostly SI', 'Neutral', 'Mostly SM', 'Pure SM'), index=2)
        Department = st.sidebar.selectbox('Department',('Admin', 'DEV', 'Test', 'SD', 'R&D'), index=1)
        Business_Domain = st.sidebar.selectbox('Business Domain',('Management', 'Accounting','HR', 'IT', 'Sales & Marketing', 'EDI', 'Shipping', 'Logistics', 'Terminal', 'Solutions', 'RD', 'Others'), index=6)
        Expertise_Level = st.sidebar.selectbox('Expertise Level',('Fresher', 'Junior', 'Senior', 'TL', 'PM', 'IT Manager', 'Manager','GM'), index=2)
        Salary_Code = st.sidebar.selectbox('Salary Code',('Range A', 'Range B', 'Range C', 'Range D'), index=0)
        Average_working_time = st.sidebar.number_input(label='Average working time per Company (Year)', value=1.5)
        data = {'Year of Birth': Year_of_Birth,
                'Gender': Gender,
                'Marriage': Marriage,
                'Child': Child,
                'Family Background': Family_Background,
                'HomeTown Region': HomeTown_Region,
                'IT Major': IT_Major,
                'Specialty': Specialty,
                'Personality': Personality,
                'Department': Department,
                'Business Domain': Business_Domain,
                'Expertise Level': Expertise_Level,
                'Salary Code': Salary_Code,
                'Average working time': Average_working_time
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
# penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'], axis=1)
# df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
    # dummy = pd.get_dummies(df[col], prefix=col)
    # df = pd.concat([df,dummy], axis=1)
    # del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# 'Fresher':0, 'Junior': 1, 'Senior':2, 'TL': 3, 'PM':4, 'IT Manager':5, 'Manager':6,'GM':7
Expertise_Level_mapper = {'Fresher':0, 'Junior': 1, 'Senior':2, 'TL': 3, 'PM':4, 'IT Manager':5, 'Manager':6,'GM':7}

# 'Range A', 'Range B', 'Range C', 'Range D'
Salary_Code_mapper = {'Range A':0, 'Range B':1, 'Range C':2, 'Range D':3}

Working_status_mapper = {'Stay': 0, 'Left':1}

# Actual level
input_df['Expertise Level'] = input_df['Expertise Level'].map(Expertise_Level_mapper ).astype(int)
# Salary range (mil VND)
input_df['Salary Code'] = input_df['Salary Code'].map(Salary_Code_mapper ).astype(int)
# Working status
# input_df['Working Status'] = input_df['Working Status'].map(Working_status_mapper ).astype(int)

from sklearn.preprocessing import OneHotEncoder
# use when different features need different preprocessing
from sklearn.compose import make_column_transformer

column_trans = make_column_transformer(
    (OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Gender', 'Marriage', 'Family Background',\
       'HomeTown Region', 'IT Major', 'Specialty', 'Personality', 'Department',\
       'Business Domain']),
    remainder='passthrough')
    
column_trans.fit_transform(input_df)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Reads in saved classification model
load_lr = pickle.load(open('lr.pkl', 'rb'))
load_rfc = pickle.load(open('rfc.pkl', 'rb'))
load_knc = pickle.load(open('knc.pkl', 'rb'))
load_dtc = pickle.load(open('dtc.pkl', 'rb'))

# Apply model to make predictions
#LogisticRegression
prediction_lr = load_lr.predict(input_df)
prediction_proba_lr = load_lr.predict_proba(input_df)
#RandomForestClassifier
prediction_rfc = load_rfc.predict(input_df)
prediction_proba_rfc = load_rfc.predict_proba(input_df)
#KNeighborsClassifier
prediction_knc = load_knc.predict(input_df)
prediction_proba_knc = load_knc.predict_proba(input_df)
# DecisionTreeClassifier
prediction_dtc = load_dtc.predict(input_df)
prediction_proba_dtc = load_dtc.predict_proba(input_df)

st.subheader('Logistic Regression')
working_status = np.array(['Stay', 'Left'])
st.write(working_status[prediction_lr])
st.subheader('Logistic Regression Probability')
st.write(prediction_proba_lr)

st.subheader('Random Forest Classifier')
working_status = np.array(['Stay', 'Left'])
st.write(working_status[prediction_rfc])
st.subheader('Random Forest Classifier Probability')
st.write(prediction_proba_rfc)

st.subheader('KNeighbors Classifier')
working_status = np.array(['Stay', 'Left'])
st.write(working_status[prediction_knc])
st.subheader('KNeighbors Classifier Probability')
st.write(prediction_proba_knc)

st.subheader('Decision Tree Classifier')
working_status = np.array(['Stay', 'Left'])
st.write(working_status[prediction_dtc])
st.subheader('Decision Tree Classifier Probability')
st.write(prediction_proba_dtc)
