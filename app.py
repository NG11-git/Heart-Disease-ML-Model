import streamlit as st
import numpy as np
from joblib import  load
import sklearn.compose._column_transformer as ct


# Temporary fix for old sklearn model compatibility
class _RemainderColsList(list):
    """Compatibility patch for old sklearn models (removed in newer versions)."""
    pass

ct._RemainderColsList = _RemainderColsList




#load the trained model
svm=load(r"C:\Users\lenovo\HeartDisease ML Model\svm.joblib")
log=load(r"C:\Users\lenovo\HeartDisease ML Model\log.joblib")


st.title("Heart Disease Risk Predictor")
st.write("Enter patient details to assess heart disease risk.")

#input fields

#age	sex	pain	BP	chol	fbs	ecg	maxhr	eiang	eist	slope	vessels	thal

age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
cp = st.sidebar.selectbox("Chest pain", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.sidebar.slider("Blood Pressure (mg/dl)", 90, 200, 120)
chol = st.sidebar.slider("Chlesterol (mg/dl)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
restecg = st.sidebar.selectbox("Resting ECG Results", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
thalach = st.sidebar.slider("Max Heart Rate Achievd", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise-Induced Angina", options=["Yes", "No"])
oldpeak = st.sidebar.slider("Exercise Ischemia Stress Test", -1, 4, 1)
slope = st.sidebar.selectbox("Slope of ST Segment", options=["Upsloping", "Flat", "Downsloping"])
ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

# creating dictionary for some input feature
sex_map = {
    "Male": 1, 
    "Female": 0}

pain_map = {
    "Typical Angina": 1,
    "Atypical Angina": 2, 
    "Non-anginal Pain": 3, 
    "Asymptomatic": 4}

fbs_map = {
    "Yes": 1,
    "No":0}

ecg_map = {
    "Normal": 0,
    "ST-T wave abnormality":1,
    "Left ventricular hypertrophy":2}

eiang_map = {
    "Yes": 1,
    "No": 0}

slope_map = {
     "Upsloping": 1, 
     "Flat": 2, 
     "Downsloping": 3}


thal_map = {
    "Normal": 3, 
    "Fixed Defect": 6,
    "Reversible Defect": 7}

# using get() for default in case the input doesn't match the value

sex_encode = sex_map.get(sex, 0)
pain_encode = pain_map.get(cp, 4)
fbs_encode = fbs_map.get(fbs, 0)
ecg_encode = ecg_map.get(restecg, 0)
eiang_encode = eiang_map.get(exang, 0)
slope_encode = slope_map.get(slope, 2)
thal_encode = thal_map.get(thal, 3)
 
#convert inputs to model format

input_data = np.array([
    age, sex_encode, pain_encode, trestbps, chol, fbs_encode, ecg_encode, thalach, eiang_encode,
    oldpeak, slope_encode, ca, thal_encode
    ]).reshape(1, -1)








# model selection

model_choice = st.selectbox("Choose a Model", ["SVM", "Logistic Regression"])


#prediction
if st.button("Predict"):
    if model_choice == "SVM": 
        prediction = svm.predict(input_data)
    else:
        prediction = log.predict(input_data)
    result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
    st.subheader(f"🩺 Prediction: {result}")
 