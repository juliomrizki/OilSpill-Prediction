import streamlit as st
import pandas as pd
import numpy as np
import joblib

with open('svc.pkl', 'rb') as file_1:
  svc = joblib.load(file_1)
  
with open('rf.pkl', 'rb') as file_2:
  rf = joblib.load(file_2)
  
with open('knn.pkl', 'rb') as file_3:
  knn = joblib.load(file_3)
  
with open('adb.pkl', 'rb') as file_4:
  adb = joblib.load(file_4)


st.header('Pipeline Spill Cause Prediction')
all_cost = st.number_input('All Costs (USD):', 0,840526118)
environmental = st.number_input('Environmental Remediation Costs (USD):', 0,635000000)
public_cost = st.number_input('Public/Private Property Damage Costs (USD):', 0,23000000)
net_loss = st.slider('Net Loss (Barrels):',0,30565, step=1)
intentional_release = st.slider('Intentional Release (Barrels):',0, 70191,step=1)
unintentional_release = st.slider('Unintentional Release (Barrels):',0, 30565,step=1)
latitude = st.slider('Accident Latitude:',18.45, 70.26,step=0.1)
longitude = st.slider('Accident Longitude:',-158.10, 104.26,step=0.1)
evacuations = st.number_input('Public Evacuations:', 0,700)

if st.button('Predict'):
    data_inf = pd.DataFrame({'All Costs' : [all_cost], 'Environmental Remediation Costs' : [environmental], 
                        'Public/Private Property Damage Costs' : [public_cost], 'Net Loss (Barrels)' : [net_loss], 
                        'Intentional Release (Barrels)' : [intentional_release], 'Unintentional Release (Barrels)' : [unintentional_release], 
                        'Accident Latitude' : [latitude], 'Accident Longitude' : [longitude], 'Public Evacuations' : [evacuations]})
    
    result_svc = svc.predict(data_inf)[0]
    result_rf = rf.predict(data_inf)[0]
    result_knn = knn.predict(data_inf)[0]
    result_adb = adb.predict(data_inf)[0]
    st.markdown(f'Prediksi Penyebab Accident (SVC) = {result_svc}')
    st.markdown(f'Prediksi Penyebab Accident (Random Forest) = {result_rf}')
    st.markdown(f'Prediksi Penyebab Accident (K-Neighbor) = {result_knn}')
    st.markdown(f'Prediksi Penyebab Accident (AdaBoosts) = {result_adb}')
    st.subheader('Bingung ya? Yoo ndak tau masa tanya saya...')
