# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# data=pd.read_csv('./SelfPracticeSalary_Data.csv')
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# X=scaler.fit_transform(data[['YearsExperience']])
# Y=scaler.fit_transform(data[['Salary']])
# st.title("Salary Prediction App")
# from tensorflow.keras.models import load_model

# model = load_model('salary_prediction_model.h5')
# years_experience = st.number_input("Enter Years of Experience:",min_value=0.0, max_value=50.0, step=0.1)
# if st.button("Predict Salary"):
#     input_data = np.array([[years_experience]])
#     predicted_salary = scaler.inverse_transform(model.predict(input_data))
#     st.write(f"Predicted Salary: ${predicted_salary[0][0]:.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
scalar = joblib.load('scaler.pkl')
model = load_model('salary_prediction_model.h5')
n =  st.slider("Enter years of Experience",1,15,2)
if st.button("Predict Salary"):
    result = model.predict(np.array([[n]]))
    result = scalar.inverse_transform(result)
    st.write("The salary is :",result[0][0])
