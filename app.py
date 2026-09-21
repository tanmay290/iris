import streamlit as st
import joblib
import numpy as np

model = joblib.load('iris_model.pkl')

st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements:")

sepal_length = st.number_input("Sepal Length (cm)",min_value=0.0,max_value=10.0,value=5.1)

sepal_width = st.number_input("Sepal Width (cm)",min_value=0.0,max_value=10.0,value=3.5)

petal_length = st.number_input("Petal Length (cm)",min_value=0.0,max_value=10.0,value=1.4)

petal_width = st.number_input("Petal Width (cm)",min_value=0.0,max_value=10.0,value=0.2)

if st.button("Predict"):
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Class: {prediction[0]}")
