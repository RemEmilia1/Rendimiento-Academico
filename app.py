import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Cargar el modelo entrenado
with open('modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Título de la aplicación
st.title("Aplicación de Predicción con GaussianNB")

# Descripción de la aplicación
st.write("Ingrese los valores para las características del modelo y haga clic en 'Predecir' para obtener una clasificación.")

# Entradas del usuario para las características
feature1 = st.number_input("Característica 1", min_value=0.0, max_value=10.0, value=5.0)
feature2 = st.number_input("Característica 2", min_value=0.0, max_value=10.0, value=3.0)
feature3 = st.number_input("Característica 3", min_value=0.0, max_value=10.0, value=1.0)
feature4 = st.number_input("Característica 4", min_value=0.0, max_value=10.0, value=0.5)

# Convertir las entradas a un array para el modelo
input_data = np.array([[feature1, feature2, feature3, feature4]])

# Botón para realizar la predicción
if st.button("Predecir"):
    # Realizar la predicción
    prediccion = modelo.predict(input_data)
    st.write(f"Predicción de la clase: {prediccion[0]}")
