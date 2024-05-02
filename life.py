import streamlit as st
import numpy as np
import pickle

# Load the model
et = pickle.load(open('life2.pkl', 'rb'))

st.title('Life Expectancy Predictor')

# User input fields
infant_deaths = st.number_input('Infant deaths')
under_five_deaths = st.number_input('Under-five deaths')
HIV_AIDS = st.number_input('HIV/AIDS')
Polio = st.number_input('Polio')
BMI_random = st.number_input('BMI')
Adult_Mortality_random = st.number_input('Adult Mortality')
Income_composition_of_resources_random = st.number_input('Income composition of resources')
Schooling_random = st.number_input('Schooling')
Diphtheria_random = st.number_input('Diphtheria')
thinness_59_years_random = st.number_input('Thinness 5-9 years')
thinness_119_years_random = st.number_input('Thinness 1-19 years')
Status_targuided = st.number_input('Status')

if st.button('Predict Life Expectancy'):
    # Gather user input into an array
    query = np.array([infant_deaths, under_five_deaths, HIV_AIDS, Polio, BMI_random, Adult_Mortality_random,
                      Income_composition_of_resources_random, Schooling_random, Diphtheria_random,
                      thinness_59_years_random, thinness_119_years_random, Status_targuided])
    # Make prediction using the model
    prediction = et.predict([query])[0]
    st.title("The Predicted Life Expectancy: {:.2f}".format(prediction))
