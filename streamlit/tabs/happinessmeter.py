import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

title = "Happiness Meter"
sidebar_name = "Happiness Meter"

def predict():
    loaded_model = load('../models/happinessmeter.joblib')
    return loaded_model.predict(X_test_scaled)

def run():

    st.title(title)

    multi = '''Veuillez sélectionner les valeurs afin d\'obtenir le score de bonheur correspondant.\nLes valeurs par défaut correspondent aux moyennes de chaque variable.\n\n
    '''
    st.text(multi)

    continent = st.selectbox(
        'Select the continental area of the country you want to predict the happiness score:',
    ('Central and eastern Europe', 'Commonwealth of Independent States', 'East Asia', 'Latin America and Caribbean', 'Middle East and North Africa', 'North America and ANZ', 'South Asia', 'Southeast Asia', 'Sub-Saharan Africa', 'Western Europe'))
    gdp = st.slider('Log GDP per capita', 0.00, 13.00, 9.38)
    socsup = st.slider('Social support level', 0.00, 10.00, 8.13)
    life_exp = st.slider('Healthy life expectancy at birth', 0.00, 100.00, 63.60)
    freedom = st.slider('Freedom to make life choices', 0.00, 10.00, 7.47)
    generosity = st.slider('Generosity', -1.00, 10.00, -0.01)
    corruption = st.slider('Perception of corruption', 0.00, 10.00, 7.40)
    positive = st.slider('Positive affects', 0.00, 10.00, 7.10)
    negative = st.slider('Negative affects', 0.00, 10.00, 2.68)

    st.write('The current Continent is ', continent)
    st.write('The current GDP is ', gdp)
    st.write('The current Social Support is ', socsup)
    st.write('The current Life expectancy is ', life_exp)
    st.write('The current Freedom is ', freedom)
    st.write('The current Generosity is ', generosity)
    st.write('The current Corruption is ', corruption)
    st.write('The current Positive affects is ', positive)
    st.write('The current Negative affects is ', negative)
