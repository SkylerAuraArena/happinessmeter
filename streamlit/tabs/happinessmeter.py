import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from joblib import load
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

title = "Happiness Meter"
sidebar_name = "Happiness Meter"
prediction = "Happiness Score :"

def normalize(df):
    scaler=MinMaxScaler()
    cols=['Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices','Generosity',
        'Perceptions of corruption','Positive affect','Negative affect']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scaler_path = os.path.join(dir_path, "../../models/scaler.pkl")
    loaded_scaler = load(scaler_path)
    with open(scaler_path, 'rb') as file:
        loaded_scaler = pickle.load(file)
    df[cols]=loaded_scaler.transform(df[cols])
    return df

def predict(continent, gdp, socsup, life_exp, freedom, generosity, corruption, positive, negative):
    # La colonne pour l'Europe Centrale et de l'Est est supprimée car elle est la référence.
    X_test = pd.DataFrame({
        'Log GDP per capita': [gdp],
        'Social support': [socsup],
        'Healthy life expectancy at birth': [life_exp],
        'Freedom to make life choices': [freedom],
        'Generosity': [generosity],
        'Perceptions of corruption': [corruption],
        'Positive affect': [positive],
        'Negative affect': [negative],
        'Regional indicator_Commonwealth of Independent States': [0],
        'Regional indicator_East Asia': [0],
        'Regional indicator_Latin America and Caribbean': [0],
        'Regional indicator_Middle East and North Africa': [0],
        'Regional indicator_North America and ANZ': [0],
        'Regional indicator_South Asia': [0],
        'Regional indicator_Southeast Asia': [0],
        'Regional indicator_Sub-Saharan Africa': [0],
        'Regional indicator_Western Europe': [0]
    })
    if continent != 'Central and eastern Europe':
        X_test['Regional indicator_' + continent] = 1
    X_test = normalize(X_test)
    # Chemin absolu vers le dossier contenant app.py
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "../../models/happinessmeter.joblib")
    loaded_model = load(model_path)
    return loaded_model.predict(X_test)

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
    trigger = st.button("Prédire", type="primary")
    if trigger:
        prediction = str(predict(continent, gdp, socsup, life_exp, freedom, generosity, corruption, positive, negative))
        st.write('The models predicts a Life Ladder level of ', prediction)