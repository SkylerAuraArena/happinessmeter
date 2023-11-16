import streamlit as st
import os
import pandas as pd

title = "Exploration"
sidebar_name = "Exploration"

# Import des données des dataframes


dir_path = os.path.dirname(os.path.realpath(__file__))

csv_path = os.path.join(dir_path, "../../data/world-happiness-report.csv")
df1 = pd.read_csv(csv_path)
df1.index = df1.index + 1

csv_path = os.path.join(dir_path, "../../data/world-happiness-report-2021.csv")
df2_full = pd.read_csv(csv_path)
df2_full.index = df2_full.index + 1

def run():

    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title(title)

    st.markdown("---")
    
    st.markdown("Nous avons utilisé les données fournies dans les différentes éditions du ***World Happiness Report*** depuis 2012, bien que certaines données \
                remontent à 2005. Le rapport est basé sur huit indicateurs permettant d'établir l'indicateur global de qualité de vie, le ***Life Ladder***. \
                Ce dernier mesure le bien-être et s'appuie sur des questions posées aux habitants des pays, qui évaluent leur vie sur une échelle de 0 à 10.\
                C'est l'***indicateur principal***, dérivé des suivants dont les descriptions sont décrites ci-dessous:")
                
    st.markdown(
    """
    L'objectif de notre analyse est de comprendre les problématiques suivantes :
    - ***Country name*** : nom du pays
    - ***year*** : année d'évaluation des données
    - ***log GDP per capita*** : PIB enregistré par habitant  qui donne des informations sur la taille de l’économie et ses performances.
    - ***Social support*** : un soutien social ou le fait d'avoir quelqu'un sur qui compter en cas de difficultés.
    - ***Healthy life expectancy at birth*** : espérance de vie en bonne santé  à la naissance
    - ***Freedom to make life choices*** : mesure la liberté qu'a une personne de prendre des décisions concernant sa propre vie
    - ***Generosity*** : mesure la générosité de la population
    - ***Perceptions of corruption*** : comment les habitants perçoivent le pays au niveau de la corruption
    - ***Positive affect*** : positivité de la population
    - ***Negative affect*** : négativité de la population
    """)

    st.write("En premier lieu, voici le premier jeu de données contenant les données de 2005 à 2020 : ")
    st.dataframe(df1.head(5))
    st.write(df1.shape)
    st.dataframe(df1.describe())

    st.write("En second lieu, les données de 2021 ont été fusionnées avec les données précédentes: ")
    st.dataframe(df2_full.head(5))
    st.write(df2_full.shape)
    st.dataframe(df2_full.describe()) 