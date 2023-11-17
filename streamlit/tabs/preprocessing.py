import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

title = "Préprocessing"
sidebar_name = "Préprocessing"

def run():

    # Import des jeux de données
    dir_path = os.path.dirname(os.path.realpath(__file__))

    csv_path = os.path.join(dir_path, "../../data/world-happiness-report.csv")
    df1 = pd.read_csv(csv_path)
    df1.index = df1.index + 1

    csv_path = os.path.join(dir_path, "../../data/world-happiness-report-2021.csv")
    df2_full = pd.read_csv(csv_path)
    df2_full.index = df2_full.index + 1

    csv_path = os.path.join(dir_path, "../../data/pays&continents.csv")
    df_continents = pd.read_csv(csv_path, sep=';')
    df_continents.index = df_continents.index + 1

    csv_path = os.path.join(dir_path, "../../data/df_global.csv")
    df = pd.read_csv(csv_path)
    df.index = df.index + 1

    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title(title)

    st.markdown("---")

    st.write("### Pré-processing") 
    
    st.markdown("Dans cette partie, nous allons voir quels sont les traitements que nous avons effectués sur le JDD afin de rendre ***propre*** \
            et de ***préparer*** notre jeu de données pour la partie Feature Engineering.\
                Comme expliqué dans la description des données, nous avons extrait de Kaggle deux bases de données :")
    
    st.markdown(
                """
                - World_hapiness_report.csv
                - World_hapiness_report_2021.csv""")

    st.write("##### Concaténation") 
    
    st.write("Dès l'import des données, la table « World_hapiness_report_2021» est adaptée au format de la table « world_hapiness_report» et on y ajoute la colonne \
            'year' non présente ainsi ses variables sont renommés pour permettre la fusion des datasets.  \
            La concaténation entre les 2 tables est alors réalisable et est effectuée.")
    
    # Harmonisation par suppression des colonnes n'existant que dans le rapport 2021.
    df2=df2_full.drop(['Regional indicator','Standard error of ladder score','upperwhisker',
                    'lowerwhisker','Ladder score in Dystopia','Explained by: Log GDP per capita',
                    'Explained by: Social support','Explained by: Healthy life expectancy',
                    'Explained by: Freedom to make life choices','Explained by: Generosity',
                    'Explained by: Perceptions of corruption','Dystopia + residual'],axis=1)

    # Ajout de la colonne "year" sur le df2021 pour pouvoir le concatener avec le df1
    df2.insert(loc=1,column='year',value=2021)
    
    # Renommage des colonnes du df2021
    df2=df2.rename(columns={'Ladder score':'Life Ladder','Logged GDP per capita':'Log GDP per capita',
                            'Healthy life expectancy':'Healthy life expectancy at birth'})

    # Réindexation du df2021 par rapport au df1
    df2.index=df2.index+1950

    
    # Concaténation des deux jeux de données en un trosième jeu global.
    df_global=pd.concat([df1,df2],axis=0)
    
    agree = st.checkbox('Afficher le DataFrame concaténé contenant tous les données de 2005 à 2021, df_global')
    if agree:
        st.write(df_global.head(5))
        st.write(df_global.shape)
    
    st.write("##### Ajout de la variable 'Regional indicator'")
    
    st.markdown("Dans la table « World_hapiness_report_2021», il existe une colonne avec les continents « Regional indicator » pour chaque pays. Cette variable \
            peut être pertinente si on l’ajoute à notre DataFrame concaténé (df_global). Pour créer cette colonne dans notre df_global, nous fusionons la colonne \
            contenant « Regional indicator » avec notre df_global. Suite à cette étape, il ressort des valeurs manquantes dans « Regional indicator ».")
            
            
            
    # Epurement du df2021 pour ne conserver que les noms de pays et de continent.
    df_continents=df2_full[['Country name','Regional indicator']] # Ajout de la colonne "Regional Indicator" pour indexer rattacher les pays à leur continent respectif indiqué dans le df2021.

    
    # Fusion fusion du df_global avec celui des continents.
    df_global2=df_global.merge(right=df_continents,on='Country name',how='outer')
    
    st.markdown("Pour y remédier, nous introduisons alors une nouvelle source de données\
            ***'pays&continents.csv'*** qui permet suite à une nouvelle fussion de compléter les continents manquants de la variable « Regional indicator ».")
            
        
    # Déplacement de la colonne continent en 2e position.
    df_global2.insert(1,'Regional indicator',df_global2.pop('Regional indicator'))

    
    # Import du fichier contenant l'ensemble des pays et de leur continent de rattachement.
    agree = st.checkbox('Afficher le DataFrame contenant les continents')
    if agree:
        st.write(df_continents.head(5))
        st.write(df_continents.shape)
    
    # Fusion du dataset des continent avec notre dataset principal.
    df_global2=df_global.merge(right=df_continents,on='Country name',how='outer')
    
    # Déplacement de la colonne régional indicator en 2e position.
    df_global2.insert(1,'Regional indicator',df_global2.pop('Regional indicator'))

    
    agree = st.checkbox('Afficher le DataFrame completé des continents manquants')
    if agree:
        df_global2.index = df_global2.index + 1
        st.write(df_global2.head(5)) 
        st.write(df_global2.shape)
    
    st.write("##### Doublons et fautes d'orthographes")
    
    # Analyse des doublons : visiblement il n'y a pas de doublon.
    print(f'Il y a {df_global2.duplicated().sum()} doublons dans le jeu de données.')
    
    # Analyse de cohérence visuelle des données (orthographe, casse, etc.).
    #df_global2['Country name'].unique()
    
    st.write("L'analyse des doublons montre que le jeu de données n'en contient pas. Et l'orthographe des variables catégorielles est sans faute.")
        
    st.write("##### Gestion des valeurs manquantes")
    
    st.write("Pour la gestion des valeurs manquantes, nous avons remarqué qu’il manquait généralement des modalités sur certaines variables selon \
            les années mais que les pays avaient toujours quelques années remplies pour chaque variable. Ainsi, nous avons décidé de remplacer les \
            valeurs manquantes par la médiane des valeurs groupées par pays.")
            
    st.write("###### Pourcentage de valeurs manquantes")
    
    
    size = df_global2.shape
    nan_values = df_global2.isna().sum()

    nan_values = nan_values.sort_values(ascending=True)*100/size[0]

    st.bar_chart(nan_values) #graphe
    
    st.write("Nous avons décidé de remplacer les valeurs manquantes par la médiane des valeurs groupées par pays pour poursuivre l'exploration.")
    

    # Calcul de la médiane en fonction des variables de pays groupés.
    test=df_global2.groupby('Country name')['Perceptions of corruption'].agg('median')
    
    
    # Exploration des differentes variables afin d'analyser les valeurs manquantes.
    df_global2.sort_values('Perceptions of corruption').tail(10)
    
    # Regroupement par pays pour poursuivre l'exploration des données.
    agree = st.checkbox('Afficher le DataFrame regroupé par pays')
    if agree:
        st.write(df_global2[df_global2['Country name']=='Somalia'].head(5)) 
        st.write(df_global2[df_global2['Country name']=='Somalia'].shape)
    
    
    # Séparation des variables catégorielles et des variables numériques et obtention des titres des colonnes.
    cat_data_col = df_global2.select_dtypes(include=[object]).columns.tolist()
    num_data_col = df_global2.select_dtypes(include=[np.number]).columns.tolist()
    num_data_col.remove("year")
    
    st.write("On impute ensuite la médiane à la place des valeurs manquantes par pays et par colonne et on vérifie l'action réalisée:")
    # Imputation de la médiane à la place des valeurs manquantes par pays et par colonne.
    for c in num_data_col:
        df_global2[c]=df_global2.groupby('Country name')[c].transform(lambda x : x.fillna(x.median()))
    
    
    st.write("###### Pourcentage de valeurs manquantes après 1ère imputation")
    
    # Vérification des valeurs manquantes.
    
    size = df_global2.shape
    nan_values = df_global2.isna().sum()
    nan_values = nan_values.sort_values(ascending=True)*100/size[0]

    st.bar_chart(nan_values) #graphe après imputation 1
    
    st.write("Le nombre de Nans a bien diminué mais n'atteint pas zéo : On comprend ici que certaines valeurs manquantes n'ont pas pu etre remplies par \
            manque de données et une impossibilité de calculer la mediane avec une absence de \
            donnée. Par manque de données, on émet l'hypothèse que ces pays sont à des scores du quartile Q1 par rapport au continent auxquel ils appartiennt.")

    # On comprend ici que certaines valeurs manquantes n'ont pas pu etre remplies par manque de données et impossibilité de calculer la mediane avec rien   
    
    # Exploration des lignes qui contiennent encore des valeurs manquantes.
    agree = st.checkbox('Afficher les lignes de df_global contenant encore des valeurs manquantes')
    if agree:
        st.write(df_global2[df_global2.isna().any(axis=1)].head(20)) 
        st.write(df_global2[df_global2.isna().any(axis=1)].shape)
        
    
    # Par manque de données, on émet l'hypothèse que ces pays sont à des scores de q1 par rapport au continent auxquel ils appartiennt

    # Imputation de la valeur du premier quartile de la zone continentale de rattachement à la place des valeurs manquantes par pays et par colonne.
    for c in num_data_col:
        df_global2[c]=df_global2.groupby('Regional indicator')[c].transform(lambda x : x.fillna(x.quantile(q=0.25)))
    
    
    # Vérification des valeurs manquantes.
    print(f'Affichage du nombre de doublons par colonne dans le jeu de données soit {df_global2.isna().sum().sum()} valeurs manquantes.')
    df_global2.isna().sum()
    
    st.markdown("Le nombre de valeurs manquantes après la 2ème imputation est maintenant ***nul***.")
    
    # Vérification des valeurs manquantes. # le graphe est vide confirmant l'absence de valeurs manquantes
    
    #size = df_global2.shape
    #nan_values = df_global2.isna().sum()
    #nan_values = nan_values.sort_values(ascending=True)*100/size[0]

    #st.bar_chart(nan_values) #graphe après imputation 2
    
    # On constate qu'il n'y a plus de valeur manquante.  
    
    st.markdown("On a maintenant un datadet complet, entier, sans valeurs manquantes, ni fautes, ni doublons")
    # Affichage du dataset complet et sans valeur manquante.
    st.write(df_global2.head(5))
    st.write(df_global2.shape)
    
    # Le dataset est entier, sans valeurs manquantes, sans fautes ni doublons.
    
    st.write("### Feature Engineering") 
    
    st.markdown("Le ***feature engineering*** est crucial afin de lancer des modèles de machine learning car il va permettre aux algorithmes de comprendre les \
            données explicatives et de les exploiter dans des calculs nécessaires aux prédictions. Pour se faire, nous devons commencer par procéder à \
            une réduction de dimensions. Suite au pré-processing, le jeu de données est composé de ***10 variables*** après le retrait de 'Country name' et  \
            'year', variables qui n'apportent pas d'informations pertinentes au modèle:")
    
    df_forML = df.drop(['Country name','year'],axis=1)
    
    agree = st.checkbox("Afficher le DataFrame pré-processé sans 'Country name' et 'year'")
    if agree:
        st.dataframe(df_forML.head(5))
        st.write(df_forML.shape)

    
    df_forML.to_csv('df_ml.csv', index=False)
    params = df_forML.drop('Life Ladder', axis=1)
    target = df_forML['Life Ladder']
    
    st.markdown('Puis nous avons procédé aux étapes suivantes:')
    
    st.write("###### Séparation du jeu de données en jeu d'entrainement et jeu test")

    # Instanciation des jeux d'entraînement et de test. 
    X_train, X_test, y_train, y_test = train_test_split(params,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42)
    
    
    st.code('''
            feats = df_global.drop('Life Ladder', axis=1)
    target = df_global['Life Ladder']
            
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(feats,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42)''', language='python')
    
    st.write("###### Utilisation du OneHotEncoder")
    
    st.markdown( "Les modèles de machine learning n’étant pas en mesure d’***interpréter les variables qualitatives***, nous devons encoder le nom des régions \
                afin de rendre ces valeurs exploitables dans X_train et X_test. Nous utilisons donc la méthode nommée ***One Hot Encoding*** qui créera n-1 colonnes remplies de valeurs \
                    booléennes pour savoir si cette ligne appartient à la région indiquée par le nom de la colonne.")
    
    # utilisation du OneHotEncoder
    ohe = OneHotEncoder(drop="first", sparse=False)

    # Adaptation de l'encodeur aux données d'entraînement
    encoded_train = ohe.fit_transform(X_train[['Regional indicator']])

    # Transformation des données de test
    encoded_test = ohe.transform(X_test[['Regional indicator']])
    # Remplacement des colonnes originales par les nouvelles colonnes transformées
    cat_train = pd.DataFrame(encoded_train, columns=ohe.get_feature_names_out(['Regional indicator']), index=X_train.index)
    cat_test = pd.DataFrame(encoded_test, columns=ohe.get_feature_names_out(['Regional indicator']), index=X_test.index)

    # Suppression de la colonne "Regional indicator dans les jeux d'entraînement et de test."
    X_train = X_train.drop('Regional indicator', axis=1)
    X_test = X_test.drop('Regional indicator', axis=1)

    
    # Concaténation des jeux avec le jeu encodé
    X_train_concat = pd.concat([X_train,cat_train], axis = 1)
    X_test_concat = pd.concat([X_test,cat_test], axis = 1)
    
    st.markdown("On obtient des jeux d'entrainement et de test finaux pour lequels la colonne 'Regional indicator' a bien été encodé.\
                ELes jeux contiennents dorénavants ***17 variables explicatives***. ***Le dataset est prêt pour l'entraînement de modèles.***")

    agree = st.checkbox('Afficher X_train encodé')
    if agree:
        st.dataframe(X_train_concat.head())
        
    
    # On obtient un jeu final pour lequel la colonne 'Regional indicator' a bien été encodé
    # Le dataset est prêt pour l'entraînement de modèles.
    