import streamlit as st
import os

title = "À propos"
sidebar_name = "À propos"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../assets/happiness_team.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    st.write("### Équipe projet")

    # Exemple avec une URL d'image hébergée en ligne
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../assets/linkedin_logo.png")

    # HTML pour intégrer le logo et le lien
    html_template = """
        <a href="{}" target="_blank">
            <img src="{}" alt="LinkedIn" style="width:20px; height:20px;"/> {}
        </a>
    """

    # Écrire les liens avec le logo
    st.markdown(html_template.format("https://www.linkedin.com/in/etienne-breton-audit-data-blockchain", img_path, "Étienne BRETON"), unsafe_allow_html=True)
    st.markdown(html_template.format("https://www.linkedin.com/in/arlene-muloway", img_path, "Arlène MULOWAY"), unsafe_allow_html=True)
    st.markdown(html_template.format("https://www.linkedin.com/in/victor-lucas", img_path, "Victor LUCAS"), unsafe_allow_html=True)
    st.markdown(html_template.format("https://www.linkedin.com/in/malak-tayeb-70b4b34b", img_path, "Malak TAYEB"), unsafe_allow_html=True)