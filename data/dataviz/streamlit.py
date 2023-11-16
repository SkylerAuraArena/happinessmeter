from collections import OrderedDict
import streamlit as st

from tabs import intro, second_tab, third_tab, Visualisation


st.set_page_config(
    page_title= 'Projet' ,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (second_tab.sidebar_name, second_tab),
        (third_tab.sidebar_name, third_tab),
        (Visualisation.sidebar_name, Visualisation)
    ]
)


def run():
    st.sidebar.image(
        "world-happiness-report.png",
        width=250,
    )
    
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    
    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()

