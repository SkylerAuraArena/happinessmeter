import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Second tab"
sidebar_name = "Second Tab"


def run():

    st.title(title)

    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))

    st.line_chart(chart_data)



    st.area_chart(chart_data)
