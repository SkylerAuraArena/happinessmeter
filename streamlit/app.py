from collections import OrderedDict
import streamlit as st
import os

from tabs import intro, exploration, dataviz, modeling, happinessmeter, about

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 475 235" class="logo"><path d="M234.51,4.11H5.63A4.1,4.1,0,0,0,1.84,9.84L63.43,156.93a4.17,4.17,0,0,0,3.82,2.55,4.13,4.13,0,0,0,3.8-2.55L120,40l49,116.91a4.11,4.11,0,0,0,3.8,2.55,4.17,4.17,0,0,0,3.82-2.55l61.59-147a4.11,4.11,0,0,0-2-5.44A4.35,4.35,0,0,0,234.51,4.11Z" fill="#fff"></path><path d="M472.36,152.45,416.63,95a47.16,47.16,0,0,0,36.93-45.36c0-25.78-21.7-46.74-48.39-46.74A4.16,4.16,0,0,0,401,7V155.32a4.12,4.12,0,0,0,4.08,4.17h64.24a4.13,4.13,0,0,0,3.8-2.54A4.09,4.09,0,0,0,472.36,152.45Z" fill="#fff"></path><path d="M268.89,96.36h85.63a4.13,4.13,0,0,1,4.13,4.13V155.3a4.13,4.13,0,0,1-4.13,4.13H268.89a4.13,4.13,0,0,1-4.13-4.13V100.49A4.14,4.14,0,0,1,268.89,96.36Z" fill="#fff"></path><path d="M268.89,4.16h85.63a4.13,4.13,0,0,1,4.13,4.13V63.1a4.13,4.13,0,0,1-4.13,4.13H268.89a4.13,4.13,0,0,1-4.13-4.13V8.29A4.14,4.14,0,0,1,268.89,4.16Z" fill="#fff"></path><path d="M472.81,225.12v-4.66a5.8,5.8,0,0,1-2.85.74c-1.3,0-1.88-.66-1.88-2v-8.79h4.81v-5.09h-4.81v-5.09h-5.81v5.09h-2.55v5.09h2.55v9.7c0,4.74,2.55,6.16,6,6.16a8.39,8.39,0,0,0,4.58-1.2m-15.89-13.92v-6.09a6.28,6.28,0,0,0-6.52,4.51v-4.13h-5.81v20.38h5.81v-7.64c0-4.89,2.37-7.21,6.21-7.21Zm-21.5,4.66a5.25,5.25,0,0,1-4.9,5.57h-.42a5.45,5.45,0,0,1-5.38-5.52v-.13h0a5.25,5.25,0,0,1,4.91-5.57H430a5.45,5.45,0,0,1,5.4,5.5A.38.38,0,0,1,435.42,215.82Zm5.73,0h0a11.08,11.08,0,0,0-22.16,0h0a10.62,10.62,0,0,0,10.66,10.58H430A10.75,10.75,0,0,0,441.13,216c0-.1,0-.19,0-.29m-31.24,0a5.1,5.1,0,0,1-10,2,5,5,0,0,1,0-2h0a5.1,5.1,0,1,1,10-2A5,5,0,0,1,409.9,215.74Zm5.81,0h0c0-6.85-4.56-10.6-9.32-10.6a7.64,7.64,0,0,0-6.39,3.34v-2.95h-5.81v26.54h5.88v-8.79a7.86,7.86,0,0,0,6.39,3.06c4.84,0,9.32-3.74,9.32-10.57m-31.48-1.66h-8.76a4.6,4.6,0,0,1,4.43-4.36,4.41,4.41,0,0,1,4.33,4.36M390,216.3h0c0-5.81-3.13-11.13-10-11.13a10.19,10.19,0,0,0-10.2,10.18c0,.17,0,.33,0,.5h0a10.8,10.8,0,0,0,19,6.75l-3.31-2.93a6.79,6.79,0,0,1-5,2.09,4.76,4.76,0,0,1-5.09-4h14.42v-1.45m-28.64-7.88c0,2.24-1.63,3.77-4.51,3.77h-6v-7.64h6c2.85,0,4.61,1.3,4.61,3.82ZM368,226l-6.55-9.55a8.28,8.28,0,0,0,5.73-8.28h0c0-5.5-3.77-8.84-10.19-8.84H344.8V226h5.88v-8.56h4.64l5.73,8.56Zm-39.28-6.22h0c0-3.74-3.34-5.09-6.19-6.09-2.22-.76-4.18-1.3-4.18-2.55h0c0-.79.74-1.4,2.14-1.4a12,12,0,0,1,5.55,1.91l2.22-4a14.33,14.33,0,0,0-7.64-2.37c-4.13,0-7.49,2.34-7.49,6.5h0c0,4,3.23,5.32,6.11,6.16,2.24.69,4.23,1.1,4.23,2.37h0c0,.92-.76,1.53-2.55,1.53a11.51,11.51,0,0,1-6.44-2.55L312,223.05a14.39,14.39,0,0,0,8.79,3.08c4.43,0,7.85-2.06,7.85-6.6m-18.8,0h0c0-3.74-3.34-5.09-6.19-6.09-2.22-.76-4.18-1.3-4.18-2.55h0c0-.79.74-1.4,2.14-1.4a12,12,0,0,1,5.55,1.91l2.22-4A14.33,14.33,0,0,0,301.7,205c-4.13,0-7.49,2.34-7.49,6.5h0c0,4,3.23,5.32,6.11,6.16,2.24.69,4.23,1.1,4.23,2.37h0c0,.92-.76,1.53-2.55,1.53a11.51,11.51,0,0,1-6.44-2.55L293,222.85a14.39,14.39,0,0,0,8.79,3.08c4.43,0,7.85-2.06,7.85-6.6m-25-5.62H275.9a4.6,4.6,0,0,1,4.43-4.36,4.41,4.41,0,0,1,4.33,4.36m5.66,2.19h0c0-5.81-3.13-11.13-10-11.13A10.19,10.19,0,0,0,270.14,215c0,.17,0,.33,0,.5h0a10.8,10.8,0,0,0,19.05,6.75l-3.31-2.93a6.79,6.79,0,0,1-5,2.09,4.76,4.76,0,0,1-5.09-4h14.72v-1.45m-24.15,9.64V212.31c0-4.71-2.55-7.64-6.95-7.64a7.3,7.3,0,0,0-6,3.29v-2.9h-5.81v20.38h5.81V214c0-2.75,1.43-4.18,3.64-4.18s3.52,1.43,3.52,4.18v11.41Zm-24.22-20.38h-5.81v20.38h5.78Zm.15-7.41h-6.11v5.09h6.11Zm-15.46,18a5.09,5.09,0,0,1-10,2.14,5.19,5.19,0,0,1,0-2.15h0a5.09,5.09,0,1,1,10-2.14A5.19,5.19,0,0,1,226.87,215.77Zm5.8,0h0c0-6.85-4.53-10.6-9.32-10.6a7.61,7.61,0,0,0-6.37,3.35v-3h-5.91v26.52h5.81V223.3a7.85,7.85,0,0,0,6.37,3.06c4.86,0,9.32-3.74,9.32-10.57m-30.82,0a5.1,5.1,0,0,1-10,2,5,5,0,0,1,0-2h0a5.1,5.1,0,0,1,10-2A5,5,0,0,1,201.75,215.77Zm5.81,0h0c0-6.85-4.56-10.6-9.32-10.6a7.62,7.62,0,0,0-6.39,3.35v-3h-5.73v26.52h5.81V223.3a7.86,7.86,0,0,0,6.39,3.06c4.84,0,9.32-3.74,9.32-10.57m-32,3c0,2.09-1.83,3.59-4.56,3.59-1.86,0-3.16-.92-3.16-2.55h0c0-1.83,1.53-2.83,4-2.83a9.14,9.14,0,0,1,3.72.76ZM181,226V214.09c0-5.55-2.8-8.84-9.14-8.84a18,18,0,0,0-7.95,1.66l1.45,4.43a15.25,15.25,0,0,1,5.65-1.12c2.9,0,4.41,1.35,4.41,3.74v.36a14.24,14.24,0,0,0-5.09-.84c-4.84,0-8.25,2.06-8.25,6.52h0a6.34,6.34,0,0,0,6.38,6.3,4.73,4.73,0,0,0,.65,0,7.81,7.81,0,0,0,6.19-2.55v2.22Zm-22.67,0V199.22h-5.86v10.6H141.64v-10.6h-5.88V226h5.88V215.24h10.85V226Zm-45.62-10.19a5.09,5.09,0,1,1-10,2.14,5.19,5.19,0,0,1,0-2.15h0a5.09,5.09,0,0,1,10-2.14A5.19,5.19,0,0,1,112.73,215.77ZM118.46,226V198.07h-5.78v10.19a7.91,7.91,0,0,0-6.37-3.06c-4.86,0-9.32,3.74-9.32,10.6h0c0,6.83,4.53,10.57,9.32,10.57a7.5,7.5,0,0,0,6.37-3.31V226ZM93.17,198.07H87.39V226H93.2ZM82.7,211.19V205.1a6.28,6.28,0,0,0-6.52,4.51v-4.13H70.4v20.38h5.81v-7.64c0-4.89,2.37-7.21,6.21-7.21Zm-21.39,4.66a5.25,5.25,0,0,1-4.9,5.57H56a5.45,5.45,0,0,1-5.38-5.52v-.13h0a5.25,5.25,0,0,1,4.91-5.57h.39a5.45,5.45,0,0,1,5.4,5.5v.15Zm5.73,0h0a11.08,11.08,0,0,0-22.16,0h0a10.62,10.62,0,0,0,10.66,10.58h.36A10.75,10.75,0,0,0,67,216.06c0-.1,0-.19,0-.29M45.36,199.21H39.15L33.6,217.32l-6-18.19H22.52l-6,18.19L11,199.21H4.7l9.14,26.95h5.09l6-17.5,6,17.5H36.1Z" fill="#fff"></path></svg>
"""

st.set_page_config(
    page_title= 'World Happiness Report analysis projet' ,
    page_icon="https://cdn-icons-png.flaticon.com/512/166/166538.png",
)

# Chemin absolu vers le dossier contenant app.py
dir_path = os.path.dirname(os.path.realpath(__file__))
css_file_path = os.path.join(dir_path, "style.css")

with open(css_file_path, "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (exploration.sidebar_name, exploration),
        (dataviz.sidebar_name, dataviz),
        (modeling.sidebar_name, modeling),
        (happinessmeter.sidebar_name, happinessmeter),
        (about.sidebar_name, about),
    ]
)


def run():
    st.sidebar.markdown(svg, unsafe_allow_html=True)
    
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    
    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()