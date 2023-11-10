import streamlit as st
st.title("Mon premier Streamlit")
st.write("Introduction")
if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")
st.markdown("""    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">""", unsafe_allow_html=True)
st.markdown("""
    <div class="p-5 max-w-sm mx-auto bg-white rounded-xl shadow-lg flex items-center space-x-4">
        <div>
            <div class="text-xl font-medium text-black">ChitChat</div>
            <p class="text-gray-500">You have a new message!</p>
        </div>
    </div>
""", unsafe_allow_html=True)