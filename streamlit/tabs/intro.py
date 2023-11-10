import streamlit as st


title = "My Awesome DataScientest project."
sidebar_name = "Introduction"


def run():
    
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

        You can browse streamlit documentation and demos to get some inspiration:
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into streamlit [documentation](https://docs.streamlit.io)
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset] (https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset]
          (https://github.com/streamlit/demo-uber-nyc-pickups)
        """
    )

    st.markdown("""
    <div class="p-5 max-w-sm mx-auto bg-white rounded-xl shadow-lg flex items-center space-x-4">
        <div>
            <div class="text-xl font-medium text-black">ChitChat</div>
            <p class="text-gray-500">You have a new message!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    import time

    'Starting a long computation...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
      # Update the progress bar with each iteration.
      latest_iteration.text(f'Iteration {i+1}')
      bar.progress(i + 1)
      time.sleep(0.1)

    '...and now we\'re done!'