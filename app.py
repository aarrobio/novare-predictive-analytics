import app1
import app2
import streamlit as st

st.set_page_config(page_title='Know Your Tumor State',layout="wide")

PAGES = {
    "Auto": app2,
    "Manual": app1
}

st.sidebar.title('How do you wish to measure?')
selection = st.sidebar.radio("Choose one", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# Source: https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4
