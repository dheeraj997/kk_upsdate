import streamlit as st
import multipage_streamlit as mt
import homepage
from model import Train
# Create a MultiPage app
app = mt.MultiPage()

# Add pages to the app
app.add("Home", homepage)
app.add("Model", Train.main)

# Run the app
app.run_selectbox()
