import streamlit as st
import extract_frames
import main

st.set_page_config(page_title="Application Fusionnée", layout="wide")

# Création des onglets
tab1, tab2 = st.tabs(["Interface 1", "Interface 2"])

with tab1:
    extract_frames.show_interface1()  # Appelle la fonction définie dans extract_frames.py

with tab2:
    main.show_interface_2()  # Appelle la fonction définie dans main.py





    
