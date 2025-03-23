import streamlit as st
import interface1
import interface2

# Set page configuration
st.set_page_config(page_title="Football Match Analysis", layout="wide")

# Page title
st.title("⚽ Football Match Analysis Dashboard")

# Tabs for different interfaces
tab1, tab2 = st.tabs(["📊 Event Timelines & Heatmaps", "🎥 Video & Image Analysis"])

with tab1:
    st.header("📊 Event Timelines & Heatmaps")
    st.markdown(
        """
        **Features:**
        - Generate event timelines for key moments (Kickoff, Final-Middle-Defensive third positions, Goal kicks, Penaltys)
        - Create match heatmaps for team/player analysis
        """
    )
    interface1.show_interface1()

with tab2:
    st.header("🎥 Video & Image Analysis")
    st.markdown(
        """
        **Features:**
        - Assist in tactical analysis with sequence-based or image visuals
        """
    )
    interface2.show_interface_2()





    
