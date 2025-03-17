import streamlit as st
import interface1
import interface2

# Page configuration
st.set_page_config(page_title="Merged Application", layout="wide")

# Main title
st.title("📹 Football Video Analysis Application")
st.markdown("---")

# Creating tabs
tab1, tab2 = st.tabs(["⚽ Action Analysis", "📊 Tactical Visualization"])

# Interface 1: Action Analysis
with tab1:
    st.header("🎥 Action Analysis and Clip Generation")
    interface1.show_interface1()

# Interface 2: Tactical Visualization
with tab2:
    st.header("📍 Advanced Tactical Visualization")
    interface2.show_interface_2()






    
