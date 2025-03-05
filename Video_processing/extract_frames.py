# ==============================
# 📦 Standard Library Imports
# ==============================
import os
import json
import csv
from datetime import timedelta

# ==============================
# 🖥️ Computer Vision & Image Processing
# ==============================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# 🚀 Machine Learning & Deep Learning
# ==============================
import torch

# ==============================
# 🎨 Visualization & UI
# ==============================
import streamlit as st
import supervision as sv  # For object detection and tracking visualizations

# ==============================
# ⚽ Soccer-Specific Utilities
# ==============================
from utils.soccer import SoccerPitchConfiguration  # Configuration for the soccer pitch

# ==============================
# 🏗️ Model Loading & Processing
# ==============================
from utils.models import (
    load_player_detection_model,  # Loads the model for detecting players
    load_field_detection_model    # Loads the model for detecting the field
)

# ==============================
# 🔄 Data Transformation & Processing
# ==============================
from utils.view import ViewTransformer  # Handles perspective transformations
from utils.team_classifier import (
    extract_player_crops,  # Extracts cropped images of players
    fit_team_classifier     # Trains a classifier to distinguish teams
)
from utils.frame_process import (
    is_pressing,              # Determines if a pressing situation is happening
    save_team_line_clusters,  # Saves clustering data for team formations
    is_kickoff                # Detects kickoff events in the match
)

# ==============================
# 📊 Data Visualization & Analysis
# ==============================
from utils.annotators import (
    plot_ball_heatmap,       # Generates heatmaps for ball positions
    plot_barycenter_heatmap  # Generates heatmaps for barycenter tracking
)
from utils.visualizations import (
    draw_pitch,  # Draws the soccer pitch on an image
    plot_heatmap_and_tracking,  # Visualizes heatmaps and player tracking data
    plot_possession_graph,
    create_exponential_peaks,
    calculate_position_percentages
)

from utils.process_tools import (
    get_zone_names,
    process_video,
    get_zone,
    extract_frames,
    
)


# ==============================
# ⚙️ Constants
# ==============================
PLAYER_ID = 0  
BALL_ID = 2    
CONFIDENCE_THRESHOLD = 0.3  
NMS_THRESHOLD = 0.5  
fig = None  

# ==============================
# 🏗️ Session State Init
# ==============================

def init_session_state():
    if 'processing' not in st.session_state:
        st.session_state.processing = False  
    if 'stop' not in st.session_state:
        st.session_state.stop = False  
    if 'preparing' not in st.session_state:
        st.session_state.preparing = False  
    if 'teams_switched' not in st.session_state:
        st.session_state.teams_switched = False  
    if 'possession_events' not in st.session_state:
        st.session_state.possession_events = []
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None  
    if 'kickoffs' not in st.session_state:
        st.session_state.kickoffs = []
    if 'team_id' not in st.session_state:
        st.session_state.team_id = None
    if 'attack_direction' not in st.session_state:
        st.session_state.attack_direction = None
    if "position" not in st.session_state:
        st.session_state.position = "defense"

init_session_state()

# User interface
st.title("⚽ Football Video Analysis System")

# Control Widgets
video_path = st.text_input("Video Path", r"C:\Users\Administrateur\Desktop\RBFA\last_ukr.mp4")

col1, col2 = st.columns(2)
col4, col5 = st.columns(2)

with col1:
    start_btn = st.button("🚀 Start Processing")
with col2:
    stop_btn = st.button("⛔ Stop Processing")



# Events gesture
if start_btn:
    st.session_state.processing = True
    st.session_state.stop = False

if stop_btn:
    st.session_state.stop = True

# Tabs to organize the interface
tab1, tab2, tab3 = st.tabs(["🎥 Video processing","📊 Timeline visualisations", "📄 Tracking data"])

with tab1:
        if st.session_state.processing and not st.session_state.stop:

            # Checks if the classifier is already loaded, to avoid re-running it every time
            if st.session_state.classifier is None:
                config = SoccerPitchConfiguration()
                player_model = load_player_detection_model()

                with st.spinner('Initializing team classifier...'):
                    st.session_state.classifier = process_video(video_path, player_model)
            
            # Display the selectors for team_id and attack_direction
            st.session_state.team_id = st.selectbox("Select Team ID", options=[0, 1], index=0)
            st.session_state.attack_direction = st.selectbox("Select Attack direction", options=["left", "right"], index=0)

            # Proceed button for analysis
            proceed_button = st.button("Proceed with Analysis")

            if proceed_button and st.session_state.team_id is not None and st.session_state.attack_direction is not None:
                try:
                    # Initialize configurations and model
                    config = SoccerPitchConfiguration()
                    player_model = load_player_detection_model()

                    # Check if classifier is loaded
                    if st.session_state.classifier is not None:
                        with st.spinner('Analyzing video frames...'):
                            extract_frames(video_path, st.session_state.team_id, st.session_state.attack_direction, st.session_state.classifier, config)

                    st.success("✅ Processing completed successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.processing = False


with tab2:
        st.subheader("Generate Possession Report")
        
        # Generates json for positions,events timeline
        if st.button("Generate JSON Report"):
            if 'possession_events' in st.session_state and st.session_state.possession_events:
                json_data = json.dumps(st.session_state.possession_events, indent=4)
                
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="possession_report.json",
                    mime="application/json"
                )
            else:
                st.warning("No possession data available. Please process the video first.")


        # Section to upload json file
        uploaded_file = st.file_uploader("Upload JSON Data", type=["json"], help="Drag and drop your JSON file here.")

        if uploaded_file is not None:
          try:
            data = json.load(uploaded_file)

            percentages = calculate_position_percentages(data)
        
            st.markdown("### Répartition des Positions (%)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="🔴 Defense", value=f"{int(percentages['back'])}%")
            
            with col2:
                st.metric(label="🔵 Middle", value=f"{int(percentages['middle'])}%")

            with col3:
                st.metric(label="🟢 Attack", value=f"{int(percentages['attack'])}%")
            
            if st.button("📊 Generate Graph"):
                fig = plot_possession_graph(data,st.session_state.kickoffs)
                st.pyplot(fig)
          except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid file.")


with tab3:
   #Visualize tracking data insights (Defense - Attack - Middle heatmaps)
   st.session_state.position = st.selectbox("Select a position :" ,["Defense","Middle","Attack"])
   if st.button("Get position heatmap"):
    csv_file = f"team_1_{st.session_state.position}.csv"
    flip_x = st.session_state.attack_direction == "left"
    fig = plot_heatmap_and_tracking(csv_file,flip_x=True)
    st.pyplot(fig)


# Parameters
st.subheader("Current Settings")
if st.session_state.team_id is not None:
    st.write(f"Team ID: {st.session_state.team_id}")
st.write(f"Team Orientation: {'Switched' if st.session_state.teams_switched else 'Normal'}")
st.write(f"Processing Status: {'Running' if st.session_state.processing else 'Stopped'}")

