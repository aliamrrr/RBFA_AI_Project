def show_interface1():   

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
        calculate_position_percentages,
        plot_ball_heatmap_and_tracking
    )

    from utils.process_tools import (
        get_zone_names,
        process_video,
        get_zone,
        extract_frames,
        clean_ball_tracking_data,
        get_times_by_pos,
        extract_video_clips
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
    tab1, tab2, tab3,tab4 = st.tabs(["🎥 Video processing","📊 Timeline visualisations", "📄 Tracking data","Generate sequences"])


    import streamlit as st
    import tempfile
    from moviepy.editor import VideoFileClip

    with tab1:
        if st.session_state.processing and not st.session_state.stop:

            # Sélecteur pour choisir la minute spécifique
            minute_selected = 20

            # Vérifie si le classifier est déjà chargé pour éviter de le recharger
            if st.session_state.classifier is None:
                config = SoccerPitchConfiguration()
                player_model = load_player_detection_model()

                # Génération du fichier temporaire contenant une séquence d'une minute
                with st.spinner(f"Extracting video segment for minute {minute_selected}..."):
                    video = VideoFileClip(video_path)
                    start_time = minute_selected * 60  # Convertir en secondes
                    end_time = min(start_time + 60, video.duration)  # Ne pas dépasser la durée totale

                    # Création d'un fichier temporaire
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                        temp_video_path = temp_file.name
                        video.subclip(start_time, end_time).write_videofile(temp_video_path, codec="libx264", audio=False)

                # Traitement de la vidéo avec la séquence extraite
                with st.spinner('Initializing team classifier...'):
                    st.session_state.classifier = process_video(temp_video_path, player_model)

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

            if st.button("Generate JSON Report"):
                print(st.session_state.possession_events)
                if 'possession_events' in st.session_state and st.session_state.possession_events:
                    json_data = json.dumps(st.session_state.possession_events, indent=4, default=str)

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
                    st.metric(label="🔴 Defensive third", value=f"{int(percentages['back'])}%")
                
                with col2:
                    st.metric(label="🔵 Middle third ", value=f"{int(percentages['middle'])}%")

                with col3:
                    st.metric(label="🟢 Final third", value=f"{int(percentages['attack'])}%")
                
                if st.button("📊 Generate Graph"):
                    fig = plot_possession_graph(data,st.session_state.kickoffs)
                    st.pyplot(fig)
             except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid file.")


    # Initialize the add_trajectory flag in session_state
    if 'add_trajectory' not in st.session_state:
        st.session_state.add_trajectory = False

    with tab3:
        tab1, tab2 = st.tabs(["📍 Position Heatmap", "⚽ Ball Tracking Heatmap"])

        with tab1:  # Position Heatmap
            st.title("📍 Position-Based Heatmap")
            st.session_state.add_trajectory = st.checkbox("Add trajectory", value=st.session_state.add_trajectory,key="1")  # Initialize checkbox with session_state value
            st.session_state.position = st.selectbox("Select a position:", ["Defense", "Middle", "Attack"]).lower()

            if st.button("Generate Position Heatmap"):
                csv_file = f"data/team_1_{st.session_state.position}.csv"
                flip_x = st.session_state.attack_direction == "left"
                print(flip_x)

                fig = plot_heatmap_and_tracking(csv_file, flip_x="left",add_trajectory=st.session_state.add_trajectory)  # Assuming you have a function for that
                    
                if fig:
                    st.pyplot(fig)

        with tab2:  # Ball Tracking Heatmap
            st.title("⚽ Ball Tracking Heatmap")
            st.session_state.add_trajectory = st.checkbox("Add trajectory", value=st.session_state.add_trajectory,key="2")  # Initialize checkbox with session_state value

            if st.button("Generate Ball Tracking Heatmap"):
                csv_path = "data/ball_tracking_data.csv"
                clean_ball_tracking_data(csv_path, output_dir="processed_data")
                st.success(f"✅ Cleaning complete. File saved at: processed_data/ball_tracking_cleaned.csv")


                fig = plot_ball_heatmap_and_tracking("processed_data/ball_tracking_cleaned.csv",add_trajectory=st.session_state.add_trajectory)  # Assuming you have a function for that

                if fig:
                    st.pyplot(fig)

        with tab4:
            st.header("Generate Video Clips 🎥")

            uploaded_json = st.file_uploader("📂 Choose a JSON file", type=["json"])

            if uploaded_json:
                data = json.load(uploaded_json)

                times_defense = get_times_by_pos(data, "back")
                times_attack = get_times_by_pos(data, "attack")
                times_middle = get_times_by_pos(data, "middle")

                st.write("🔍 **Choose a position to extract the videos**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🛡 Defense"):
                        print(video_path)
                        with st.spinner("⏳ Extraction in progress..."):
                            clips = extract_video_clips(video_path, times_defense)
                        st.success("✅ Extraction done !")
                        for clip in clips:
                            st.video(clip)
                            st.download_button("📥 Download", clip, file_name=os.path.basename(clip))

                with col2:
                    if st.button("🎯 Attack"):
                        with st.spinner("⏳ Extraction in progress..."):
                            clips = extract_video_clips(video_path, times_attack)
                        st.success("✅ Extraction done !")
                        for clip in clips:
                            st.video(clip)
                            st.download_button("📥 Download", clip, file_name=os.path.basename(clip))

                with col3:
                    if st.button("🔄 Middle"):
                        with st.spinner("⏳ Extraction in progress..."):
                            clips = extract_video_clips(video_path, times_middle)
                        st.success("✅ Extraction terminée !")
                        for clip in clips:
                            st.video(clip)
                            st.download_button("📥 Download", clip, file_name=os.path.basename(clip))


    # Parameters
    st.subheader("Current Settings")
    if st.session_state.team_id is not None:
        st.write(f"Team ID: {st.session_state.team_id}")
    st.write(f"Team Orientation: {'Switched' if st.session_state.teams_switched else 'Normal'}")
    st.write(f"Processing Status: {'Running' if st.session_state.processing else 'Stopped'}")

