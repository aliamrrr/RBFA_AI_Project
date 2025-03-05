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

# ==============================
# ⚙️ Constants
# ==============================
PLAYER_ID = 0  
BALL_ID = 2    
CONFIDENCE_THRESHOLD = 0.3  
NMS_THRESHOLD = 0.5  
fig = None  

def get_zone_names():
    if st.session_state.teams_switched:
        return {'attack': 'back', 'middle': 'middle', 'back': 'attack'}
    return {'attack': 'attack', 'middle': 'middle', 'back': 'back'}


def process_video(video_path, player_detection_model):
    st.write("⏳ Processing the video...")

    crops = extract_player_crops(video_path, player_detection_model, max_frames=5)
    
    if not crops:
        st.error("⚠️ No players detected. Please try another video.")
        return None
    
    try:
        team_classifier, cluster_images = fit_team_classifier(crops, device = "cuda" if torch.cuda.is_available() else "cpu")

        # Display cluster representative images with their IDs
        st.write("🏷️ **Representative images per cluster:**")
        cols = st.columns(len(cluster_images))

        for i, (cluster_id, img_path) in enumerate(cluster_images.items()):
            image_bgr = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            with cols[i]:
                st.image(image_rgb, caption=f"Cluster {cluster_id}", use_column_width=True)

        return team_classifier

    except Exception as e:
        st.error(f"❌ Classification error: {str(e)}")
        return None


def get_zone(ball_pos, zone_width, first_half, attack_direction):
    ball_x = ball_pos[0]

    if attack_direction == 'right':
        if first_half:
            if ball_x > 2 * zone_width:
                return 'attack'
            elif ball_x < zone_width:
                return 'back'
            else:
                return 'middle'
        else:
            if ball_x > 2 * zone_width:
                return 'back'
            elif ball_x < zone_width:
                return 'attack'
            else:
                return 'middle'
    
    elif attack_direction == 'left':
        if first_half:
            if ball_x > 2 * zone_width:
                return 'back'
            elif ball_x < zone_width:
                return 'attack'
            else:
                return 'middle'
        else:
            if ball_x > 2 * zone_width:
                return 'attack'
            elif ball_x < zone_width:
                return 'back'
            else:
                return 'middle'


def extract_frames(video_path, team_id,attack_direction, team_classifier, config,output_dir_1="visuals/half_1",output_dir_2="visuals/half_2",get_pressing=True):

    first_half = True
    zones = get_zone_names()
    output_dirs_1 = {k: os.path.join(output_dir_1, v) for k, v in zones.items()}
    output_dirs_2 = {k: os.path.join(output_dir_2, v) for k, v in zones.items()}
    
    for d in output_dirs_1.values():
        os.makedirs(d, exist_ok=True)
    
    for d in output_dirs_2.values():
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * 30)
    fast_interval = frame_interval * 5

    player_detection_model = load_player_detection_model()
    keypoints_model = load_field_detection_model()

    progress_bar = st.progress(0)
    status_text = st.empty()

    data = []
    kickoffs = []

    frame_count = 0
    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            break

        st.image(frame)

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        time_seconds = current_frame / fps
        time_minutes = round(time_seconds / 60, 2)

        if time_minutes >= 47:
            first_half = False

        results = player_detection_model.predict(frame, conf=CONFIDENCE_THRESHOLD)
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy()
        )

        detections = detections.with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        players = detections[detections.class_id == PLAYER_ID]
        ball = detections[detections.class_id == BALL_ID]
        
        if players.xyxy.any():
            players.class_id = team_classifier.predict([sv.crop_image(frame, xyxy) for xyxy in players.xyxy])
        else :
            skip = 2 * fast_interval
            for _ in range(skip - 1):
              if not cap.grab() or st.session_state.stop:
                break
        
        # Projection sur le terrain
        keypoints = keypoints_model.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
        kps = sv.KeyPoints.from_inference(keypoints)
        
        if sum(kps.confidence[0] > 0.5) < 4:
            continue

        transformer = ViewTransformer(
            source=kps.xy[0][kps.confidence[0] > 0.5],
            target=np.array(config.vertices)[kps.confidence[0] > 0.5]
        )

        # Calcul de possession
        team_in_possession = False
        if ball.xyxy.any() and players.xyxy.any():
            ball_pos = transformer.transform_points(ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))[0]
            players_pos = transformer.transform_points(players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
            
            if time_minutes <= 1:
               kickoff = is_kickoff(ball_pos, players_pos, players.class_id, pitch_size=(12000, 7000), center_radius=500, threshold=80)
               if kickoff:
                  st.session_state.kickoffs.append(time_minutes)
            elif time_minutes>=44:
               kickoff = is_kickoff(ball_pos, players_pos, players.class_id, pitch_size=(12000, 7000), center_radius=500, threshold=80)
               if kickoff:
                  st.session_state.kickoffs.append(time_minutes)

            distances = np.linalg.norm(players_pos - ball_pos, axis=1)
            closest_team = players.class_id[np.argmin(distances)]
            team_in_possession = (closest_team == team_id)

            data_row = [ball_pos[0], ball_pos[1], closest_team,frame_count]
            data.append(data_row)

            with open('ball_tracking_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["posX", "posY", "team_possession","frame"])
                writer.writerow(data_row)

            clusters = save_team_line_clusters(players_pos, team_id = st.session_state.team_id , output_dir="clusters/")
 
            if team_in_possession:

                zone_width = config.length // 3
                zone = get_zone(ball_pos, zone_width, first_half,attack_direction)

                pressure = is_pressing(players, players_pos, ball_pos, team_id = st.session_state.team_id, zone=zone, team_possession = team_in_possession)

                st.session_state.possession_events.append({
                    "half": 'first' if first_half else 'second',
                    "pos": zone,
                    "time_in_minutes": time_minutes,
                    "time_in_seconds":time_seconds,
                    "team_id": team_id,
                    "pressure":pressure,
                    "kickoff":kickoff
                })

                print(st.session_state.possession_events)

                if first_half:
                   cv2.imwrite(os.path.join(output_dirs_1[zone], f"frame_{time_minutes}_min.jpg"), frame)

                else :
                   cv2.imwrite(os.path.join(output_dirs_2[zone], f"frame_{time_minutes}_min.jpg"), frame)

                skip = frame_interval
            else:
                skip = frame_interval
        
        else : 
            data_row = [None, None, None,frame_count]
            data.append(data_row)
            with open('ball_tracking_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["posX", "posY", "team_possession","frame"])
                writer.writerow(data_row)

        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processed frames: {frame_count}/{total_frames}")

        for _ in range(skip - 1):
            if not cap.grab() or st.session_state.stop:
                break
          
    cap.release()

    st.session_state.processing = False
    return st.session_state.possession_events