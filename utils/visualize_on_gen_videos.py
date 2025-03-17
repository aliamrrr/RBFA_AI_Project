import cv2
import numpy as np
import supervision as sv

# ==============================
# 📦 Standard Library Imports
# ==============================
import os
import json
import csv
from datetime import timedelta
from scipy.interpolate import splprep, splev

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

def get_team_classifier(video_path, player_detection_model):
    print("Processing frames crops")
    crops = extract_player_crops(video_path, player_detection_model, max_frames=10)
    
    if not crops:
        print("no crops")
        return None
    
    try:
        print("Fitting team classifier")
        team_classifier, cluster_images = fit_team_classifier(crops, device = "cuda" if torch.cuda.is_available() else "cpu")

        return team_classifier

    except Exception as e:
        print("error")
        return None

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from view import ViewTransformer  # Assure-toi d'importer la bonne classe

CONFIG = SoccerPitchConfiguration()
keypoints_model = load_field_detection_model()

def generated_lines_video(input_path, output_path, model, team_classifier, keypoints_model, config,
                   confidence_threshold=0.5, nms_threshold=0.4, fps_output=30):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_output, (width, height))

    frame_count = 0
    halo_annotator = sv.HaloAnnotator()
    triangle_annotator = sv.TriangleAnnotator()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if frame_count % 3 != 0:
            frame_count += 1
            continue
        
        if not ret:
            break

        results = model.predict(frame, conf=confidence_threshold)
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy()
        ).with_nms(threshold=nms_threshold, class_agnostic=True)

        players = detections[detections.class_id == 0]
        ball = detections[detections.class_id == 2]

        if len(players.xyxy) > 0:
            players.class_id = team_classifier.predict([sv.crop_image(frame, xyxy) for xyxy in players.xyxy])

        keypoints = keypoints_model.infer(frame, confidence=confidence_threshold)[0]
        kps = sv.KeyPoints.from_inference(keypoints)
        
        if sum(kps.confidence[0] > 0.5) < 4:
            continue

        transformer = ViewTransformer(
            source=kps.xy[0][kps.confidence[0] > 0.5],
            target=np.array(config.vertices)[kps.confidence[0] > 0.5]
        )

        if players.xyxy.any():
            players_pos = transformer.transform_points(players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))

            # ---- TEAM 0 (Lignes segmentées) ----
            team_0_indices = np.where(players.class_id == 0)[0]  

            if len(team_0_indices) > 2:
                team_players_pos = players_pos[team_0_indices]
                team_players_frame = players.xyxy[team_0_indices]

                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(team_players_pos[:, 0].reshape(-1, 1))

                color_white = (255, 255, 255)

                for cluster_id in range(3):
                    role_indices = np.where(clusters == cluster_id)[0]
                    sorted_indices = role_indices[np.argsort(team_players_pos[role_indices, 1])]

                    role_players = team_players_pos[sorted_indices]
                    role_players_frame = team_players_frame[sorted_indices]

                    if len(role_players) > 1:
                        for i in range(len(role_players) - 1):
                            x1 = int((role_players_frame[i][0] + role_players_frame[i][2]) / 2)
                            y1 = int(role_players_frame[i][3])  

                            x2 = int((role_players_frame[i + 1][0] + role_players_frame[i + 1][2]) / 2)
                            y2 = int(role_players_frame[i + 1][3])  

                            pt1 = (x1, y1)
                            pt2 = (x2, y2)

                            # Lignes segmentées
                            num_segments = 10  
                            for j in range(num_segments):
                                alpha1 = j / num_segments
                                alpha2 = (j + 0.5) / num_segments  

                                seg_x1 = int(pt1[0] * (1 - alpha1) + pt2[0] * alpha1)
                                seg_y1 = int(pt1[1] * (1 - alpha1) + pt2[1] * alpha1)

                                seg_x2 = int(pt1[0] * (1 - alpha2) + pt2[0] * alpha2)
                                seg_y2 = int(pt1[1] * (1 - alpha2) + pt2[1] * alpha2)

                                cv2.line(frame, (seg_x1, seg_y1), (seg_x2, seg_y2), color_white, 2)

            # ---- TEAM 1 (Polygones remplis) ----
            team_1_indices = np.where(players.class_id == 0)[0]  

            if len(team_1_indices) > 2:
                team_players_pos = players_pos[team_1_indices]
                team_1_players_frame = players.xyxy[team_1_indices]

                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(team_players_pos[:, 0].reshape(-1, 1))

                polygon_colors = [(0, 0, 255)]  # Rouge

                overlay = frame.copy()  # Copie pour transparence
                alpha = 0.4  # Niveau de transparence

                for cluster_id in range(3):
                    role_indices = np.where(clusters == cluster_id)[0]

                    if len(role_indices) >= 3:  # On ne dessine un polygone que si 3 joueurs ou +
                        print("polygone tracé")
                        cluster_points = np.array([
                                [(box[0] + box[2]) // 2, box[3]]  # x_center, y_bas (pieds des joueurs)
                                for box in team_1_players_frame[role_indices]
                            ])
                        sorted_points = cluster_points[np.argsort(cluster_points[:, 1])]

                        polygon_pts = np.array(sorted_points, np.int32).reshape((-1, 1, 2))
                        
                        # Remplir le polygone sur l'overlay
                        cv2.fillPoly(overlay, [polygon_pts], polygon_colors[0])

                # Mélange l'overlay avec l'image originale pour ajouter la transparence
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


            # ---- ANNOTATIONS ----
            ellipse_annotator = sv.EllipseAnnotator()
            frame = ellipse_annotator.annotate(scene=frame.copy(), detections=players)


            out.write(frame)
            frame_count += 1

            if frame_count>=30:
                    break

    cap.release()
    out.release()
    print(f"✅ Vidéo annotée sauvegardée sous {output_path}")


# Exemple d'utilisation
video_path = "clips/clip_1.mp4"
player_detection_model = load_player_detection_model()
team_classifier = get_team_classifier(video_path,player_detection_model)
generated_lines_video(video_path, "output.mp4", player_detection_model, team_classifier,keypoints_model,CONFIG,fps_output=1)
