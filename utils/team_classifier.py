import supervision as sv
from tqdm import tqdm
import torch

# Constantes
PLAYER_ID = 0  # Identifier l'ID correspondant aux joueurs
STRIDE = 10  # Nombre d'images à sauter entre chaque itération pour échantillonner les frames

import cv2
import random
import supervision as sv
from tqdm import tqdm

import cv2
import random
import supervision as sv
from tqdm import tqdm

import cv2
import numpy as np
import supervision as sv
import random
from tqdm import tqdm

def extract_player_crops(video_path, player_detection_model, max_frames=None, min_stride=3, max_stride=10):
    """
    Extracts player crops from random frames across the entire video with a random stride.

    Args:
    - video_path (str): Path to the video.
    - player_detection_model (RTDETR): Player detection model.
    - max_frames (int, optional): Maximum number of frames to process.
    - min_stride (int): Minimum stride value.
    - max_stride (int): Maximum stride value.

    Returns:
    - crops (list): List of cropped player images.
    """
    crops = []

    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return crops

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frame_index = 0  # Start at the first frame
    frame_count = 0

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Jump to the selected frame
        ret, frame = cap.read()
        if not ret:
            break  # Stop if frame can't be read

        # Run inference with RT-DETR model
        results = player_detection_model.predict(frame, conf=0.3)
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
            class_id=results[0].boxes.cls.detach().cpu().numpy(),
            confidence=results[0].boxes.conf.detach().cpu().numpy()
        )

        # Apply NMS to reduce false positives
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        # Filter player detections
        player_detections = detections[detections.class_id == PLAYER_ID]

        # Crop detected players
        for xyxy in player_detections.xyxy:
            crop = sv.crop_image(frame, xyxy)
            crops.append(crop)

        frame_count+=1

        # Stop if max frames reached
        if max_frames and frame_count >= max_frames:
            break

        # Choose a random stride for the next iteration
        stride = random.randint(min_stride, max_stride)
        frame_index += stride

    cap.release()  # Release video capture
    return crops



import cv2
import streamlit as st
import numpy as np
from PIL import Image

def show_crops(crops, cols=5):
    """
    Affiche les crops des joueurs dans l'interface Streamlit en sous-figures (subplots).
    
    Args:
    - crops (list): Liste des images crops des joueurs détectés.
    - cols (int): Nombre de colonnes dans la grille.
    """
    if not crops:
        st.warning("Aucun crop détecté.")
        return

    # Calcul du nombre de lignes nécessaires pour afficher toutes les images
    rows = 3

    # Afficher les images dans un format de grille
    for i in range(rows):
        # Sélectionner les images pour cette ligne
        row_crops = crops[i * cols:(i + 1) * cols]
        
        # Créer une colonne dans Streamlit
        cols_layout = st.columns(cols)
        
        for j, crop in enumerate(row_crops):
            if j < len(cols_layout):
                # Convertir le crop en format Image PIL pour l'affichage Streamlit
                pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                # Afficher le crop dans la colonne correspondante
                cols_layout[j].image(pil_image, use_column_width=True, caption=f"Crop {i * cols + j + 1}")


# Fonction pour ajuster (fit) le classifieur d'équipes
from utils.team import TeamClassifier

import os
import cv2
import numpy as np

def fit_team_classifier(crops, device="cpu"):
    """
    Entraîne un classifieur d'équipes à partir des crops extraits et retourne une image par cluster.

    Args:
    - crops (list): Liste des images crops des joueurs détectés.
    - device (str): Appareil à utiliser pour le calcul ('cuda' ou 'cpu').

    Returns:
    - team_classifier (TeamClassifier): Classifieur d'équipes entraîné.
    - cluster_images (dict): Dictionnaire {id_cluster: image représentative}.
    """
    if not crops:
        raise ValueError("La liste de crops est vide. Assurez-vous que les crops ont été correctement extraits.")
    
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    cluster_images = team_classifier.get_cluster_representatives(crops)

    # Sauvegarder les images des clusters
    cluster_dir = "clusters"
    os.makedirs(cluster_dir, exist_ok=True)
    cluster_paths = {}

    for cluster_id, img in cluster_images.items():
        path = os.path.join(cluster_dir, f"cluster_{cluster_id}.jpg")
        cv2.imwrite(path,img)
        cluster_paths[cluster_id] = path  # Stocke le chemin de l'image
    
    return team_classifier, cluster_paths

