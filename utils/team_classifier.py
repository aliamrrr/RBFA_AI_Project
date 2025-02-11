import supervision as sv
from tqdm import tqdm
import torch

# Constantes
PLAYER_ID = 0  # Identifier l'ID correspondant aux joueurs
STRIDE = 120  # Nombre d'images à sauter entre chaque itération pour échantillonner les frames

# Fonction pour extraire les crops des joueurs depuis une vidéo
def extract_player_crops(video_path, player_detection_model):
    """
    Extrait les crops des joueurs à partir d'une vidéo en utilisant un modèle de détection.
    
    Args:
    - video_path (str): Le chemin vers la vidéo.
    - player_detection_model (RTDETR): Le modèle de détection des joueurs.
    
    Returns:
    - crops (list): Liste des images crops des joueurs détectés.
    """
    # Liste pour collecter les crops
    crops = []

    # Générateur de frames avec un pas fixe
    frame_generator = sv.get_video_frames_generator(
        source_path=video_path, stride=STRIDE
    )

    # Parcourir les frames avec une barre de progression
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        # Exécuter l'inférence avec le modèle RT-DETR
        results = player_detection_model.predict(frame, conf=0.3)  # Exécution de l'inférence avec confiance
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
            class_id=results[0].boxes.cls.detach().cpu().numpy(),
            confidence=results[0].boxes.conf.detach().cpu().numpy()
        )

        # Appliquer NMS directement sur les détections pour réduire les faux positifs
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        # Filtrer uniquement les détections correspondant à PLAYER_ID
        player_detections = detections[detections.class_id == PLAYER_ID]

        # Recadrer (crop) les détections des joueurs
        for xyxy in player_detections.xyxy:
            crop = sv.crop_image(frame, xyxy)  # Recadrer l'image autour de la détection
            crops.append(crop)

    # Retourner les crops collectés
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

def fit_team_classifier(crops, device="cpu"):
    """
    Entraîne un classifieur d'équipes à partir des crops extraits.
    
    Args:
    - crops (list): Liste des images crops des joueurs détectés.
    - device (str): Appareil à utiliser pour le calcul ('cuda' ou 'cpu').
    
    Returns:
    - team_classifier (TeamClassifier): Classifieur d'équipes entraîné.
    """
    if not crops:
        raise ValueError("La liste de crops est vide. Assurez-vous que les crops ont été correctement extraits.")
    
    # Initialiser le classifieur d'équipes
    team_classifier = TeamClassifier(device=device)
    
    # Entraîner le classifieur avec les crops fournis
    team_classifier.fit(crops)
    
    # Retourner le classifieur entraîné
    return team_classifier

