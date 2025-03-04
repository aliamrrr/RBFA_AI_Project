import numpy as np
import supervision as sv
from utils.annotators import draw_pitch, draw_line_on_pitch,draw_points_on_pitch, draw_pitch_voronoi_diagram, draw_paths_and_hull_on_pitch,draw_arrow_on_pitch
from utils.soccer import SoccerPitchConfiguration
from utils.view import ViewTransformer
import os
import supervision as sv
import cv2
from ultralytics import RTDETR
from utils.models import load_player_detection_model,load_field_detection_model
import pandas as pd



CONFIG = SoccerPitchConfiguration()


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

# Class IDS
PLAYER_ID = 0
GOALKEEPER_ID = 1
BALL_ID = 2
REFEREE_ID = 3
SIDE_REFEREE_ID = 4
STAFF_MEMBER_ID = 5

red = sv.Color.from_hex('FF0000')
yellow = sv.Color.from_hex('FFFF00')


direct_link = "https://drive.google.com/file/d/1FuibHhLGI7PvaZxSPrxhtdQxveyqdKTg/view?usp=drive_link"
# load models
model = load_player_detection_model()
keypoints_model = load_field_detection_model()

def process_frame(frame, team_classifier):
    """
    Fonction qui effectue l'inférence sur une seule frame, détecte les objets,
    puis les annote avec leurs couleurs respectives.

    Args:
    - frame (ndarray): La frame de la vidéo à traiter.
    - team_classifier (object): Le modèle de classification des équipes pour les joueurs.

    Retourne:
    - frame_annotated (ndarray): La frame annotée avec les détections.
    """
    # Appliquer l'inférence avec le modèle RT-DETR pour détecter les objets
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS (Non-Maximum Suppression) pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type (joueurs, arbitres, etc.)
    referees_detections = detections[detections.class_id == REFEREE_ID]
    side_referees_detections = detections[detections.class_id == SIDE_REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]
    staff_members_detections = detections[detections.class_id == STAFF_MEMBER_ID]
    players_detections = detections[detections.class_id == PLAYER_ID]
    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    # Résoudre les gardiens de but en fonction de l'équipe
    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections
    )

    # Fusionner les détections des joueurs et gardiens de but
    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

    # Annoter la frame originale
    annotated_frame = frame.copy()

    # Définir les couleurs d'annotation pour chaque type d'objet
    class_colors = {
        "referee": (0, 255, 255),  # Couleur pour les arbitres (orange)
        "side_referee": (128, 0, 128),  # Couleur pour les arbitres latéraux (violet)
        "ball": (255, 0, 0),  # Couleur pour le ballon (rouge)
        "staff_member": (255, 255, 153)  # Couleur pour les membres du staff (jaune clair)
    }

    # Annoter les arbitres
    for i in range(len(referees_detections)):
        x1, y1, x2, y2 = map(int, referees_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["referee"], 2)
        label = f"Referee ({referees_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["referee"], 2)

    # Annoter les arbitres latéraux
    for i in range(len(side_referees_detections)):
        x1, y1, x2, y2 = map(int, side_referees_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["side_referee"], 2)
        label = f"Side Referee ({side_referees_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["side_referee"], 2)

    # Annoter les ballons
    for i in range(len(ball_detections)):
        x1, y1, x2, y2 = map(int, ball_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["ball"], 2)
        label = f"Ball ({ball_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["ball"], 2)

    # Annoter les membres du staff
    for i in range(len(staff_members_detections)):
        x1, y1, x2, y2 = map(int, staff_members_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["staff_member"], 2)
        label = f"Staff Member ({staff_members_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["staff_member"], 2)

    # Annoter les joueurs dynamiquement en fonction de leur équipe
    for i in range(len(all_detections)):
        x1, y1, x2, y2 = map(int, all_detections.xyxy[i])
        team_color = (0, 255, 0) if all_detections.class_id[i] == 1 else (255, 0, 0)  # Vert pour l'équipe 1, Bleu pour l'équipe 0
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), team_color, 2)
        label = f"{'Team 1' if all_detections.class_id[i] == 1 else 'Team 0'}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

    return annotated_frame

def annotate_player_name(frame, click_coords, player_name):
    """
    Annoter l'image avec le nom du joueur de manière stylée.

    Args:
    - frame (ndarray): Image à annoter.
    - click_coords (tuple): Coordonnées (x, y) du clic.
    - player_name (str): Nom du joueur à afficher.

    Retourne:
    - frame_annotated (ndarray): Image annotée.
    """
    x, y = int(click_coords[0]), int(click_coords[1])
    annotated_frame = frame.copy()
    
    font = cv2.FONT_HERSHEY_TRIPLEX  # Police stylée
    font_scale = 0.5  # Taille plus grande
    color = (255, 255, 255)  # Blanc
    thickness = 1

    # Décalage vers le haut et à gauche
    offset_x, offset_y = -22, -50 
    text_position = (x + offset_x, y + offset_y)

    # Ajouter une ombre noire pour la visibilité
    cv2.putText(annotated_frame, player_name, (text_position[0] + 2, text_position[1] + 2), 
                font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)

    # Ajouter le texte en blanc par-dessus l'ombre
    cv2.putText(annotated_frame, player_name, text_position, 
                font, font_scale, color, thickness, cv2.LINE_AA)

    return annotated_frame


import cv2
import numpy as np
import scipy.spatial

def process_frame_with_convex_hull(frame, team_classifier, team_id=None):
    """
    Fonction qui effectue l'inférence sur une seule frame, détecte les objets,
    puis les annote avec leurs couleurs respectives, tout en traçant des convex hulls
    autour des joueurs des équipes demandées.

    Args:
    - frame (ndarray): La frame de la vidéo à traiter.
    - team_classifier (object): Le modèle de classification des équipes pour les joueurs.
    - team_id (list): Liste des identifiants des équipes à tracer (0 ou 1). Si None, trace les deux équipes.

    Retourne:
    - frame_annotated (ndarray): La frame annotée avec les détections et les convex hulls.
    """
    # Appliquer l'inférence avec le modèle RT-DETR pour détecter les objets
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS (Non-Maximum Suppression) pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type (joueurs, arbitres, etc.)
    players_detections = detections[detections.class_id == PLAYER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    # Annoter la frame originale
    annotated_frame = frame.copy()

    # Sélectionner les joueurs de chaque équipe
    team_players = {0: [], 1: []}
    for i in range(len(players_detections)):
        x1, y1, x2, y2 = map(int, players_detections.xyxy[i])
        if players_detections.class_id[i] == 0:
            team_players[0].append(((x1 + x2) / 2, (y1 + y2) / 2))
        elif players_detections.class_id[i] == 1:
            team_players[1].append(((x1 + x2) / 2, (y1 + y2) / 2))

    # Fonction pour tracer un convex hull rempli avec plus de transparence et des couleurs différentes
    def draw_filled_convex_hull(annotated_frame, points, color, alpha=0.15):
        if points:
            hull = scipy.spatial.ConvexHull(points)
            hull_points = [tuple(map(int, hull.points[vertex])) for vertex in hull.vertices]

            # Créer une image temporaire pour dessiner le remplissage avec transparence
            overlay = annotated_frame.copy()
            cv2.fillConvexPoly(overlay, np.array(hull_points, dtype=np.int32), color)

            # Superposer l'overlay avec transparence
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # Tracer le contour avec des couleurs distinctes pour chaque équipe
            line_color = (255, 255, 0) if color == (0, 255, 0) else (255, 0, 0)  # Bleu pour équipe 0, Jaune pour équipe 1
            for i in range(len(hull_points)):
                cv2.line(annotated_frame, hull_points[i], hull_points[(i+1) % len(hull_points)], line_color, 1)

    # Tracer les convex hulls pour chaque équipe si nécessaire
    if team_id is None or team_id == 0:
        draw_filled_convex_hull(annotated_frame, team_players[0], (0, 0, 255))  # Rouge pour équipe 0
    if team_id is None or team_id == 1:
        draw_filled_convex_hull(annotated_frame, team_players[1], (0, 255, 0))  # Vert pour équipe 1

    # Définir les couleurs d'annotation pour les joueurs
    team_color = (0, 255, 0) if team_id == 1 else (255, 0, 0)  # Vert pour l'équipe 1, Bleu pour l'équipe 0

    return annotated_frame




def draw_radar_view(frame, CONFIG, team_classifier, type='tactical',color_1=red,color_2=yellow):
    """
    Fonction pour afficher la vue radar de la position des joueurs et de la balle sur le terrain.
    
    Args:
    - frame (ndarray): La frame de la vidéo.
    - CONFIG (SoccerPitchConfiguration): Configuration du terrain.
    - team_classifier (Classifier): Classificateur pour prédire l'équipe des joueurs.
    - FIELD_DETECTION_MODEL (Model): Modèle de détection de terrain pour la projection.
    
    Retourne:
    - frame (ndarray): La frame avec la vue radar affichée.
    """
    # Traiter les détections avec le modèle RT-DETR
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type
    players_detections = detections[detections.class_id == PLAYER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]
    referees_detections = detections[detections.class_id == REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]

    # Fusionner toutes les détections
    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

    # Projeter les positions sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    # Projection des positions de la balle
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    # Projection des positions des joueurs
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    # Dessiner la vue radar (petite vue du terrain)
    annotated_frame = draw_pitch(CONFIG)

    if type == "tactical":
        # Dessiner la balle projetée
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=10,
            thickness=2,
            pitch=annotated_frame)

        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 0],
            face_color=sv.Color.from_hex('FFFF00'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
        
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 1],
            face_color=sv.Color.from_hex('FF0000'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
    
    elif type == "voronoi":
        print('we want voronoi!')
        annotated_frame = draw_pitch_voronoi_diagram(
            config=CONFIG,
            team_1_xy=pitch_players_xy[players_detections.class_id == 0],
            team_2_xy=pitch_players_xy[players_detections.class_id == 1],
            team_1_color=color_1,
            team_2_color=color_2,
            pitch=annotated_frame)

    return annotated_frame


import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def save_team_line_clusters(players_xy, team_id, output_dir="clusters/"):
    os.makedirs(output_dir, exist_ok=True)

    if len(players_xy) < 3:
        print("Pas assez de joueurs pour former des lignes.")
        return

    # Clustering basé sur la position X
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(players_xy[:, 0].reshape(-1, 1))

    # Regrouper les joueurs par cluster
    cluster_dict = {0: [], 1: [], 2: []}
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(players_xy[i])

    # Trier les clusters par position X moyenne (du plus bas au plus haut)
    sorted_clusters = sorted(cluster_dict.items(), key=lambda x: np.mean([p[0] for p in x[1]]))

    cluster_names = ["defense", "middle", "attack"]
    
    for i, (cluster_id, players) in enumerate(sorted_clusters):
        players = np.array(players)

        # Calcul du barycentre
        barycenter = np.mean(players, axis=0) if len(players) > 0 else [np.nan, np.nan]

        # Construire la ligne de données avec max 4 joueurs
        max_players = 4
        df_data = [players[j].tolist() if j < len(players) else [np.nan, np.nan] for j in range(max_players)]

        # Convertir en DataFrame avec un format propre
        df = pd.DataFrame([df_data], columns=[f"Player{j+1}" for j in range(max_players)])
        df["Barycenter"] = [barycenter.tolist()]

        # Déterminer si le fichier existe déjà
        csv_filename = f"{output_dir}team_{team_id}_{cluster_names[i]}.csv"
        append_mode = os.path.exists(csv_filename)  # True si le fichier existe déjà

        # Sauvegarde CSV en mode append si le fichier existe
        df.to_csv(csv_filename, mode="a", header=not append_mode, index=False)
        print(f"Cluster {cluster_names[i]} ajouté à { csv_filename}")


def is_pressing(players, players_pos, ball_pos, team_id,zone,team_possession):
    pressure = False

    if team_possession and zone=="back":
            # opponents
            opponents_pos = players_pos[players.class_id != team_id]

            # get opponent - ball distances 
            distances = np.linalg.norm(opponents_pos-ball_pos, axis=1)
            print(distances)

            # get closest opponent
            closest_opponent = opponents_pos[np.argmin(distances)]

            distance = np.min(distances)
            print(distance)

            if distance <= 300:
                pressure = True

    elif team_possession == False:
        team_players_pos = players_pos[players.class_id == team_id]

        distances = np.linalg.norm(team_players_pos - ball_pos, axis=1)
        print("Our players' distances:", distances)

        distance = np.min(distances)
        print("Closest teammate distance:", distance)

        if distance <= 300:
            pressure = True

    return pressure

import numpy as np

def is_kickoff(ball_pos, players_pos, players_teams, pitch_size=(12000, 7000), center_radius=200, threshold=80):
    """
    Détecte un kick-off en analysant la répartition des joueurs autour du ballon.

    Args:
        ball_pos (np.array): Coordonnées (x, y) du ballon.
        players_pos (np.array): Coordonnées (x, y) des joueurs.
        players_teams (np.array): Liste des équipes des joueurs (0 ou 1).
        pitch_size (tuple): Dimensions du terrain (largeur, hauteur).
        center_radius (int): Distance max pour considérer le ballon au centre.
        threshold (int): Seuil (%) de répartition des joueurs pour valider un kick-off.

    Returns:
        bool: True si c'est un kick-off, False sinon.
    """
    # 1️⃣ Vérifier si le ballon est au centre
    center_x, center_y = pitch_size[0] // 2, pitch_size[1] // 2
    print("distance balle : "+ str(np.linalg.norm(ball_pos - np.array([center_x, center_y]))))
    if np.linalg.norm(ball_pos - np.array([center_x, center_y])) > center_radius:
        return False

    # 2️⃣ Déterminer les joueurs à gauche et à droite de la ligne verticale passant par le ballon
    left_team0 = right_team0 = left_team1 = right_team1 = 0

    for (x, y), team in zip(players_pos, players_teams):
        if x < ball_pos[0]:  # Côté gauche
            if team == 0:
                left_team0 += 1
            else:
                left_team1 += 1
        else:  # Côté droit
            if team == 0:
                right_team0 += 1
            else:
                right_team1 += 1

    # 3️⃣ Calculer les pourcentages de joueurs de chaque équipe de chaque côté
    total_team0 = left_team0 + right_team0
    total_team1 = left_team1 + right_team1

    if total_team0 == 0 or total_team1 == 0:
        return False  # Impossible de juger

    pct_team0_left = (left_team0 / total_team0) * 100
    pct_team1_left = (left_team1 / total_team1) * 100

    print("LEFT")
    print(pct_team0_left)
    print(pct_team1_left)

    # 4️⃣ Vérifier si la répartition est proche de 98% / 2%
    if (pct_team0_left >= threshold and pct_team1_left <= (100 - threshold)) or \
       (pct_team1_left >= threshold and pct_team0_left <= (100 - threshold)):
        return True  # Kick-off détecté !

    return False  # Sinon, ce n'est pas un kick-off



from sklearn.cluster import KMeans
import itertools

def draw_team_connections(frame, config, team_classifier, team_id,color_1=red,color_2=yellow):
    """
    Ajoute des lignes reliant les joueurs d'un même rôle (défense, milieu, attaque)
    pour une équipe donnée.
    
    Args:
    - frame (ndarray): Frame de la vidéo.
    - config (SoccerPitchConfiguration): Configuration du terrain.
    - team_classifier (Classifier): Classificateur d'équipe.
    - keypoints_model (Model): Modèle de points clés.
    - model (Model): Modèle de détection.
    - team_id (int): ID de l'équipe à analyser.
    
    Retourne:
    - annotated_frame (ndarray): Frame annotée avec les connexions entre joueurs.
    """
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type
    players_detections = detections[detections.class_id == PLAYER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]
    referees_detections = detections[detections.class_id == REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]

    # Fusionner toutes les détections
    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

    # Projeter les positions sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(config.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    # Projection des positions de la balle
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    # Projection des positions des joueurs
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    # Dessiner la vue radar (petite vue du terrain)
    annotated_frame = draw_pitch(config)

    team_players_xy = pitch_players_xy[players_detections.class_id == team_id]
    
    if len(team_players_xy) < 3:
        return annotated_frame  # Pas assez de joueurs pour former des lignes
    
    # Clustering basé sur la position X pour déterminer les lignes
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(team_players_xy[:, 0].reshape(-1, 1))
    
    colors = {
    0: (0, 255, 0),   # Bleu
    1: (0, 255, 0),   # Vert
    2: (0, 255, 0)    # Rouge
}

    
    for cluster_id in range(3):
        role_players = team_players_xy[clusters == cluster_id]
        
        # Trier les joueurs de gauche à droite (par position X)
        role_players = role_players[np.argsort(role_players[:, 1])]

        # Relier chaque joueur à son suivant
        for i in range(len(role_players) - 1):
            annotated_frame = draw_line_on_pitch(
                config=config,
                xy1=role_players[i],
                xy2=role_players[i + 1],
                color=sv.Color(*colors[cluster_id]),  # Déstructure le tuple (r, g, b)
                thickness=2,
                pitch=annotated_frame
            )

    annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 0],
            face_color=color_1,
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
        
    annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 1],
            face_color=color_2,
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
    
    annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=10,
            thickness=2,
            pitch=annotated_frame)
    
    return annotated_frame
    





import cv2
import tempfile
import numpy as np
import streamlit as st
from moviepy.video.io import ImageSequenceClip

def process_and_save_video(tmp_file_path, start_time, end_time, fps, team_classifier):
    cap = cv2.VideoCapture(tmp_file_path)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = draw_radar_view(frame, CONFIG, team_classifier, 'tactical')
        frames.append(annotated_frame)
    
    cap.release()
    
    # Sauvegarde en vidéo
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    clip.write_videofile(output_video_path, codec="libx264")

def passes_options(frame, CONFIG, team_classifier, passes_mode,color_1=red,color_2=yellow):
    # Traiter les détections avec le modèle RT-DETR
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type
    players_detections = detections[detections.class_id == PLAYER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]
    referees_detections = detections[detections.class_id == REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]

    # Fusionner toutes les détections
    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

    # Projeter les positions sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    # Projection des positions de la balle
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    # Projection des positions des joueurs
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    # Dessiner la vue radar (petite vue du terrain)
    annotated_frame = draw_pitch(CONFIG)

    if passes_mode == 'build':
        print('mode 1')
        if len(ball_detections) > 0:
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            distances = np.linalg.norm(pitch_players_xy - pitch_ball_xy, axis=1)
            closest_player_idx = np.argmin(distances)
            closest_player_team = players_detections.class_id[closest_player_idx]

            teammates_idx = np.where(players_detections.class_id == closest_player_team)[0]
            teammates_xy = pitch_players_xy[teammates_idx]

            opponent_team = 1 - closest_player_team
            opponent_idx = np.where(players_detections.class_id == opponent_team)[0]
            opponent_xy = pitch_players_xy[opponent_idx]

            pressure_radius = 500
            pressures = []

            for teammate in teammates_xy:
                pressure = sum(np.linalg.norm(teammate - opponent) < pressure_radius for opponent in opponent_xy)
                pressures.append(pressure)

            for i, teammate in enumerate(teammates_xy):
                if np.array_equal(teammate, pitch_ball_xy[0]):
                    continue

                distance = np.linalg.norm(teammate - pitch_ball_xy[0])
                pressure_factor = pressures[i]

                # Couleur dynamique selon la difficulté de la passe
                difficulty = np.clip(
                    1 - np.exp(-distance / 3000) * (1 + np.exp(pressure_factor)) * (1 - 0.5 * pressure_factor),
                    0,
                    1
                )


                print('distance', distance)
                print('pressure_factor', pressure_factor)
                print('difficulty', difficulty)

                color = sv.Color(
                    int(difficulty * 255),
                    int((1 - difficulty) * 255),  # Red component: dimmer as difficulty increases
                    0  # Blue component: always 0
                )

                annotated_frame = draw_arrow_on_pitch(
                    config=CONFIG,
                    xy_start=[pitch_ball_xy[0]],
                    xy_end=[teammate],
                    color=color,
                    thickness=7,  # Épaisseur fixe
                    pitch=annotated_frame
                )

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.from_hex('FFD700'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    thickness=2,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 0],
                    face_color=color_1,
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 1],
                    face_color=color_2,
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

    elif passes_mode == 'interception':
        print('mode 2')
        if len(ball_detections) > 0:
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            distances = np.linalg.norm(pitch_players_xy - pitch_ball_xy, axis=1)
            closest_player_idx = np.argmin(distances)
            closest_player_team = players_detections.class_id[closest_player_idx]

            teammates_idx = np.where(players_detections.class_id == closest_player_team)[0]
            teammates_xy = pitch_players_xy[teammates_idx]

            opponent_team = 1 - closest_player_team
            opponent_idx = np.where(players_detections.class_id == opponent_team)[0]
            opponent_xy = pitch_players_xy[opponent_idx]

            pressure_radius = 500
            pressures = []

            # Calcul de la pression pour chaque coéquipier
            for teammate in teammates_xy:
                pressure = sum(np.linalg.norm(teammate - opponent) < pressure_radius for opponent in opponent_xy)
                pressures.append(pressure)

            # Calcul de la difficulté de chaque passe avec interception
            for i, teammate in enumerate(teammates_xy):
                if np.array_equal(teammate, pitch_ball_xy[0]):
                    continue

                distance = np.linalg.norm(teammate - pitch_ball_xy[0])
                pressure_factor = pressures[i]

                # Trajectoire de la balle
                ball_to_teammate_vector = teammate - pitch_ball_xy[0]
                ball_to_teammate_vector_normalized = ball_to_teammate_vector / np.linalg.norm(ball_to_teammate_vector)

                # Détection d'interception
                interception_radius = 500  # Rayon autour de la trajectoire de la balle pour l'interception
                interception_risk = 0

                for opponent in opponent_xy:
                    # Projection du joueur adverse sur la trajectoire de la balle
                    projection = np.dot(opponent - pitch_ball_xy[0], ball_to_teammate_vector_normalized)
                    projected_point = pitch_ball_xy[0] + projection * ball_to_teammate_vector_normalized

                    # Vérification si le joueur adverse est proche de la trajectoire
                    if np.linalg.norm(opponent - projected_point) < interception_radius:
                        interception_risk += 1  # Plus d'opposants proches = plus grand risque d'interception

                    difficulty = np.clip(
                        (np.exp(-distance / 4000) * (1 + np.exp(pressure_factor)) * (1 - 0.5 * pressure_factor)) 
                        * (1 + (distance / 4000))   # Favoriser la distance longue
                        * (1 - (interception_risk * 0.5)),  # Réduire la difficulté en fonction du risque d'interception (plus faible = meilleur)
                        0,
                        1
                    )

                    difficulty = 1 - difficulty

                    print('sidtance  '+ str(distance) + 'risk  ' + str(interception_risk) + 'diff  ' + str(difficulty))

                # Calcul de la couleur dynamique selon la difficulté de la passe
                color = sv.Color(
                    int(difficulty * 255),
                    int((1 - difficulty) * 255),  # Red component: dimmer as difficulty increases
                    0  # Blue component: always 0
                )

                # Dessiner la flèche pour la passe
                annotated_frame = draw_arrow_on_pitch(
                    config=CONFIG,
                    xy_start=[pitch_ball_xy[0]],
                    xy_end=[teammate],
                    color=color,
                    thickness=7,  # Épaisseur fixe
                    pitch=annotated_frame
                )

                # Dessiner les points sur le terrain
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.from_hex('FFD700'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    thickness=2,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 0],
                    face_color=color_1,
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 1],
                    face_color=color_2,
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

    return annotated_frame



def calculate_optimal_passes(frame, CONFIG, team_classifier, max_passes=3,color_1=red,color_2=yellow):
    # Traitement des détections
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Extraction des différentes entités
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    ball_detections = detections[detections.class_id == BALL_ID]

    # Projection sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

    # Initialisation des positions
    current_ball_pos = transformer.transform_points(ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))[0]
    all_players_xy = transformer.transform_points(players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
    team_ids = players_detections.class_id

    # Configuration du terrain
    pitch_min_y = min(CONFIG.vertices, key=lambda p: p[1])[1]
    pitch_max_y = max(CONFIG.vertices, key=lambda p: p[1])[1]
    max_y_length = pitch_max_y - pitch_min_y
    print(max_y_length)

    # Détermination de l'équipe en possession
    distances = np.linalg.norm(all_players_xy - current_ball_pos, axis=1)
    closest_player_idx = np.argmin(distances)
    current_team = team_ids[closest_player_idx]

    # Paramètres d'optimisation
    pressure_radius = 500
    interception_radius = 500
    passes_sequence = []
    used_players = []

    for _ in range(max_passes):
        # Filtrage des joueurs disponibles
        teammates_mask = (team_ids == current_team)
        available_mask = ~np.isin(np.arange(len(all_players_xy)), used_players)
        candidates_mask = teammates_mask & available_mask
        
        if not np.any(candidates_mask):
            break

        candidates_xy = all_players_xy[candidates_mask]
        opponents_xy = all_players_xy[team_ids != current_team]

        # Calcul des difficultés de passe
        difficulties = []
        for candidate in candidates_xy:
            # Calcul de la distance
            distance = np.linalg.norm(candidate - current_ball_pos)
            
            # Calcul de la pression
            pressure = sum(np.linalg.norm(candidate - opponent) < pressure_radius for opponent in opponents_xy)
            
            # Calcul du risque d'interception
            pass_vector = candidate - current_ball_pos
            pass_vector_norm = pass_vector / np.linalg.norm(pass_vector)
            interceptions = 0
            
            for opponent in opponents_xy:
                projection = np.dot(opponent - current_ball_pos, pass_vector_norm)
                if projection < 0 or projection > np.linalg.norm(pass_vector):
                    continue
                closest_point = current_ball_pos + projection * pass_vector_norm
                if np.linalg.norm(opponent - closest_point) < interception_radius:
                    interceptions += 1
            
            # Calcul de la difficulté
            difficulty = (distance / 4000) * (1 + pressure) * (1 + interceptions)
            
            # Bonus pour les passes en avant
            delta_y = (candidate[0] - current_ball_pos[0])

            if delta_y>0 :
                difficulty = difficulty/1000
            else :
                difficulty = difficulty*100
            
            print(difficulty)

            
            difficulties.append(difficulty)

        # Sélection de la meilleure passe
        best_idx = np.argmin(difficulties)
        print('def')
        print(difficulties[best_idx])
        best_receiver = candidates_xy[best_idx]
        
        # Mise à jour pour la prochaine itération
        passes_sequence.append((current_ball_pos, best_receiver))
        used_players.append(np.where(candidates_mask)[0][best_idx])
        current_ball_pos = best_receiver

    # Dessin du terrain
    annotated_frame = draw_pitch(CONFIG)

    # Dessin des passes optimales
    for pass_start, pass_end in passes_sequence:
        annotated_frame = draw_arrow_on_pitch(
            config=CONFIG,
            xy_start=[pass_start],
            xy_end=[pass_end],
            color=sv.Color.GREEN,
            thickness=9,
            pitch=annotated_frame
        )

    # Dessin des joueurs
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=all_players_xy[team_ids == 0],
        face_color=color_1,
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame)

    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=all_players_xy[team_ids == 1],
        face_color=color_2,
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame)

    return annotated_frame

def calculate_realistic_optimal_passes(frame, CONFIG, team_classifier, max_passes=2,color_1=red,color_2=yellow):
    # Traitement des détections
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Extraction des différentes entités
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    ball_detections = detections[detections.class_id == BALL_ID]

    # Projection sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

    # Initialisation des positions
    current_ball_pos = transformer.transform_points(ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))[0]
    pitch_ball_pos = current_ball_pos.copy()
    all_players_xy = transformer.transform_points(players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
    team_ids = players_detections.class_id

    # Configuration dynamique
    DEFENSIVE_PARAMS = {
        'reaction_radius': 800,
        'aggressivity': 30,
        'pass_duration': 1.2,
        'player_speed': 10.0
    }

    # Initialisation des positions dynamiques
    dynamic_opponents = all_players_xy[team_ids != team_ids[np.argmin(np.linalg.norm(all_players_xy - current_ball_pos, axis=1))]].copy()
    reaction_movements = []
    original_teammates = all_players_xy[team_ids == team_ids[np.argmin(np.linalg.norm(all_players_xy - current_ball_pos, axis=1))]].copy()

    passes_sequence = []
    used_players = []
    current_possession = team_ids[np.argmin(np.linalg.norm(all_players_xy - current_ball_pos, axis=1))]

    for pass_num in range(max_passes):
        # Filtrage des joueurs disponibles
        teammates_mask = (team_ids == current_possession)
        available_mask = ~np.isin(np.arange(len(all_players_xy)), used_players)
        candidates_mask = teammates_mask & available_mask
        
        if not np.any(candidates_mask):
            break

        candidates_xy = all_players_xy[candidates_mask]
        opponents_xy = dynamic_opponents

        # Calcul des difficultés de passe
        difficulties = []
        for candidate in candidates_xy:
            distance = np.linalg.norm(candidate - current_ball_pos)
            
            # Calcul pression et interception
            pressure = sum(np.linalg.norm(candidate - opp) < 500 for opp in opponents_xy)
            pass_vector = candidate - current_ball_pos
            pass_vector_norm = pass_vector / np.linalg.norm(pass_vector)
            interceptions = 0
            
            for opp in opponents_xy:
                projection = np.dot(opp - current_ball_pos, pass_vector_norm)
                if 0 < projection < np.linalg.norm(pass_vector):
                    closest_point = current_ball_pos + projection * pass_vector_norm
                    if np.linalg.norm(opp - closest_point) < 400:
                        interceptions += 1
            
            # Calcul difficulté avec bonus directionnel
            delta_x = candidate[0] - current_ball_pos[0]
            direction_bonus = 20 if delta_x < 0 else 0.5
            difficulty = (distance / 3500) * (1 + pressure) * (1 + interceptions) * direction_bonus
            difficulties.append(difficulty)

        best_idx = np.argmin(difficulties)
        best_receiver = candidates_xy[best_idx]
        passes_sequence.append((current_ball_pos, best_receiver))
        used_players.append(np.where(candidates_mask)[0][best_idx])

        # Simulation mouvement défenseurs
        if len(opponents_xy) > 0:
            max_move = DEFENSIVE_PARAMS['player_speed'] * DEFENSIVE_PARAMS['pass_duration']
            distances_to_receiver = np.linalg.norm(dynamic_opponents - best_receiver, axis=1)
            
            # Sélection des 2 plus proches + 1 aléatoire
            closest_defenders = np.argsort(distances_to_receiver)[:2]
            random_defender = np.random.choice(len(dynamic_opponents)) if len(dynamic_opponents) > 2 else None
            
            for def_idx in list(closest_defenders) + [random_defender]:
                if def_idx is None or def_idx >= len(dynamic_opponents):
                    continue
                
                defender_pos = dynamic_opponents[def_idx]
                direction = best_receiver - defender_pos
                move_dist = min(np.linalg.norm(direction), max_move)

                print((direction / np.linalg.norm(direction)) * move_dist * DEFENSIVE_PARAMS['aggressivity'])
                new_pos = defender_pos + (direction / np.linalg.norm(direction)) * move_dist * DEFENSIVE_PARAMS['aggressivity']
                new_pos = np.clip(new_pos, [0, 0], [12000, 7000])
                
                reaction_movements.append({
                    'start': defender_pos.copy(),
                    'end': new_pos.copy(),
                    'team': 'defense'
                })
                dynamic_opponents[def_idx] = new_pos

        current_ball_pos = best_receiver

    # Dessin du terrain
    annotated_frame = draw_pitch(CONFIG)

    print(reaction_movements)

    # Dessin des réactions défensives
    for movement in reaction_movements:
        annotated_frame = draw_arrow_on_pitch(
            config=CONFIG,
            xy_start=[movement['start']],
            xy_end=[movement['end']],
            color=sv.Color.RED,
            thickness=8,
            pitch=annotated_frame
        )

    # Dessin des passes
    for pass_start, pass_end in passes_sequence:
        annotated_frame = draw_arrow_on_pitch(
            config=CONFIG,
            xy_start=[pass_start],
            xy_end=[pass_end],
            color=sv.Color.GREEN,
            thickness=8,
            pitch=annotated_frame
        )

    # Dessin des joueurs
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=dynamic_opponents,
        face_color=color_2,
        edge_color=sv.Color.WHITE,
        radius=14,
        thickness=1,
        pitch=annotated_frame)

    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=original_teammates,
        face_color=color_1,
        edge_color=sv.Color.WHITE,
        radius=14,
        thickness=1,
        pitch=annotated_frame)

    return annotated_frame

def extract_tracking_data(frame, CONFIG, team_classifier,tracker):
    """
    Fonction pour extraire les informations de suivi des joueurs et de la balle sur le terrain.

    Args:
    - frame (ndarray): La frame de la vidéo.
    - CONFIG (SoccerPitchConfiguration): Configuration du terrain.
    - team_classifier (Classifier): Classificateur pour prédire l'équipe des joueurs.

    Retourne:
    - players_tracking (DataFrame): Suivi des joueurs avec colonnes ['Team', 'player_id', 'player_x', 'player_y'].
    - ball_tracking (DataFrame): Suivi de la balle avec colonnes ['Team_possession', 'ball_x', 'ball_y'].
    """

    # Détection des objets sur la frame
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Mettre à jour le tracker avec les détections actuelles
    tracked_detections = tracker.update_with_detections(detections=detections)

    # Regrouper les détections par type
    referees_detections = tracked_detections[tracked_detections.class_id == REFEREE_ID]
    ball_detections = tracked_detections[tracked_detections.class_id == BALL_ID]
    players_detections = tracked_detections[tracked_detections.class_id == PLAYER_ID]

    # Associer les joueurs à leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    # Créer le dossier "crops" si nécessaire
    crops_directory = "crops"
    if not os.path.exists(crops_directory):
        os.makedirs(crops_directory)

    # Enregistrer chaque crop dans un sous-dossier basé sur tracker_id
    for i, crop in enumerate(players_crops):
        player_id = tracked_detections[i].tracker_id
        tracker_id_directory = os.path.join(crops_directory, str(player_id))
        
        # Créer un dossier pour chaque tracker_id si nécessaire
        if not os.path.exists(tracker_id_directory):
            os.makedirs(tracker_id_directory)

        # Sauvegarder l'image cropée dans le sous-dossier du tracker_id
        crop_filename = os.path.join(tracker_id_directory, f"crop_{i}.jpg")
        cv2.imwrite(crop_filename, crop)

    # Résolution des équipes pour les gardiens
    goalkeepers_detections = tracked_detections[tracked_detections.class_id == GOALKEEPER_ID]
    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections)

    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

    # Projeter les positions sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    # Suivi des joueurs
    players_tracking_data = []
    for i, xy in enumerate(pitch_players_xy):
        if i >= len(players_detections.class_id):
            print(f"Skipping index {i} as it exceeds players_detections size.")
            continue

        player_id = tracked_detections[i].tracker_id
        team = players_detections.class_id[i]
        x, y = xy
        players_tracking_data.append({'Team': team, 'player_id': player_id, 'player_x': x, 'player_y': y})

    players_tracking = pd.DataFrame(players_tracking_data)

    # Suivi de la balle
    if len(pitch_ball_xy) > 0:
        ball_x, ball_y = pitch_ball_xy[0]
        # Déterminer le joueur le plus proche de la balle
        distances = np.linalg.norm(pitch_players_xy - np.array([ball_x, ball_y]), axis=1)
        closest_player_idx = np.argmin(distances)
        team_possession = players_tracking.iloc[closest_player_idx]['Team']
    else:
        ball_x, ball_y = None, None
        team_possession = None

    ball_tracking = pd.DataFrame([{
        'Team_possession': team_possession,
        'ball_x': ball_x,
        'ball_y': ball_y
    }])

    return players_tracking, ball_tracking




def annotate_frame_with_hulls(frame,team_id,team_classifier,CONFIG,color_1=red,color_2=yellow):
    """
    Annotates a frame by drawing pitch hulls for players and other detections.

    Args:
        frame (np.ndarray): Input video frame.
        detections (sv.Detections): Detections from the RT-DETR model.
        team_classifier (Any): Classifier to determine team membership.
        keypoints_model (Any): Model to infer keypoints for pitch projection.
        config (SoccerPitchConfiguration): Configuration of the pitch.
        transformer (ViewTransformer): Transformer for pitch projection.
        player_id (int): Class ID for players.
        goalkeeper_id (int): Class ID for goalkeepers.
        referee_id (int): Class ID for referees.
        ball_id (int): Class ID for the ball.

    Returns:
        np.ndarray: Annotated frame with hulls drawn.
    """

    # Traiter les détections avec le modèle RT-DETR
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type
    players_detections = detections[detections.class_id == PLAYER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]
    referees_detections = detections[detections.class_id == REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]

    # Fusionner toutes les détections
    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

    # Projeter les positions sur le terrain
    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    # Projection des positions de la balle
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    # Projection des positions des joueurs
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    # Dessiner la vue radar (petite vue du terrain)
    annotated_frame = draw_pitch(CONFIG)

    # Utiliser draw_pitch_hull pour dessiner les enveloppes des joueurs
    team_xy=pitch_players_xy[players_detections.class_id == team_id]
    
    annotated_frame = None 

    annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 0],
                face_color=color_2,
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=1,
                pitch=annotated_frame)
            
    annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 1],
                face_color=color_1,
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=1,
                pitch=annotated_frame)
    
    annotated_frame = draw_paths_and_hull_on_pitch(
        config=CONFIG,                    # Configuration du terrain
        paths=[team_xy],         # Liste des chemins ou positions projetées des joueurs
        hull_color=(0, 0, 255),
        path_color=sv.Color.WHITE,        # Couleur des chemins, ici blanc
        thickness=2,                      # Épaisseur des lignes des chemins
        padding=50,                       # Espace autour du terrain
        scale=0.1,                        # Échelle du terrain
        alpha=0.3,                        # Transparence de la zone convexe
        pitch=annotated_frame                      # Générer un nouveau terrain (ou passer un terrain existant)
    )

    # Dessiner les points sur le terrain
    annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.from_hex('FFD700'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    thickness=2,
                    pitch=annotated_frame)


    return annotated_frame





