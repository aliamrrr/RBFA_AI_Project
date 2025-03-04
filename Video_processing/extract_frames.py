import os
import cv2
import streamlit as st
import supervision as sv
import numpy as np
from utils.view import ViewTransformer
from utils.team_classifier import extract_player_crops, fit_team_classifier
from utils.models import load_player_detection_model, load_field_detection_model
from utils.soccer import SoccerPitchConfiguration
import json
from datetime import timedelta
import matplotlib.pyplot as plt 
from PIL import Image
import csv
from utils.frame_process import is_pressing,save_team_line_clusters, is_kickoff
from utils.annotators import plot_ball_heatmap,plot_barycenter_heatmap

import torch
from utils.visualizations import draw_pitch,plot_heatmap_and_tracking


# Configuration des constantes
PLAYER_ID = 0
BALL_ID = 2
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
fig = None

# Initialisation de l'état de session
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
 
init_session_state()

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def create_exponential_peaks(times, amplitudes, width=0.1, num_points=1000):
    """
    Create a signal with sharp exponential peaks at the specified times.
    
    Args:
        times (list): List of times to place the peaks.
        amplitudes (list): List of amplitudes for each peak.
        width (float): Width of the peaks (controls the "sharpness").
        num_points (int): Number of points for the time axis.
    
    Returns:
        tuple: (time_axis, signal)
    """
    t_min = min(times) - width
    t_max = max(times) + width
    time_axis = np.linspace(t_min, t_max, num_points)
    signal = np.zeros_like(time_axis)
    for t, amplitude in zip(times, amplitudes):
        signal += amplitude * np.exp(- (np.abs(time_axis - t) / width)**2)

    return time_axis, signal

def plot_possession_graph(data, kickoffs=[], width=0.1, threshold=1):
    print(kickoffs)
    if not data:
        st.warning("No data available for plotting.")
        return None
    
    colors = {"back": "red", "middle": "blue", "attack": "green"}
    mapping_y = {"back": 0.3, "middle": 0.5, "attack": 0.75}
    
    times = [entry["time_in_minutes"] for entry in data]
    positions = [entry["pos"] for entry in data]
    values = [mapping_y.get(pos, 1.0) for pos in positions]

    amplitudes = []
    for i in range(len(values)):
      if positions[i] == 'attack':
        amplitudes.append(values[i])
      elif positions[i] == 'middle':
        amplitudes.append(values[i])
      else:
        amplitudes.append(values[i])
    print(amplitudes)

    
    if kickoffs:
        ref_time = kickoffs[0]
        times = [t - ref_time for t in times]
        kickoffs = [t - ref_time for t in kickoffs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if times:
        t, y = create_exponential_peaks(times, amplitudes, width=width)
        ax.fill_between(t, y, color='skyblue', alpha=0.2, label='Possession Intensity', zorder=0)
        ax.plot(t, y, color='navy', alpha=0.5, linewidth=1, zorder=1)
    
    ax.plot(times, [0.95] * len(times), color="orange", linewidth=2, zorder=2, label='Possession Sequence')
    
    i, n = 0, len(times)
    while i < n:
        group = [i]
        j = i + 1
        while j < n and positions[j] == positions[i] and (times[j] - times[j - 1]) <= threshold:
            group.append(j)
            j += 1
        
        pos_val = positions[i]
        y_val = mapping_y.get(pos_val, 0.5)  # Amplitude correcte
        print(y_val)
        
        if len(group) in [2, 3]:
            group_times = [times[k] for k in group]
            group_middle = np.mean(group_times)
            y_line = y_val - 0.05  # Ajustement ligne

            ax.plot([group_times[0], group_times[-1]], [y_line, y_line],
                    color=colors.get(pos_val, "black"), lw=2, zorder=5)
            ax.scatter(group_middle, y_val, color=colors.get(pos_val, "black"),
                       marker='v', s=150, zorder=6)  # 👈 Ajusté ici
            ax.annotate(pos_val, (group_middle, y_val), xytext=(0, 10),
                        textcoords="offset points", ha='center',
                        color=colors.get(pos_val, "black"), fontsize=10, fontweight='bold')
        else:
            ax.scatter(times[i], y_val, color=colors.get(pos_val, "black"),
                       marker='v', s=150, zorder=4)  # 👈 Ajusté ici
            ax.annotate(pos_val, (times[i], y_val), xytext=(0, 10),
                        textcoords="offset points", ha='center',
                        color=colors.get(pos_val, "black"), fontsize=10, fontweight='bold')
        i = j
    
    if kickoffs:
        # Premier Kickoff
        ax.plot([0, 0], [0.92, 1], color='black', linewidth=2)  # Mât du drapeau (plus court)
        ax.scatter(0, 1, color='red', s=200, marker='o', edgecolors='black', zorder=10)  # Drapeau rond
        ax.text(0, 0.88, "Kick-off", color='black', fontsize=8, ha='center')  # Texte en dessous

        if len(kickoffs) > 1:
            # Deuxième Kickoff
            ax.plot([kickoffs[1], kickoffs[1]], [0.92, 1], color='black', linewidth=2)  # Mât plus court
            ax.scatter(kickoffs[1], 1, color='red', s=200, marker='o', edgecolors='black', zorder=10)  # Drapeau rond
            ax.text(kickoffs[1], 0.89, "Kick-off", color='black', fontsize=12, ha='center', fontweight='bold')  # Texte en dessous


    ax.set_xlabel("Time (minutes from kickoff)", fontsize=12)
    ax.set_title("Temporal Analysis of Possession", fontsize=14, pad=20)
    ax.set_xlim(-1, max(times) + 2 if times else 0)
    ax.set_yticks([])
    ax.grid(True, linestyle='--', alpha=0.4)
    
    return fig

def calculate_position_percentages(data):
    """
    Calcule le pourcentage des positions 'attack', 'middle' et 'back' dans les données.
    
    Args:
        data (list): Liste de dictionnaires contenant les informations.
    
    Returns:
        dict: Dictionnaire avec les pourcentages de chaque position.
    """
    total = len(data)
    if total == 0:
        return {"attack": 0.0, "middle": 0.0, "back": 0.0}

    position_counts = {"attack": 0, "middle": 0, "back": 0}
    
    for entry in data:
        pos = entry["pos"]
        if pos in position_counts:
            position_counts[pos] += 1
    
    percentages = {pos: (count / total) * 100 for pos, count in position_counts.items()}
    
    return percentages

# Interface utilisateur
st.title("⚽ Football Video Analysis System")

# Widgets de contrôle
video_path = st.text_input("Video Path", r"C:\Users\Administrateur\Desktop\RBFA\last_ukr.mp4")


col1, col2 = st.columns(2)
col4, col5 = st.columns(2)


with col1:
    start_btn = st.button("🚀 Start Processing")
with col2:
    stop_btn = st.button("⛔ Stop Processing")



# Gestion des événements
if start_btn:
    st.session_state.processing = True
    st.session_state.stop = False

if stop_btn:
    st.session_state.stop = True



# Détermination des noms de dossiers
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
            image_bgr = cv2.imread(img_path) # Fix color issue
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

        # Détection des joueurs et balle
        results = player_detection_model.predict(frame, conf=CONFIDENCE_THRESHOLD)
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy()
        )

        detections = detections.with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        players = detections[detections.class_id == PLAYER_ID]
        ball = detections[detections.class_id == BALL_ID]
        

        # Classification des équipes
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

            # Enregistrement des données dans un fichier CSV au fur et à mesure
            with open('ball_tracking_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # Écrire l'en-tête seulement si le fichier est vide
                if file.tell() == 0:
                    writer.writerow(["posX", "posY", "team_possession","frame"])
                writer.writerow(data_row)

            clusters = save_team_line_clusters(players_pos, team_id = st.session_state.team_id , output_dir="clusters/")
 
            # Sauvegarde des frames
            if team_in_possession:

                zone_width = config.length // 3
                zone = get_zone(ball_pos, zone_width, first_half,attack_direction)

                pressure = is_pressing(players, players_pos, ball_pos, team_id = st.session_state.team_id, zone=zone, team_possession = team_in_possession)

                # Enregistrement de l'événement
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
                # Écrire l'en-tête seulement si le fichier est vide
                if file.tell() == 0:
                    writer.writerow(["posX", "posY", "team_possession","frame"])
                writer.writerow(data_row)
        
        # Mise à jour de la progression
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processed frames: {frame_count}/{total_frames}")

        # Saut de frames
        for _ in range(skip - 1):
            if not cap.grab() or st.session_state.stop:
                break
          
    cap.release()


    st.session_state.processing = False
    return st.session_state.possession_events


# Initializing values in session_state if not already initialized
if 'team_id' not in st.session_state:
    st.session_state.team_id = None
if 'attack_direction' not in st.session_state:
    st.session_state.attack_direction = None
if "position" not in st.session_state:
    st.session_state.position = "defense"

tab1, tab2, tab3 = st.tabs(["🎥 Video processing","📊 Timeline visualisations", "📄 Tracking data"])

with tab1:
        # Exécution principale
        if st.session_state.processing and not st.session_state.stop:

            # Check if the classifier is already loaded, to avoid re-running it every time
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

        uploaded_file = st.file_uploader("Upload JSON Data", type=["json"], help="Drag and drop your JSON file here.")

        if uploaded_file is not None:
          try:
            data = json.load(uploaded_file)

            percentages = calculate_position_percentages(data)
        
            st.markdown("### Répartition des Positions (%)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="🔴 Défense (Back)", value=f"{int(percentages['back'])}%")
            
            with col2:
                st.metric(label="🔵 Milieu (Middle)", value=f"{int(percentages['middle'])}%")

            with col3:
                st.metric(label="🟢 Attaque (Attack)", value=f"{int(percentages['attack'])}%")
            
            if st.button("📊 Generate Graph"):
                print("kickofffffff")
                print(st.session_state.kickoffs)
                fig = plot_possession_graph(data,st.session_state.kickoffs)
                st.pyplot(fig)
          except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid file.")

with tab3:
   st.session_state.position = st.selectbox("Select a position :" ,["defense","middle","attack"])
   if st.button("Get position heatmap"):
    csv_file = f"team_1_{st.session_state.position}.csv"
    flip_x = st.session_state.attack_direction == "left"
    fig = plot_heatmap_and_tracking(csv_file,flip_x=True)
    st.pyplot(fig)


# Affichage des paramètres
st.subheader("Current Settings")
if st.session_state.team_id is not None:
    st.write(f"Team ID: {st.session_state.team_id}")
st.write(f"Team Orientation: {'Switched' if st.session_state.teams_switched else 'Normal'}")
st.write(f"Processing Status: {'Running' if st.session_state.processing else 'Stopped'}")

