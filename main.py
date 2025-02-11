import streamlit as st
import tempfile
from utils.models import load_player_detection_model, load_field_detection_model
from utils.team_classifier import show_crops, extract_player_crops, fit_team_classifier
from PIL import Image
from utils.frame_process import process_frame,process_and_save_video, process_frame_with_convex_hull, draw_radar_view,extract_tracking_data,annotate_frame_with_hulls,annotate_player_name,passes_options,calculate_optimal_passes,calculate_realistic_optimal_passes
from streamlit_drawable_canvas import st_canvas
import supervision as sv
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from utils.utils_events import (
    load_model,
    predict_probabilities,
    extract_frames,
    analyze_video_probabilities,
    plot_probabilities_with_significant_changes,
    generate_goalkick_clips,
    display_goalkick_clips,
)

# Liste des joueurs (exemple)
players = [
    "Koen Casteels",
    "Arthur Theate",
    "Wout Faes",
    "Jan Vertonghen",
    "Timothy Castagne",
    "Kevin De Bruyne",
    "Amadou Onana",
    "Romelu Lukaku",
    "Yannick Carrasco",
    "Loïs Openda",
    "Jeremy Doku",
    "Thomas Kaminski",
    "Matz Sels",
    "Zeno Debast",
    "Axel Witsel",
    "Youri Tielemans",
    "Leandro Trossard",
    "Aster Vranckx",
    "Orel Mangala",
    "Arthur Vermeeren",
    "Maxim De Cuyper",
    "Dodi Lukebakio",
    "Charles De Ketelaere",
    "Johan Saint-Cyr Bakayoko"
]

import cv2
from utils.soccer import SoccerPitchConfiguration
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
import json

CONFIG = SoccerPitchConfiguration()
tracker = sv.ByteTrack()
tracker.reset()

# Charger le modèle de détection des goal kicks
def load_goal_kick_model():
    num_classes = 2  # Classes : background, goal kick
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("goalkick.pth", map_location=torch.device('cpu')))
    model.eval()
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# Sidebar Menu
menu = st.sidebar.selectbox("Select an option", 
                            ["Tracking data","Football Data Chatbot", "Event Detection"])

if menu == "Football Data Chatbot":
    # Initialize session state for chat history and data
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "db" not in st.session_state:
        st.session_state.db = None

    # Title of the Streamlit app
    st.title('Streamlit CSV Uploader and Chatbot with Ollama Model')

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Reading the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Show the first few rows of the dataframe
        st.write("CSV File Preview:", df.head())

        # Filter columns based on user input
        columns_to_keep = st.multiselect("Select columns to keep", options=df.columns.tolist(), default=df.columns.tolist())
        
        if st.button('Process CSV'):
            # Filter CSV based on selected columns
            filtered_df = df[columns_to_keep]
            filtered_csv = "filtered_data.csv"
            filtered_df.to_csv(filtered_csv, index=False)

            # Display the filtered CSV content
            st.write("Filtered Data:", filtered_df.head())

            # Load CSV into LangChain's CSVLoader
            loader = CSVLoader(file_path=filtered_csv)
            data = loader.load()

            # Split documents into smaller chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunked_docs = text_splitter.split_documents(data)

            # Use HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

            # Store in FAISS vector database
            st.session_state.db = FAISS.from_documents(chunked_docs, embeddings)
            
            # Create a retriever from FAISS index
            retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={'k': 4})

            # Set up Ollama model
            ollama_api = Ollama(model="gemma2:2b")

            # Set up LangChain ConversationalRetrievalChain with Ollama model
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(ollama_api, retriever, return_source_documents=True)

            st.success("CSV processed and QA chain ready!")

    # Interaction section
    if st.session_state.qa_chain is not None:
        st.subheader("Ask questions related to the data:")

        # Get user input for queries
        query = st.text_input("Ask a question")

        if query:
            try:
                # Retrieve the answer from the ConversationalRetrievalChain
                result = st.session_state.qa_chain.invoke({'question': query, 'chat_history': []})

                # Show the answer
                st.write('Answer: ' + result['answer'])

                # Update chat history
                st.session_state.chat_history.append((query, result['answer']))

                # Display chat history
                st.subheader("Chat History:")
                for q, a in st.session_state.chat_history:
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")
                    st.write("---")
            except Exception as e:
                st.error(f"Error while retrieving the answer: {e}")
    else:
        st.write("Please upload a CSV file and process it to get started.")

# 1. **Tracking Data Tab**
if menu == "Tracking data":
    tracking_option = st.selectbox("Choose a feature", 
                                  ["View the video", "Collect, classify and view teams", 
                                   "View an annotated frame","Real-time processing","Tactical visualizations","Get video visualizations"])

    # 1.1 **View the video**
    if tracking_option == "View the video":
        uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])

        if uploaded_file is not None:
            # Create a temporary file for the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Read and display the video
            st.video(tmp_file_path)


        # 1.2 **Collect, classify and view teams**
    elif tracking_option == "Collect, classify and view teams":
            uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])

            if uploaded_file is not None:
                # Create a temporary file for the uploaded video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load player and field detection models
                player_detection_model = load_player_detection_model()  # Load the player model
                field_detection_model = load_field_detection_model()    # Load the field model

                # Display a button to start player detection
                if st.button("Detect players"):
                    # Show a loading indicator during detection
                    with st.spinner("Detection in progress..."):
                        # Extract player crops
                        crops = extract_player_crops(tmp_file_path, player_detection_model)

                        # Check if crops were detected
                        if crops:
                            # Ensure the 'crops' directory exists
                            crops_directory = "crops"
                            os.makedirs(crops_directory, exist_ok=True)

                            # Save each crop as an image in the 'crops' directory
                            for i, crop in enumerate(crops):
                                crop_path = os.path.join(crops_directory, f"crop_{i+1}.jpg")  # Save as .jpg or .png based on your preference

                                # Convert the numpy array (crop) to a PIL image
                                pil_crop = Image.fromarray(crop)
                                
                                # Save the image
                                pil_crop.save(crop_path)

                            # Display the crops in the interface
                            show_crops(crops)
                            st.write(f"Number of players detected: {len(crops)}")  # Display the number of detected crops

                            with st.spinner("Classification in progress..."):
                                try:
                                    # Train the classifier
                                    team_classifier = fit_team_classifier(crops, device="cpu")
                                    st.session_state.team_classifier = team_classifier  # Save to session_state
                                    st.success("Team classification model is ready!")
                                    st.write("You can now view the teams.")
                                except Exception as e:
                                    st.error(f"Error during classification: {str(e)}")
                        else:
                            st.warning("No players detected. Please try with another video.")

    elif tracking_option == "View an annotated frame":
        uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            if "team_classifier" not in st.session_state:
                st.warning("Please first classify the players in the previous tab.")
            else:
                team_classifier = st.session_state.team_classifier

                # Sélecteur des joueurs à annoter
                selected_players = st.multiselect("Select players to annotate", players)

                frame_index = st.slider("Select the frame index", 0, 100, 0)

                cap = cv2.VideoCapture(tmp_file_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if ret:
                    st.image(frame, channels="BGR", caption=f"Frame {frame_index}")
                    annotated_frame = process_frame(frame, team_classifier)

                    # Convertir le frame annoté en image PIL
                    annotated_frame = Image.fromarray(annotated_frame)

                    # Canevas pour capturer les clics
                    st.write("Click on a detection to annotate the player name:")

                    if isinstance(annotated_frame, (str, Image.Image, type(None))):
                        canvas_result = st_canvas(
                            stroke_width=1,
                            background_image=annotated_frame,
                            update_streamlit=True,
                            height=400,
                            width=1000,
                            drawing_mode="point",
                            key="canvas",
                        )

                    # Vérifier si un clic a été fait
                    if canvas_result.json_data is not None:
                        click_coords_list = []
                        for point in canvas_result.json_data["objects"]:
                            click_coords = (point["left"], point["top"])
                            click_coords_list.append(click_coords)

                        # Annoter les joueurs sélectionnés dans l'ordre
                        for i, click_coords in enumerate(click_coords_list):
                            if i < len(selected_players):  # Assurez-vous qu'il n'y ait pas d'index hors limites
                                player_name = selected_players[i]
                                annotated_frame = np.array(annotated_frame)
                                annotated_frame = annotate_player_name(annotated_frame, click_coords, player_name)

                    # Afficher l'image annotée
                    st.image(annotated_frame, channels="BGR", caption=f"Annotated frame {frame_index}")

                cap.release()


    elif tracking_option == "Get video visualizations":
        uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])
        if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                cap = cv2.VideoCapture(tmp_file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                total_duration_seconds = total_frames / fps
                
                save_video = st.checkbox("Save Video")
                if save_video:
                    start_time = st.slider("Select start time", 0, int(total_duration_seconds), 0)
                    end_time = st.slider("Select end time", start_time, int(total_duration_seconds), start_time + 1)
                    
                    if st.button("Generate and Save Video"):
                        if "team_classifier" not in st.session_state:
                            st.warning("Please first classify the players in the previous tab.")
                        else:
                            output_video_path = process_and_save_video(tmp_file_path, start_time, end_time, fps, st.session_state.team_classifier)
                            st.video(output_video_path)
                            st.success("Video saved successfully! Download below:")
                            st.download_button("Download Video", open(output_video_path, "rb"), file_name="annotated_video.mp4")


    
    # 1.3 **View an annotated frame**
    elif tracking_option == "Tactical visualizations":
        type = 'tactical'
        players_tracking,ball_tracking = None,None
        uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])
    
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Vérifier si le modèle de classification des équipes est disponible
            if "team_classifier" not in st.session_state:
                st.warning("Please first classify the players in the previous tab.")
            else:
                team_classifier = st.session_state.team_classifier  # Charger le modèle de classification des équipes

                # Charger la vidéo pour obtenir le nombre total de frames
                cap = cv2.VideoCapture(tmp_file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Nombre total de frames
                fps = cap.get(cv2.CAP_PROP_FPS)  # Frames par seconde
                cap.release()

                # Calculer la durée totale en secondes
                total_duration_seconds = total_frames / fps

                # Créer un slider pour sélectionner le temps en secondes
                time_seconds = st.slider("Select the time", 0, int(total_duration_seconds), 0)

                # Convertir le temps en minutes et secondes
                minutes = time_seconds // 60
                seconds = time_seconds % 60

                # Afficher le temps en minutes et secondes
                st.write(f"Time: {minutes} minutes {seconds} seconds")

                # Calculer l'index de la frame correspondant au temps sélectionné
                frame_index = int(time_seconds * fps)

                annotated_frame = None

                # Lire la vidéo et sélectionner la frame correspondante
                cap = cv2.VideoCapture(tmp_file_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if ret:
                    # Afficher la frame originale
                    st.image(frame, channels="BGR", caption=f"Frame {frame_index}")

                # Initialisation de l'état du mode (si nécessaire)
                if 'passes_mode' not in st.session_state:
                    st.session_state.passes_mode = None  # Initialisation de passes_mode

                # Créer deux colonnes pour les boutons
                col1, col2, col3,col4, col5 = st.columns(5)

                # Placer les boutons dans chaque colonne
                with col1:
                    if st.button("Generate tactical view"):
                        annotated_frame = draw_radar_view(frame, CONFIG, team_classifier, 'tactical')

                with col2:
                    if st.button("Generate Voronoi diagram"):
                        type = 'voronoi'
                        annotated_frame = draw_radar_view(frame, CONFIG, team_classifier, type)

                with col4:
                    # Afficher le bouton "Generate pass options"
                    if st.button("Pass options : Play short"):
                       passes_mode = 'build'
                       mode = 1
                       print(CONFIG)
                       annotated_frame = calculate_realistic_optimal_passes(frame, CONFIG, team_classifier, max_passes=5)
                mode = 1
                if annotated_frame is not None and mode == 1:
                    st.image(annotated_frame, channels="BGR", caption=f"Annotated frame {frame_index}")

                with col5:
                    # Afficher le bouton "Generate pass options"
                    if st.button("Pass option : Safe and high"):
                       passes_mode = 'interception'
                       mode = 2
                       annotated_frame = passes_options(frame, CONFIG, team_classifier, passes_mode)
                
                if annotated_frame is not None and mode == 2:
                    st.image(annotated_frame, channels="BGR", caption=f"Annotated frame {frame_index}")

                with col3:
                  if st.button("Preview Tracking dataframes"):

                            players_tracking,ball_tracking = extract_tracking_data(frame, CONFIG, team_classifier,tracker)
                        
                            players_tracking['player_x'] /= 100
                            players_tracking['player_y'] /= 100

                            ball_tracking['ball_x'] /= 100
                            ball_tracking['ball_y'] /= 100
                
                if players_tracking is not None and ball_tracking is not None:

                 column1, column2 = st.columns(2)
                            
                 # Afficher le DataFrame des joueurs dans la première colonne
                 with column1:
                    st.subheader("Players Tracking Data")
                    st.dataframe(players_tracking)

                            # Afficher le DataFrame de la balle dans la deuxième colonne
                 with column2:
                    st.subheader("Ball Tracking Data")
                    st.dataframe(ball_tracking)
                    
                cap.release()

        # 3. **Real-time processing**
    elif tracking_option == "Real-time processing":
      
      uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])

      team_id = st.selectbox("Select team (0 or 1) to view convex hull", [0, 1])

      include_hull = st.checkbox("Include Hull Convex Area Annotation", value=True)

      if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the team classifier model
        if "team_classifier" not in st.session_state:
            st.warning("Please first classify the players in the 'Collect, classify and view teams' tab.")
        else:
            team_classifier = st.session_state.team_classifier

            # Button to start real-time detection
            if st.button("Start Detection"):
                cap = cv2.VideoCapture(tmp_file_path)
                frame_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if include_hull : 
                        annotated_frame = process_frame_with_convex_hull(frame, team_classifier, team_id)
                    # Process the frame and get the annotated one
                    else :
                        annotated_frame = process_frame(frame, team_classifier)

                    # Combine the frames side by side
                    combined_frame = cv2.hconcat([frame, annotated_frame])

                    # Update the same image in Streamlit
                    frame_placeholder.image(combined_frame, channels="BGR")

                cap.release()


# **Goal Kick Detection Tab**
elif menu == "Event Detection":

    event_option = st.selectbox("Choose an event", 
                                  ["Goalkicks", "Freekicks"])
    st.subheader("Football Video Event Detection")
    if event_option == "Goalkicks":
            uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
            if uploaded_video is not None:
                video_path = os.path.join("uploaded_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                # Display the uploaded video
                st.video(uploaded_video)

                model, device = load_model()
                frames_folder = "extracted_frames"

                if st.button("Extract frames from video"):
                    with st.spinner("Extracting frames..."):
                        extract_frames(video_path, frames_folder)

                class_names = ['background', 'freekicks']
                thresholds = {'background': 0.2, 'freekicks': 0}
                if st.button("Analyze video probabilities"):
                    with st.spinner("Analyzing video..."):
                        probabilities = analyze_video_probabilities(model, frames_folder, class_names, thresholds, device)
                        st.subheader("Probability Graph")
                        plot_probabilities_with_significant_changes(probabilities)

                if st.button("Generate Goal Kick clips"):
                    json_file = "merged_zones.json"
                    if os.path.exists(json_file):
                        with st.spinner("Generating Goal Kick clips..."):
                            generate_goalkick_clips(json_file, video_path, "goalkick_clips")
                        st.success("Goal Kick clips generated successfully!")
                    else:
                        st.error(f"The file {json_file} was not found. Please analyze probabilities first.")

                display_goalkick_clips()

