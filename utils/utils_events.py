import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Load the pretrained ResNet model and adapt it
def load_model():
    num_classes = 2  # Number of classes: goal kicks, penalty kicks, background
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved model
    model.load_state_dict(torch.load("events.pth", map_location=torch.device('cpu')))
    model.eval()

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device


# Function to predict probabilities on a given image
def predict_probabilities(model, frame_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(frame_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Calculate probabilities
    return probs.squeeze(0).cpu().numpy()


# Function to extract frames from a video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Unable to open the video.")
        return

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:  # tqdm progress bar
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            pbar.update(1)  # Update the progress bar

    cap.release()
    st.success(f"{frame_count} frames extracted and saved in {output_folder}")


# Function to analyze probabilities of frames in a video
def analyze_video_probabilities(model, frames_folder, class_names, thresholds, device):
    probabilities = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    with tqdm(total=len(frame_files), desc="Analyzing video", unit="frame") as pbar:  # tqdm progress bar
        for frame_filename in frame_files:
            frame_path = os.path.join(frames_folder, frame_filename)
            probs = predict_probabilities(model, frame_path, device)

            filtered_probs = [
                prob if prob >= thresholds[class_name] else 0
                for prob, class_name in zip(probs, class_names)
            ]

            probabilities.append(filtered_probs)
            pbar.update(1)  # Update the progress bar

    probabilities = np.array(probabilities)
    return probabilities


# Function to plot probabilities with significant changes
def plot_probabilities_with_significant_changes(probabilities, fps=30, prob_threshold=0.3):
    num_frames = len(probabilities)
    time = np.arange(num_frames) / fps

    prob_play = probabilities[:, 1]
    prob_goal_kick = probabilities[:, 0]

    fig, ax = plt.subplots(figsize=(12, 6))  # Explicit creation of figure and axes

    ax.plot(time, prob_play, label='Play Probability', color='blue', linewidth=2)
    ax.plot(time, prob_goal_kick, label='Goal Kick Probability', color='orange', linewidth=2)

    current_class = None
    start_time = 0
    color = None
    zones = []

    for i in range(1, num_frames):
        if prob_goal_kick[i] > prob_play[i]:
            dominant_class = 'goal kick'
            new_color = 'violet'
        else:
            dominant_class = 'play'
            new_color = 'yellow'

        prob_diff = abs(prob_goal_kick[i] - prob_play[i])

        if prob_diff >= prob_threshold:
            if dominant_class != current_class:
                if current_class is not None:
                    zones.append({
                        'class': current_class,
                        'start': start_time,
                        'end': time[i-1],
                        'color': color
                    })

                current_class = dominant_class
                color = new_color
                start_time = time[i-1]

    zones.append({
        'class': current_class,
        'start': start_time,
        'end': time[-1],
        'color': color
    })

    filtered_zones = []
    for i, zone in enumerate(zones):
        if (
            i > 0 and i < len(zones) - 1 and
            zone['class'] == 'play' and
            zone['end'] - zone['start'] < 2 and
            zones[i-1]['class'] == 'goal kick' and
            zones[i+1]['class'] == 'goal kick'
        ):
            filtered_zones[-1]['end'] = zones[i+1]['end']
        else:
            filtered_zones.append(zone)

    merged_zones = []
    for zone in filtered_zones:
        if (
            merged_zones and
            zone['class'] == 'goal kick' and
            merged_zones[-1]['class'] == 'goal kick'
        ):
            merged_zones[-1]['end'] = zone['end']
        else:
            merged_zones.append(zone)

    json_result = json.dumps(merged_zones, indent=4)

    json_file_name = "merged_zones.json"
    with open(json_file_name, "w") as json_file:
        json_file.write(json_result)

    # Display zones on the graph
    for zone in merged_zones:
        mid_time = (zone['start'] + zone['end']) / 2
        ax.axvspan(zone['start'], zone['end'], color=zone['color'], alpha=0.3)
        ax.text(mid_time, 0.5, zone['class'].capitalize(), horizontalalignment='center', verticalalignment='center', color='black', fontsize=12, fontweight='bold')

    ax.set_title("Probabilities of Play and Goal Kick with Time", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    st.pyplot(fig) 

def generate_goalkick_clips(json_file, video_file, output_dir):
    with open(json_file, "r") as file:
        zones = json.load(file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load video with MoviePy
    video = VideoFileClip(video_file)
    fps = video.fps

    with tqdm(total=len(zones), desc="Generating clips", unit="clip") as pbar:  # tqdm progress bar
        for i, zone in enumerate(zones):
            if zone["class"] == "goal kick":
                start_time = zone["start"]
                end_time = zone["end"]

                # Cut the video segment
                clip = video.subclip(start_time, end_time)

                # Save the clip in MP4 format (H264 codec)
                clip_filename = os.path.join(output_dir, f"goalkick_clip_{i+1}.mp4")
                clip.write_videofile(clip_filename, codec="libx264", audio=False)
                
                pbar.update(1)  # Update the progress bar


import streamlit as st
import os

def display_goalkick_clips():
    clips_folder = "goalkick_clips"
    
    # List MP4 files in the goalkick_clips folder
    clips = [f for f in os.listdir(clips_folder) if f.endswith(".mp4")]
    
    # Log: Check if clips are found
    st.write(f"Clips found in the folder {clips_folder}: {clips}")
    
    if clips:
        st.subheader("Select a Goal Kick clip")
        for clip in clips:
            if st.button(f"View {clip}"):
                clip_path = os.path.join(clips_folder, clip)
                
                st.write(f"Attempting to access the file: {clip_path}")
                
                if os.path.exists(clip_path):
                    st.write(f"File found: {clip_path}")
                    st.video(clip_path)
    else:
        st.warning("No Goal Kick clips found in the folder.")
