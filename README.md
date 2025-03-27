# 📹 Football Video Analysis Application

This Streamlit-based application provides two main functionalities for football video analysis:

1. **⚽ Action Analysis:** Upload football videos to generate timelines, visualizations, and analytical clips.
2. **📊 Tactical Visualization:** Upload videos to generate advanced tactical visualizations.

## 🚀 Features
- **Action Analysis:**
  - Generates interactive timelines and video clips.
  
- **Tactical Visualization:**
  - Generate detailed tactical insights from match footage.
  - Visualize player movements and team formations.

## 🛠️ Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/RBFA_AI_Project.git
   cd RBFA_AI_Project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## 📂 Project Structure
```
📁 your-repo/
│-- 📜 main.py             # Main Streamlit application
│-- 📜 interface1.py      # Action analysis module
│-- 📜 interface2.py      # Tactical visualization module
│-- 📜 requirements.txt   # Python dependencies
```

## 📝 Usage
- Open the web interface and navigate between the two tabs:
  - **Action Analysis:** Upload a football video to extract key moments.
  - **Tactical Visualization:** Upload a video for tactical insights.

## 🏆 Results & Examples

### 📸 Image Visualizations

#### 🔹 Tactical Views & Team Connections  
Visualization of player positions and their connections to analyze team structure.

<p align="center">
  <img src="github_visuals/area_pitch.jpg" alt="Tactical Views" width="50%">
  <img src="github_visuals/back_lines.jpg" alt="Team Connections" width="50%">
</p>

#### 🏟️ Convex Areas & Team Pitch Occupation  
Analysis of pitch control and space occupation by each team.

<p align="center">
  <img src="github_visuals/back_area_frame.jpg" alt="Convex Areas" width="50%">
  <img src="github_visuals/voronoi_back.jpg" alt="Team Pitch Occupation" width="50%">
</p>

#### ⏳ Timeline Generation  
Automatic timeline generation of key match events.

<p align="center">
  <img src="github_visuals/kickoff.png" alt="Timeline Generation" width="100%">
</p>

---

### 🎥 Generated Videos  

#### 🔹 Tactical Insights Compilation  
A video showcasing multiple analytical visualizations:

- **Team Occupation (Areas)**  
- **Quantitative Team Occupation**  
- **Defense, Middle, or Attack Areas or Lines**  

<p align="center">
  <img src="github_visuals/video_resluts.mp4" alt="Generated Video" width="70%">
</p>

</p>

