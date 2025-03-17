import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def calculate_position_percentages(data):
    
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
    
    if len(kickoffs)>1:
        adjusted_times = []
        for t in times:
            if t >= kickoffs[1]:
                adjusted_times.append((t - kickoffs[1]) + 45)
            else:
                adjusted_times.append(t - kickoffs[0])
        times = adjusted_times
    
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
        y_val = mapping_y.get(pos_val, 0.5)
        print(y_val)
        
        if len(group) in [2, 3]:
            group_times = [times[k] for k in group]
            group_middle = np.mean(group_times)
            y_line = y_val - 0.05

            ax.plot([group_times[0], group_times[-1]], [y_line, y_line],
                    color=colors.get(pos_val, "black"), lw=2, zorder=5)
            ax.scatter(group_middle, y_val, color=colors.get(pos_val, "black"),
                       marker='v', s=150, zorder=6)
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
        ax.plot([0, 0], [0.92, 1], color='black', linewidth=2) 
        ax.scatter(0, 1, color='red', s=200, marker='o', edgecolors='black', zorder=10)
        ax.text(0, 0.88, "Kick-off", color='black', fontsize=8, ha='center')

        if len(kickoffs) > 1:
            ax.plot([kickoffs[1], kickoffs[1]], [0.92, 1], color='black', linewidth=2)
            ax.scatter(kickoffs[1], 1, color='red', s=200, marker='o', edgecolors='black', zorder=10)
            ax.text(kickoffs[1], 0.89, "Kick-off", color='black', fontsize=12, ha='center', fontweight='bold')


    ax.set_xlabel("Time (minutes from kickoff)", fontsize=12)
    ax.set_title("Temporal Analysis of Possession", fontsize=14, pad=20)
    ax.set_xlim(-1, max(times) + 2 if times else 0)
    ax.set_yticks([])
    ax.grid(True, linestyle='--', alpha=0.4)
    
    return fig

def draw_pitch(ax, flip_x=False):
    """Dessine un terrain de football avec fond vert et lignes blanches, possibilité d'inverser les X."""
    ax.set_facecolor("green")

    def transform_x(x):
        return 12000 - x if flip_x else x

    ax.plot([transform_x(0), transform_x(12000), transform_x(12000), transform_x(0), transform_x(0)], 
            [0, 0, 7000, 7000, 0], color="white", linewidth=2)

    ax.plot([transform_x(6000), transform_x(6000)], [0, 7000], color="white", linewidth=2)

    ax.add_patch(plt.Circle((transform_x(6000), 3500), 900, color="white", fill=False, linewidth=2))

    ax.add_patch(plt.Circle((transform_x(6000), 3500), 100, color="white", fill=True))

    ax.plot([transform_x(0), transform_x(1800), transform_x(1800), transform_x(0)], [2000, 2000, 5000, 5000], color="white", linewidth=2)
    ax.plot([transform_x(12000), transform_x(10200), transform_x(10200), transform_x(12000)], [2000, 2000, 5000, 5000], color="white", linewidth=2)

    ax.plot([transform_x(0), transform_x(-200), transform_x(-200), transform_x(0)], [3000, 3000, 4000, 4000], color="white", linewidth=2)
    ax.plot([transform_x(12000), transform_x(12200), transform_x(12200), transform_x(12000)], [3000, 3000, 4000, 4000], color="white", linewidth=2)

    ax.set_xlim(0, 12000)
    ax.set_ylim(0, 7000)

    ax.set_xticks(range(0, 13000, 2000))  
    ax.set_yticks(range(0, 8000, 1000)) 

    ax.grid(True, linestyle="--", linewidth=0.5, color="white")


def plot_heatmap_and_tracking(csv_path, flip_x=False,add_trajectory = False):
    """Charge les données, trace la heatmap et le tracking du barycentre."""
    df = pd.read_csv(csv_path)
    df['Barycenter'] = df['Barycenter'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    if flip_x:
        bary_x = [12000 - b[0] for b in df['Barycenter']]
    else:
        bary_x = [b[0] for b in df['Barycenter']]
    
    bary_y = [b[1] for b in df['Barycenter']]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_pitch(ax, flip_x)

    sns.kdeplot(x=bary_x, y=bary_y, ax=ax, cmap="Reds", fill=True, alpha=0.6, levels=50)

    if add_trajectory:
       ax.plot(bary_x, bary_y, marker="o", markersize=4, color="blue", alpha=0.6)
    
    return fig

def plot_ball_heatmap_and_tracking(csv_path,add_trajectory):
    """Loads data, plots heatmap and tracking for ball position."""
    df = pd.read_csv(csv_path)

    ball_x = df['posX']
    
    ball_y = df['posY']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_pitch(ax)

    sns.kdeplot(x=ball_x, y=ball_y, ax=ax, cmap="Reds", fill=True, alpha=0.6, levels=50)
    
    if add_trajectory:
       ax.plot(ball_x, ball_y, marker="o", markersize=4, color="blue", alpha=0.6)
    
    return fig

