import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Football Heatmap", layout="wide")

def draw_pitch(ax, flip_x=False):
    """Dessine un terrain de football avec fond vert et lignes blanches, possibilité d'inverser les X."""
    ax.set_facecolor("green")  # Fond vert

    def transform_x(x):
        return 12000 - x if flip_x else x

    # Contour du terrain
    ax.plot([transform_x(0), transform_x(12000), transform_x(12000), transform_x(0), transform_x(0)], 
            [0, 0, 7000, 7000, 0], color="white", linewidth=2)

    # Ligne médiane
    ax.plot([transform_x(6000), transform_x(6000)], [0, 7000], color="white", linewidth=2)

    # Cercle central
    ax.add_patch(plt.Circle((transform_x(6000), 3500), 900, color="white", fill=False, linewidth=2))
    
    # Point central
    ax.add_patch(plt.Circle((transform_x(6000), 3500), 100, color="white", fill=True))

    # Surfaces de réparation
    ax.plot([transform_x(0), transform_x(1800), transform_x(1800), transform_x(0)], [2000, 2000, 5000, 5000], color="white", linewidth=2)
    ax.plot([transform_x(12000), transform_x(10200), transform_x(10200), transform_x(12000)], [2000, 2000, 5000, 5000], color="white", linewidth=2)

    # Buts
    ax.plot([transform_x(0), transform_x(-200), transform_x(-200), transform_x(0)], [3000, 3000, 4000, 4000], color="white", linewidth=2)
    ax.plot([transform_x(12000), transform_x(12200), transform_x(12200), transform_x(12000)], [3000, 3000, 4000, 4000], color="white", linewidth=2)

    ax.set_xlim(0, 12000)
    ax.set_ylim(0, 7000)
    
    # Affichage des axes
    ax.set_xlabel("X (longueur du terrain)")
    ax.set_ylabel("Y (largeur du terrain)")
    
    # Suppression des ticks pour le style
    ax.set_xticks(range(0, 13000, 2000))  
    ax.set_yticks(range(0, 8000, 1000))  
    ax.grid(True, linestyle="--", linewidth=0.5, color="white")  # Ajout d'une grille blanche en pointillés


def plot_heatmap_and_tracking(csv_path, flip_x=False):
    """Charge les données, trace la heatmap et le tracking du barycentre."""
    df = pd.read_csv(csv_path)
    df['Barycenter'] = df['Barycenter'].apply(lambda x: eval(x) if isinstance(x, str) else x)  # Conversion des listes
    
    # Extraction des coordonnées du barycentre
    if flip_x:
        bary_x = [12000 - b[0] for b in df['Barycenter']]
    else:
        bary_x = [b[0] for b in df['Barycenter']]
    
    bary_y = [b[1] for b in df['Barycenter']]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_pitch(ax, flip_x)
    
    # Tracé de la heatmap
    sns.kdeplot(x=bary_x, y=bary_y, ax=ax, cmap="Reds", fill=True, alpha=0.6, levels=50)
    
    # Tracé du tracking du barycentre
    ax.plot(bary_x, bary_y, marker="o", markersize=4, color="blue", alpha=0.6)
    
    return fig

