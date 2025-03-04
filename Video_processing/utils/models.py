import os
import torch
from ultralytics import RTDETR
import inference
from inference import get_model
import requests
import os
import requests


# Fonction pour télécharger un fichier depuis Google Drive
def download_file_from_drive(direct_link, output_path):
    """
    Télécharge un fichier depuis Google Drive en utilisant un lien direct.
    :param direct_link: Lien direct de téléchargement
    :param output_path: Chemin de sortie local pour enregistrer le fichier
    """
    # Créer le dossier 'models' si il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    session = requests.Session()
    response = session.get(direct_link, stream=True)
    
    # Vérifier si le téléchargement a réussi
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Fichier téléchargé et enregistré sous {output_path}")
    else:
        raise Exception(f"Échec du téléchargement. Code d'état : {response.status_code}")

# Fonction pour charger le modèle RT-DETR
def load_player_detection_model(model_path="models/player_detect.pt", direct_link=None, device=None):
    """
    Charge le modèle RT-DETR pour la détection de joueurs.
    Si le modèle n'existe pas localement, il est téléchargé depuis Google Drive.
    
    :param model_path: Chemin local du fichier .pt
    :param direct_link: Lien direct de téléchargement (optionnel)
    :param device: "cuda" ou "cpu" (par défaut, auto-détection)
    :return: Modèle RT-DETR chargé
    """
    # Télécharger le modèle si nécessaire
    if direct_link and not os.path.exists(model_path):
        print(f"Téléchargement du modèle RT-DETR depuis Google Drive...")
        download_file_from_drive(direct_link, model_path)

    # Vérification de l'existence du fichier
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier {model_path} est introuvable.")

    # Charger le modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RTDETR(model_path)
    model.to(device)
    print(f"Player detection model loaded on {device}")
    return model


# Fonction pour charger le modèle de détection de terrain depuis Roboflow
def load_field_detection_model(api_key="cxtZ0KX74eCWIzrKBNkM", model_id="football-field-detection-f07vi/14"):
    """
    Charge le modèle de détection de terrain depuis Roboflow en utilisant l'API key et le model ID.
    """
    field_detection_model = get_model(model_id=model_id, api_key=api_key)
    print("Field detection model loaded from Roboflow")
    return field_detection_model
