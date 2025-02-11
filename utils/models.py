import os
import torch
from ultralytics import RTDETR
import inference
from inference import get_model
import requests

# Fonction pour télécharger un fichier depuis Google Drive
def download_file_from_drive(direct_link, output_path):
    """
    Télécharge un fichier depuis Google Drive en utilisant un lien direct.
    :param direct_link: Lien direct de téléchargement
    :param output_path: Chemin de sortie local pour enregistrer le fichier
    """
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

# Fonction pour charger le modèle de détection de joueurs RBFA
def load_player_detection_model(model_path="models/player_detect.pt", direct_link=None):
    """
    Charge le modèle de détection de joueurs (RBFA Detection Model).
    Si le modèle n'existe pas localement, il est téléchargé depuis Google Drive.
    :param model_path: Chemin local du fichier .pt
    :param direct_link: Lien direct de téléchargement (optionnel)
    :return: Modèle chargé
    """
    # Télécharger le modèle depuis Google Drive si nécessaire
    if direct_link and not os.path.exists(model_path):
        print(f"Téléchargement du modèle depuis Google Drive...")
        download_file_from_drive(direct_link, model_path)

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

# Exemple d'utilisation des fonctions
if __name__ == "__main__":
    # Lien direct de téléchargement pour player_detect.pt
    direct_link = "https://drive.google.com/uc?export=download&id=1FuibHhLGI7PvaZxSPrxhtdQxveyqdKTg"

    # Charger les modèles
    player_detection_model = load_player_detection_model(direct_link=direct_link)
    field_detection_model = load_field_detection_model()

    # Vous pouvez maintenant utiliser ces modèles pour l'inférence
    # Par exemple :
    # results = player_detection_model.predict(image)
    # results = field_detection_model.predict(image)
