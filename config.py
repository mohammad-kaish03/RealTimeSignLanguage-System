import os

# Base directory (project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# File paths
DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")

# Create folders if not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)