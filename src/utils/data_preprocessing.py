import pandas as pd

def load_data(filepath):
    data = pd.read_csv(/content/drive/MyDrive/datasets/spotify_tracks.csv)
    return data

def preprocess_data(data):
    # Example: drop null values or encode categorical variables
    data = data.dropna()
    return data
