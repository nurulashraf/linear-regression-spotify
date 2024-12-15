import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/datasets/spotify_tracks.csv')

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Display data info
print("\nDataset Info:")
print(df.info())
