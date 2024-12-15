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

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Select relevant features for the analysis
music2 = df[['speechiness', 'energy', 'danceability', 'valence']]
print("\nSelected Features Preview:")
print(music2.head())

# Define features (X) and target variable (y)
X = music2.drop("valence", axis=1).values
y = music2["valence"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Test the model with a specific example
print("\nExample Prediction:")
example = np.array([[0.103, 0.97, 0.753]])
predicted_valence = model.predict(example)[0]
print(f"Predicted Valence for input {example}: {predicted_valence}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot the Actual vs Predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Valence")
plt.ylabel("Predicted Valence")
plt.title("Actual vs Predicted Valence")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

# Visualize the correlation matrix
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"fontsize": 8})
plt.xticks(rotation=45)
plt.show()

# Scatter plot of one feature vs target variable
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], y)  # Scatter plot of 'speechiness' vs 'valence'
plt.xlabel("Speechiness")
plt.ylabel("Valence")
plt.title('Speechiness vs Valence')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
