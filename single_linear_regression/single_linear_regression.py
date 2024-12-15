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

# Checking if data is linear
X = df['valence'].values
y = df['danceability'].values

# Calculate the correlation coefficient
corr, _ = pearsonr(X, y)
print(f"Correlation: {corr}")

# Visualize the correlation matrix
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"fontsize": 8})
plt.show()

# Scatter plot of Valence vs Danceability
plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.xlabel("Valence")
plt.ylabel("Danceability")
plt.title('Valence vs Danceability')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train-Test Split Visualization
plt.figure(figsize=(10, 8))
plt.scatter(X_train, y_train, label='Training Data', color='r', alpha=0.5)
plt.scatter(X_test, y_test, label='Testing Data', color='g', alpha=0.5)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.title("Train Test Split")
plt.show()

# Create and train the Linear Regression model
model = LinearRegression()
X_train = X_train.reshape(-1, 1)
model.fit(X_train, y_train)

# Predict the target variable for the test set
X_test = X_test.reshape(-1, 1)
y_pred = model.predict(X_test)

# Visualizing model against test data
plt.figure(figsize=(10, 8))
plt.plot(X_test, y_pred, label='Linear Regression', color='r')
plt.scatter(X_test, y_test, label="Actual Test Data", color="g")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.title('Model Prediction vs Actual Data')
plt.show()

# Test the model with a specific example
example_valence = np.array([[0.821]])
predicted_danceability = model.predict(example_valence)[0]
print(f"\nPredicted Danceability for Valence 0.821: {predicted_danceability}")

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
