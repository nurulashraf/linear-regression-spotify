import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/datasets/spotify_tracks.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Select relevant features for the analysis
music = df[['energy', 'danceability', 'valence']]
print("\nSelected Features Preview:")
print(music.head())

# Define features (X) and target variable (y)
X = music.drop('valence', axis=1).values
y = music['valence'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)
y_actual = y_test

# Create a 3D plot for the model's plane
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Create a mesh grid for the model's surface
x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 50)
x2_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 50)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

# Predict y values (valence) for the mesh grid
y_mesh = model.predict(X_mesh).reshape(x1_mesh.shape)

# Plot the model's plane
ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.3, color='blue', label="Model Plane")
ax.set_title("3D Linear Regression: Actual vs Predicted vs Model")
ax.set_xlabel("Energy")
ax.set_ylabel("Danceability")
ax.set_zlabel("Valence")
ax.legend()
plt.show()

# Create a 3D plot for actual vs predicted data points
fig = plt.figure(figsize=(15, 20))
ax = fig.add_subplot(111, projection='3d')

# Plot actual test data
ax.scatter(X_test[:, 0], X_test[:, 1], y_actual, color='g', alpha=0.7, label="Actual Data")

# Plot predicted data
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='b', alpha=0.7, label="Predicted Data")

# Set x and y limits with minimum set to 0
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Labels and title
ax.set_xlabel("Energy (x1)")
ax.set_ylabel("Danceability (x2)")
ax.set_zlabel("Valence (y)")
ax.set_title("3D Linear Regression: Actual vs Predicted")
ax.legend()
plt.show()

# Plotly 3D plot for interactive visualization
# Create a scatter plot for actual data points
scatter_actual = go.Scatter3d(x=X_test[:, 0], y=X_test[:, 1], z=y_actual, mode='markers', marker=dict(color='green', size=5), name="Actual Data")

# Create a scatter plot for predicted data points
scatter_pred = go.Scatter3d(x=X_test[:, 0], y=X_test[:, 1], z=y_pred, mode='markers', marker=dict(color='blue', size=5), name="Predicted Data")

# Create a grid for the regression plane
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x_grid_flattened = np.c_[x1_grid.ravel(), x2_grid.ravel()]
y_grid_pred = model.predict(x_grid_flattened).reshape(x1_grid.shape)

# Create the regression plane
surface = go.Surface(x=x1_grid, y=x2_grid, z=y_grid_pred, colorscale='reds', opacity=0.5, showscale=False, name='Regression Plane')

# Create the layout with axis limits
layout = go.Layout(
    scene=dict(
        xaxis=dict(title="Energy", range=[0, X_test[:, 0].max()]),
        yaxis=dict(title="Danceability", range=[0, X_test[:, 1].max()]),
        zaxis=dict(title="Valence")
    ),
    title='3D Linear Regression: Actual vs Predicted with Regression Plane'
)

# Create the figure
fig = go.Figure(data=[scatter_actual, scatter_pred, surface], layout=layout)

# Show the plot
fig.show()

# Test the model with a specific example
print("\nExample Prediction:")
example = np.array([[0.97, 0.753]])
predicted_valence = model.predict(example)[0]
print(f"Predicted Valence for input {example}: {predicted_valence}")

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot the Actual vs Predicted values using Matplotlib
plt.figure(figsize=(15, 10))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Valence')
plt.ylabel('Predicted Valence')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.title('Actual vs Predicted Valence')
plt.show()
