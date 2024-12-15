X = df['valence'].values
y = df['danceability'].values

# Creating linear model
model = LinearRegression()

# Reshape X_train to be a 2D array
X_train = X_train.reshape(-1, 1)
model.fit(X_train,y_train)
