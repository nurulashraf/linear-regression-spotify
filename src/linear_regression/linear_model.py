from sklearn.linear_model import LinearRegression

def train_linear_model(X, y):
    model = LinearRegression()
    X_train = X_train.reshape(-1, 1)
    model.fit(X_train,y_train)
    return model
