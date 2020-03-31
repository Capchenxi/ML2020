import numpy as np

# Normalization
def Normalization(X):
    epsilon = 1e-8
    X_mean = np.mean(X, axis=0).reshape((1, len(X[0])))
    X_std = np.std(X, axis=0).reshape((1, len(X[0]))) + epsilon

    X = (X - X_mean)/X_std

    return X, X_mean, X_std

def train_val_split(X, Y, ratio=0.8):
    row = len(X)
    X_train = X[:int(row*ratio), :]
    Y_train = Y[:int(row*ratio), :]
    X_val = X[int(row*ratio):, :]
    Y_val = Y[int(row * ratio):, :]

    return X_train, Y_train, X_val, Y_val

def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)

    return X[randomize], Y[randomize]

def gradient(X, Y, w):
    Y_pred = logistic_regression(X, w)
    loss = Y - Y_pred
    w_grad = -np.dot(X.T, loss)

    return w_grad

def cross_entropy(Y, Y_pred):
    epsilon = 0
    error = -(np.dot(Y.T,np.log(Y_pred)) + np.dot((1 - Y.T),np.log(1 - Y_pred)))

    return error

def sigmoid(x):

    return np.clip(1/(1+np.exp(-x)), 1e-8, 1-(1e-8))


def logistic_regression(X, w):

    Y_pred = sigmoid(np.dot(X, w))

    return Y_pred
