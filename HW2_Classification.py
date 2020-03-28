import numpy as np

np.random.seed(0)

X_train_fpath = 'Data/hw2_data/X_train'
Y_train_fpath = 'Data/hw2_data/Y_train'
X_test_fpath = 'Data/hw2_data/X_test'
output_fpath = 'Data/hw2_results/output_{}.csv'

# Readin data
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

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

    return X, Y

def gradient(X, Y, w):
    Y_pred = logistic_regression(X, w)
    loss = Y_pred - Y
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


# Normalization
X_train, X_mean, X_std = Normalization(X_train)
X_test = (X_test - X_mean)/ X_std
print(X_train.shape, np.ones((X_train.shape[0], 1)).shape)
X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
# Split train, val dataset

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, ratio=0.8)


'''
Training process
'''

Iterations = 30
batchsize = 10
lr = 0.2

w = np.zeros((X_train.shape[1], 1))
Y_pred = logistic_regression(X_train, w)

step = 1
for epoch in range(Iterations):

    for i in range(int(np.floor(len(X_train)/batchsize))):
        X_mini = X_train[i*batchsize: (i+1)*batchsize, :]
        Y_mini = Y_train[i*batchsize: (i+1)*batchsize, :]

        w_grad= gradient(X_mini, Y_mini, w)
        Y_pred = logistic_regression(X_mini, w)
        w = w - lr*w_grad

        step += 1

    Y_pred = logistic_regression(X_train, w)
    loss = cross_entropy(Y_train, Y_pred)/len(Y_train)
    print(loss)
    Y_pred_label = np.round(Y_pred)
    acc = 1 - np.sum(np.abs(Y_pred_label - Y_train))/Y_train.shape[0]
    print(acc)





