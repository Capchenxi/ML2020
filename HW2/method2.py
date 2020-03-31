import numpy as np
import matplotlib.pyplot as plt
from utils import *

'''
==================================
|  Probalistic Generative Model  |
==================================
'''
np.random.seed(0)

X_train_fpath = 'Data/hw2_data/X_train'
Y_train_fpath = 'Data/hw2_data/Y_train'
X_test_fpath = 'Data/hw2_data/X_test'
output_fpath = 'Data/hw2_results/output_{}.csv'

with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, ratio=1)

# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)

# Compute in-class covariance
cov_0 = np.zeros((X_train.shape[1], X_train.shape[1]))
cov_1 = np.zeros((X_train.shape[1], X_train.shape[1]))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_0 - mean_1)
b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])
w = np.concatenate((np.reshape(w, (1, -1)), np.reshape(b,(1,1))), axis=1)
X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
Y_train_pred = logistic_regression(X_train, w.T)
Y_train_pred_label = np.round(Y_train_pred)

loss = cross_entropy(Y_train, Y_train_pred)[0][0]/len(Y_train)
Y_train_pred_label = np.reshape(Y_train_pred_label, (-1, ))
acc = 1 - np.sum(np.abs(Y_train_pred_label - Y_val)/Y_val.shape[0])

print(loss, acc)