import numpy as np
import matplotlib.pyplot as plt
from utils import *

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
X_train, X_mean, X_std = Normalization(X_train)
X_test = (X_test - X_mean)/ X_std
print(X_train.shape, np.ones((X_train.shape[0], 1)).shape)
X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
# Split train, val dataset

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, ratio=0.9)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

'''
Training process
'''

Iterations = 30
batchsize = 8
lr = 0.0002

w = np.zeros((X_train.shape[1], 1))
Y_pred = logistic_regression(X_train, w)

step = 1
for epoch in range(Iterations):

    X_train, Y_train = shuffle(X_train, Y_train)

    for i in range(int(np.floor(len(X_train)/batchsize))):
        X_mini = X_train[i*batchsize: (i+1)*batchsize, :]
        Y_mini = Y_train[i*batchsize: (i+1)*batchsize, :]

        w_grad= gradient(X_mini, Y_mini, w)
        w = w - lr*w_grad
        Y_pred = logistic_regression(X_mini, w)
        step += 1

    Y_train_pred = logistic_regression(X_train, w)
    train_loss.append(cross_entropy(Y_train, Y_train_pred)[0][0]/len(Y_train))

    Y_train_pred_label = np.round(Y_train_pred)
    acc = 1 - np.sum(np.abs(Y_train_pred_label - Y_train))/Y_train.shape[0]
    train_acc.append(acc)

    Y_val_pred = logistic_regression(X_val, w)
    val_loss.append(cross_entropy(Y_val, Y_val_pred)[0][0]/len(Y_val))

    Y_val_pred_label = np.round(Y_val_pred)
    acc = 1 - np.sum(np.abs(Y_val_pred_label - Y_val)/Y_val.shape[0])
    val_acc.append(acc)

plt.figure(figsize=(12,3))
plt.subplot(1,2,1)
plt.plot(range(1, len(train_loss)+1), train_loss, 'b', label='train loss')
plt.plot(range(1, len(val_loss)+1), val_loss, 'r', label='val loss')
plt.legend()
plt.xlabel('Iteration times')
plt.ylabel('cross entropy loss')
plt.title('Compare loss between train and val sets')

plt.subplot(1,2,2)
plt.plot(range(1, len(train_acc)+1), train_acc, 'b', label='train accuracy')
plt.plot(range(1, len(val_acc)+1), val_acc, 'r', label='val accuracy')
plt.legend()
plt.xlabel('Iteration times')
plt.ylabel('Accuracy')
plt.title('Compare accuracy between train and val sets')
plt.show()

