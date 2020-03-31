import pandas as pd
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt

'''
Data Preprocessing
'''

# download dataset from https://drive.google.com/uc?id=1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm

data = pd.read_csv('Data/hw1_data/train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# 4320 rows x 24 columns : 18 (features) x 240 (days) = 4320; 24 columns: 24hrs

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# month_data = {0: sample, 1: sample, ...} sample.shape = 18 x 480
# x = features in first 9hrs ; x.shape = (471, 18x9)
# y = PM2.5 in 10th hr; y.shape = (471, 1)

x = np.empty([12 * 471, 18 * 9 + 1], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :-1] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            x[month * 471 + day * 24 + hour, -1] = 1
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value

# Normalization
# mean_x = np.mean(x, axis = 0) #18 * 9
# std_x = np.std(x, axis = 0) #18 * 9
# for i in range(len(x)): #12 * 471
#     for j in range(len(x[0])): #18 * 9
#         if std_x[j] != 0:
#             x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# Split dataset into training and validation set.
print(x.shape)
x_train = x[: math.floor(len(x) * 0.8), :]
y_train = y[: math.floor(len(y) * 0.8), :]
x_val = x[math.floor(len(x) * 0.8): , :]
y_val = y[math.floor(len(y) * 0.8): , :]

'''
Define different optimization 
'''

def gradientDescent(X, Y, lr, w, iterations, L2):
    cost_list = []
    for i in range(iterations):
        Y_hat = np.dot(X, w)
        loss = Y_hat - Y
        cost = np.sum(loss**2)/X.shape[0]
        cost_list.append(cost)
        gradient = 2*np.dot(X.transpose()/X.shape[0], loss) + 2*L2*w
        w -= lr*gradient
        if i % 1000 == 0:
            print("iteration:{}, cost:{} ".format(i, cost))

    return w, cost_list

def Adagrad(X, Y, lr, w, iterations, L2):
    prev_grad = np.zeros(w.shape)
    epsilon = 1e-8
    cost_list = []
    for i in range(iterations):
        Y_hat = np.dot(X, w)
        loss = Y_hat - Y
        cost = np.sum(loss**2)/X.shape[0]

        cost_list.append(cost)
        gradient = 2*np.dot(X.transpose(), loss) + 2*L2*w
        prev_grad = prev_grad + gradient**2
        adapt_lr = lr / np.sqrt(prev_grad + epsilon)
        w = w - adapt_lr * gradient
        if i%1000 == 0:
            print("iteration:{}, cost:{} ".format(i, cost))

    return w, cost_list

'''
Training error with different optimization
'''
# Parameters initialization
iterations = 10000
lr = 0.000001

#Initialize weight every time before training,
#otherwise, it will use the weight from previous training.
w0 = np.zeros([9*18+1, 1])
w_gd, cost_gd = gradientDescent(x_train, y_train, lr, w0, iterations, 0)
w0 = np.zeros([9*18+1, 1])
w_gd_L2, cost_gd_L2 = gradientDescent(x_train, y_train, lr, w0, iterations, 100)
w0 = np.zeros([9*18+1, 1])
w_ada, cost_ada = Adagrad(x_train, y_train, 0.5, w0, iterations, 0)
print(cost_ada)

plt.figure()
plt.plot(np.arange(len(cost_gd)), cost_gd, '--b', label='Gradient Descent')
plt.plot(np.arange(len(cost_gd_L2)), cost_gd_L2, '--r', label='Gradient Descent w/ L2')
# plt.plot(np.arange(len(cost_ada)), cost_ada, '--g', label='Adagrad')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Cost Function(MSE)')
plt.title('Training error comparison without Data Normalization')
plt.savefig(os.path.join(os.path.dirname("Data/hw1_results/cost_compare")))
plt.show()

'''
Validation error with different optimization
'''

y_pred_gd = np.dot(x_val, w_gd)
y_pred_gd_L2 = np.dot(x_val, w_gd_L2)
y_pred_ada = np.dot(x_val, w_ada)
print(y_pred_ada, y_pred_gd)

'''
Visualization
'''

plt.figure(figsize=(12,6))
plt.title('Predict PM2.5 with different optimization')
# plt.subplot(1,3,1)
# plt.plot(np.arange(1, len(y_val)+1), y_val, 'b--', label='GroundTruth')
plt.subplot(1,3,1)
plt.plot(np.arange(1, len(y_pred_gd)+1), y_pred_gd,'r--', label='Gradient Descent')
plt.legend()
plt.subplot(1,3,2)
plt.plot(np.arange(1, len(y_pred_gd_L2)+1), y_pred_gd_L2,'y--', label='Gradient Descent w/ L2')
plt.legend()
plt.subplot(1,3,3)
plt.plot(np.arange(1, len(y_pred_ada)+1), y_pred_ada,'g--', label='Adagrad')
plt.legend()
plt.savefig(os.path.join(os.path.dirname("Data/hw1_results/opt_compare")))
plt.show()