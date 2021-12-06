import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_size = 124
test_size = 32

path_to_data = "./tsla_data_file_with_sent.csv"

def feature_normalize(X): 
	return (X-X.mean(0))/X.std(0)

def compute_cost(X, y, theta):
	h = np.sum(np.transpose(theta)*X, axis=1)
	J = np.sum((h-y)**2)/(2*m)	
	return J
def gradient_descent(X, y, theta, alpha, num_iters):
	theta = theta.copy()
	J_history = []
	for it in range(num_iters):
		for i in range(m):
			h = theta@X[i]
			for j in range(theta.shape[0]):
				if j == 0:
					theta[j] -= (alpha/m)*np.sum(h-y[i])
				else: 
					theta[j] -= (alpha/m)*np.sum(h-y[i])*X[i][j]
		J_history.append(compute_cost(X, y, theta))
	return theta, J_history

X = pd.read_csv(path_to_data, sep=",").to_numpy()
# X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
y = X[:,0]
X = np.delete(X, 0, axis=1)

alpha = 0.1
num_iters = 20
theta = np.zeros(len(X[0])+1)


X = feature_normalize(X)
y = feature_normalize(y)

X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
X_test = X[train_size:]
y_test = y[train_size:]

X = X[:train_size]
y = y[:train_size]
m = len(y)

theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel("number of iterations")
plt.ylabel("Cost J")
#plt.show()

# print("X_test@theta")
# print(X_test@theta)
# print("y_test")
# print(y_test)
predictions = X_test@theta
MSE = np.square(np.subtract(y_test, predictions)).mean()
print(MSE)
# print(X_test.shape)
# print(theta.shape)