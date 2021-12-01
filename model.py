import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_to_data = "./aapl_data_file.csv"
X = pd.read_csv(path_to_data, sep=",").to_numpy()
y = X[:,1]
X = np.delete(X, 1, axis=1)
m = len(y)
alpha = 0.1
num_iters = 20
theta = np.zeros(len(X[0])+1)

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

X = feature_normalize(X)
y = feature_normalize(y)
X = np.concatenate([np.ones((m, 1)), X], axis=1)

theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel("number of iterations")
plt.ylabel("Cost J")
plt.show()
