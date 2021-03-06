import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.model_selection import cross_val_score

train_size = 124
test_size = 32

path_to_data = "./tsla_data_file_with_sent.csv"

def sigmoid(z):
	z = np.array(z)
	g = np.zeros(z.shape)
	g = (1/(1+np.exp(-z)))
	return g


def feature_normalize(X): 
	return (X-X.mean(0))/X.std(0)

def compute_cost(X, y, theta):
	h = np.sum(np.transpose(theta)*X, axis=1)
	J = np.sum((h-y)**2)/(2*m)	
	return J

def gradient_descent(X, y, theta, alpha, num_iters):
	theta = theta.copy()
	J_history = []
	for i in range(num_iters):
		for i in range(m):
			h = theta@X[i]
			for j in range(theta.shape[0]):
				if j == 0:
					theta[j] -= (alpha/m)*np.sum(h-y[i])
				else: 
					theta[j] -= (alpha/m)*np.sum(h-y[i])*X[i][j]
		J_history.append(lr_cost(theta, X, y))
	return theta, J_history



def lr_cost(theta, X, y):
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    for i in range(m):
        J += ((-1 * y[i]) * np.log(sigmoid(theta.T.dot(X[i])))) - ((1 - y[i]) * np.log(1 - sigmoid(theta.T.dot(X[i]))))
    J = J/m

    for j in range(len(theta)):
        for i in range(m):
            grad[j] += (sigmoid(theta.T.dot(X[i])) - y[i]) * X[i][j]
        grad[j] = grad[j] / m
    
    # =============================================================
    return J, grad

def predict(theta, X):
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    for i in range(len(X)):
        if sigmoid(theta.T.dot(X[i])) >= 0.5:
            p[i] = 1
    return p

def cross_validate(X, y, n):
    options = {'maxiter': 100}
    #res = optimize.minimize(lr_cost, theta, (X, y), jac=True, method='TNC', options=options)
    scores = []
    X_folds = np.array_split(X, n)
    y_folds = np.array_split(y, n)

    for i in range(n):
        
        test_X = X_folds[i]
        test_y = y_folds[i]
        training_X = X_folds.copy()
        training_y = y_folds.copy()
        training_X.pop(i)
        training_y.pop(i)
        training_X = np.concatenate(training_X)
        training_y = np.concatenate(training_y)

        theta = np.zeros(len(training_X[0]))
        res = optimize.minimize(lr_cost, theta, (training_X, training_y), jac=True, method='TNC', options=options)
        predictions = predict(res.x, test_X)
        accuracy = np.mean(predictions == test_y)
        scores.append(accuracy)

    print(scores)
    return(np.mean(accuracy))



X = pd.read_csv(path_to_data, sep=",").to_numpy()
np.random.shuffle(X)
# X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
# y = X[:,0]

increase = np.array(X[:,0] > X[:,1], dtype=int)
# increase = increase.reshape((len(increase), 1))
y = increase
X = np.delete(X, 0, axis=1)

# Just compound sentiment
X = np.delete(X, len(X[0])-2, axis=1)
X = np.delete(X, len(X[0])-2, axis=1)
X = np.delete(X, len(X[0])-2, axis=1)

# Remove all sentiment
# X = np.delete(X, len(X[0])-1, axis=1)
# X = np.delete(X, len(X[0])-1, axis=1)
# X = np.delete(X, len(X[0])-1, axis=1)
# X = np.delete(X, len(X[0])-1, axis=1)


# Remove high value
#X = np.delete(X, 1, axis=1)


alpha = 0.1
num_iters = 2
theta = np.zeros(len(X[0])+1)

X = feature_normalize(X)

X = np.concatenate([np.ones((len(X), 1)), X], axis=1)

X_test = X[train_size:]
y_test = y[train_size:]

X = X[:train_size]
y = y[:train_size]
m = len(y)

options = {'maxiter': 100}
res = optimize.minimize(lr_cost, theta, (X, y), jac=True, method='TNC', options=options)


predictions = predict(res.x, X_test)
print(predictions)
print('Train Accuracy: {:.2f} %'.format(np.mean(predictions == y_test)))
