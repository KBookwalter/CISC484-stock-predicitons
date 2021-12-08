import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import utils
from sklearn import svm
from sklearn.model_selection import cross_val_score

train_size=124
test_size=32

path_to_data = "./tsla_data_file_with_sent.csv"

def feature_normalize(X): 
	return (X-X.mean(0))/X.std(0)

X = pd.read_csv(path_to_data, sep=',').to_numpy()

np.random.shuffle(X)

increase = np.array(X[:,0] > X[:,1], dtype=int)
y = increase
X = np.delete(X, 0, axis=1)

#X = np.concatenate([np.ones((len(X), 1)), X], axis=1)

# Just Compound Sentiment
# X = np.delete(X, len(X[0])-2, axis=1)
# X = np.delete(X, len(X[0])-2, axis=1)
# X = np.delete(X, len(X[0])-2, axis=1)

# Remove all sentiment
# X = np.delete(X, len(X[0])-1, axis=1)
# X = np.delete(X, len(X[0])-1, axis=1)
# X = np.delete(X, len(X[0])-1, axis=1)
# X = np.delete(X, len(X[0])-1, axis=1)

X = feature_normalize(X)

X_test = X[train_size:]
y_test = y[train_size:]

X = X[:train_size]
y = y[:train_size]
m = len(y)

model = svm.SVC(C=1000)
model.fit(X, y)

correct = 0

pred = model.predict(X_test)
print(pred)

for i in range(len(pred)):
    if pred[i] == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print(accuracy)
#print(model.support_vectors_)

# scores = cross_val_score(model, X, y, cv=10)
# print(scores)
# print(scores.mean())
