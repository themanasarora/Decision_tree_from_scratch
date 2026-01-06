from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

# create moons dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.3,
    random_state=42
)

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train decision tree
clf = DecisionTree(max_depth=2)
clf.fit(X_train, Y_train)

# predictions
prediction = clf.predict(X_test)

# accuracy function
def accuracy(Y_test, y_pred):
    return np.sum(Y_test == y_pred) / len(Y_test)

acc = accuracy(Y_test, prediction)
print(acc)
