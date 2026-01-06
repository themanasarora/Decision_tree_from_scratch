from sklearn.tree import DecisionTreeClassifier
from DecisionTree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

data=datasets.load_breast_cancer()
X,y=data.data,data.target
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

my_clf = DecisionTree(max_depth=3)
my_clf.fit(X_train, Y_train)
my_acc = np.mean(my_clf.predict(X_test) == Y_test)

sk_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
sk_clf.fit(X_train, Y_train)
sk_acc = np.mean(sk_clf.predict(X_test) == Y_test)

print("My tree:", my_acc)
print("Sklearn:", sk_acc)
