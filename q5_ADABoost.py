"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn import tree
from sklearn.metrics import precision_score, recall_score
import csv
from sklearn.datasets import make_classification
from ensemble.ADABoost import AdaBoostClassifier
#from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
#from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'entropy'
Dtree= tree.DecisionTreeClassifier(max_depth=1,criterion='entropy')
Classifier_AB = AdaBoostClassifier(base_estimator=Dtree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
#[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Classification data set using the entire data set

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X=pd.DataFrame(X)
y=pd.Series(y)
n_samples=len(X)
train_index=int(0.6*n_samples)
X_train=X.iloc[:train_index]
Y_train=y.iloc[:train_index]

X_test=X.iloc[train_index:]
Y_test=y.iloc[train_index:]

n_estimators = 3
Dtree =  tree.DecisionTreeClassifier( max_depth=1,criterion='entropy')

Classifier_AB = AdaBoostClassifier(base_estimator=Dtree, n_estimators=n_estimators )
Classifier_AB.fit(X_train, Y_train)

y_hat = Classifier_AB.predict(X_test)

print("Accuracy:", accuracy(y_hat,Y_test))

for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat,Y_test, cls))
    print('Recall for ',cls ,': ', recall(y_hat, Y_test, cls))