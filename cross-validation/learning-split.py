import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
print ("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print ("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))
print ("X_test.shape: {}, y_test.shape: {}".format(X_test.shape, y_test.shape))

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print ("Score: {}".format(score))
