import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import recall_score
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
print ("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print ("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))
print ("X_test.shape: {}, y_test.shape: {}".format(X_test.shape, y_test.shape))

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)

scores = cross_validate(clf, X_train, y_train, scoring=scoring, return_estimator=True)
print ("Scores: {}".format(scores))

lst = scores["test_f1_macro"]
mvalue = max(lst)

idxes = []
for idx in range(len(lst)):
    if lst[idx] == mvalue:
        idxes.append(idx)

# collect models that output the maximum value, removing the duplicated model
candidates = []
models = scores["estimator"]
for idx in idxes:
    m = models[idx]

    included = True
    for c in candidates:
        if m.__class__.__name__ == c.__class__.__name__:
            mparams = m.get_params()
            cparams = c.get_params()

            for k in mparams.keys():
                if k not in cparams:
                    included = False
                    break
                elif mparams[k] != cparams[k]:
                    included = False
                    break

    if not included or len(candidates) == 0:
        candidates.append(m)

# list parameters
print ("\n==========")
for model in candidates:
    print ("Model: {}".format(model))
    print ("Name: {}".format(model.__class__.__name__))
    print ("Parameters: {}".format(model.get_params()))
    print ("==========")
