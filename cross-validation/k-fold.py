import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)

for train, test in kf.split(X):
    print ("{} {}".format(train, test))
