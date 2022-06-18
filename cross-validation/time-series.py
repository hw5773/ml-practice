import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print ("tscv: {}".format(tscv))

for train_index, test_index in tscv.split(X):
    print ("{} {}".format(train_index, test_index))
