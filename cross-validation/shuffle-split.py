import numpy as np
from sklearn.model_selection import ShuffleSplit

X = np.arange(10)
print ("X: {}".format(X))

ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)

for train_index, test_index in ss.split(X):
    print ("{} {}".format(train_index, test_index))
