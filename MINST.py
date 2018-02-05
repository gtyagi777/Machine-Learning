from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
# =============================================================================
# some_digit = X[36001]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
# interpolation="nearest")
# plt.axis("off")
# plt.show()
# 
# =============================================================================
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# =============================================================================
# print (cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
# =============================================================================
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
