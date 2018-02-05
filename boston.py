import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.datasets import load_boston
# =============================================================================
# data = load_boston()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# lb = preprocessing.LabelBinarizer()
# print (df.head())
# style.use('ggplot')
# df = df.astype(float)
# X=np.array(df.drop(['LSTAT'],1),dtype=np.float64) 
# X = preprocessing.scale(X)
# Y=np.array(df['LSTAT'],dtype=np.float64)
# #Y=Y.astype('float')
# 
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train,Y_train)
# accuracy = clf.score(X_test,Y_test)
# print (accuracy)
# =============================================================================

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
# =============================================================================
# print (df.head())
# print (df.info())
# #print(df["ocean_proximity"].value_counts())
# df.hist(bins=50, figsize=(20,15))
# plt.show()
# =============================================================================

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
housing = train_set.copy()     
corr_matrix = housing.corr()
print (corr_matrix["LSTAT"].sort_values(ascending=False))