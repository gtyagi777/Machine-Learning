import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, neighbors,svm
from sklearn.model_selection import train_test_split

df=pd.read_csv("breast-cancer.txt")
df.replace('?', -99999, inplace = True)
df.drop(['id'],1,inplace= True)

X=np.array(df.drop(['class'],1)) 
Y=np.array(df['class'])


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
clf = svm.SVC()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)
print (accuracy)
