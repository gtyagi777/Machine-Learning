import pandas as pd
import quandl, math, datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']= (df['Adj. High']- df['Adj. High'])/df['Adj. Close'] * 100.0
df['PCT_Change']= (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open'] * 100.0

df =df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
##print (df.head())
forecast_col = 'Adj. Close'

df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))   
print(forecast_out)          
df['label']= df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace = True)

Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)
#clf = LinearRegression(n_jobs=-1)
#clf.fit(X_train,Y_train)
#with open('linearregression.pickle','wb') as f:
#    pickle.dump(clf,f)
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test,Y_test)
#print(accuracy)
forecast_set = clf.predict(X_lately)

print (forecast_set,accuracy)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + 86400

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=86400
    df.loc[next_date]= [np.nan for j in range(len(df.columns) -1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()