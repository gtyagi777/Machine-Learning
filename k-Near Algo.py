import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

dataset={'k':[[1,2],[2,3],[3,1]],'x':[[6,5],[7,7],[8,6]]}
new_feature = [5,7]

        
def k_nearest_neighbors(data, predict, k=3):
    if (len(data)>= k):
        warnings.warn('K is set to less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            eu_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([eu_distance,group])
    votes = [i[1]for i in sorted(distances)[:k]] 
    print (Counter(votes).most_common(1))    
    vote_result = Counter(votes).most_common(1)[0][0]      
    return vote_result
print(k_nearest_neighbors(dataset, new_feature,k=3))

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s =100 )
plt.scatter(new_feature[0], new_feature[1], color = 'red')        
plt.show()