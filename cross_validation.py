
"""cross validation.ipynb

"""

import pandas as pd
df=pd.read_csv('diabetes1.csv')
df.head()

###  Independent And dependent features
X=df.iloc[:,2:]
y=df.iloc[:,1]

y

y.value_counts()

"""HoldOut Validation Approach- Train And Test Split"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)

"""K Fold Cross Validation"""

from sklearn.model_selection import KFold
model=DecisionTreeClassifier()
kfold_validation=KFold(10)

import numpy as np
from sklearn.model_selection import cross_val_score
results=cross_val_score(model,X,y,cv=kfold_validation)
print(results)
print(np.mean(results))

"""stratified K-fold Cross Validation"""

from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold(n_splits=5)
model=DecisionTreeClassifier()
scores=cross_val_score(model,X,y,cv=skfold)
print(np.mean(scores))

scores

"""Leave One Out Cross Validation(LOOCV)"""

from sklearn.model_selection import LeaveOneOut
model=DecisionTreeClassifier()
leave_validation=LeaveOneOut()
results=cross_val_score(model,X,y,cv=leave_validation)

results

print(np.mean(results))

"""Repeated Random Test-Train Splits"""

from sklearn.model_selection import ShuffleSplit
model=DecisionTreeClassifier()
ssplit=ShuffleSplit(n_splits=10,test_size=0.30)
results=cross_val_score(model,X,y,cv=ssplit)

results

np.mean(results)