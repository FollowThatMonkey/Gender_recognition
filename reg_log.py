import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]

clf = linear_model.LogisticRegressionCV()
clf_pre = linear_model.LogisticRegressionCV()

X_pre = preprocessing.StandardScaler().fit_transform(X)

clf.fit(X, y)
clf_pre.fit(X_pre, y)

print("Bez preprocessingu:", clf.score(X, y))
print("Po preprocessingu:", clf_pre.score(X, y))