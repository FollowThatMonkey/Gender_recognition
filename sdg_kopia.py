#import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import csv

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
y_labels = ('Mężczyzna', 'Kobieta')

## GRID SEARCH
parameters = {
    'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
    'fit_intercept': (True, False),
    'alpha': (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3),
    'n_jobs': (-1,)
}
sdg = linear_model.SGDClassifier()
clf = model_selection.GridSearchCV(sdg, parameters)
clf.fit(X, y)

w = csv.writer(open("output.csv", 'w'))
for key, val in clf.cv_results_.items():
    w.writerow([key, val])

