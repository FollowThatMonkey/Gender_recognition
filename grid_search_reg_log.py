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
    'C': (1e-4, 1e-3, 1e-2, 1e-1, 1),
    'solver': ('liblinear', 'lbfgs'),
    'max_iter': (100, 1000, 10000),
    'n_jobs': (-1,)
}
lin_reg = linear_model.LogisticRegression()
clf = model_selection.GridSearchCV(lin_reg, parameters)
clf.fit(X, y)

w = csv.writer(open("cv_results_reg_log.csv", 'w'))
for key, val in clf.cv_results_.items():
    w.writerow([key, val])
    
w = csv.writer(open("best_params_reg_log.csv", 'w'))
for key, val in clf.best_params_.items():
    w.writerow([key, val])

print("Najlepszy wynik:", clf.best_score_)