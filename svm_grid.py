import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import svm, model_selection
import csv

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]

## UTWORZENIE OBIEKTU KLASYFIKATORA
clf = svm.SVC()

## UTWORZENIE GRID_SEARCH'A
parameters = {'C': (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)}

score = model_selection.GridSearchCV(clf, parameters).fit(X, y)

with open('svm_grid_best.csv', 'w') as f:
    w = csv.writer(f)
    for key, val in score['best_params_']:
        w.writerow([key, val])
        
with open('svm_grid_score.csv', 'w') as f:
    w = csv.writer(f)
    for key, val in score['cv_results_']:
        w.writerow([key, val])
        