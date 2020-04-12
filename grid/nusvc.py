import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import svm, model_selection, preprocessing, decomposition
import csv

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]

## PREPROCESSING
## usunięcie złej twarzyczki
X = np.delete(X, (8656), axis=0)
y = np.delete(y, (8656), axis=0)
## zerowanie podbródka
X[:, ::2] -= X[:, 0].reshape((X.shape[0], 1))
X[:, 1::2] -= X[:, 1].reshape((X.shape[0], 1))
## rotacja
for row in range(len(y)):
    xx, yy = X[row, ::2], X[row, 1::2]
    xa, xb = xx[72], xx[105]
    ya, yb = yy[72], yy[105]

    theta = -np.arctan((ya-yb)/(xa-xb))
    if (ya-yb)/(xa-xb) < -8:
        theta -= np.pi
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    xx, yy = np.dot(R, [xx, yy])
## skalowanie
    xx /= max(np.absolute(xx))
    yy /= max(np.absolute(yy))
## przypisanie do X
    X[row, ::2] = xx
    X[row, 1::2] = yy

## UTWORZENIE OBIEKTU KLASYFIKATORA
clf = svm.NuSVC()

## UTWORZENIE GRID_SEARCH'A
nu = np.linspace(0.5, 0.56, 10)
kernel = ('linear', 'rbf')
parameters = {'kernel': kernel, 'nu': nu}

score = model_selection.GridSearchCV(clf, parameters, n_jobs=-1)
score.fit(X, y)

with open('wyniki/nusvc_najlepszy.csv', 'w') as f:
    w = csv.writer(f)
    for key, val in score.best_params_.items():
        w.writerow([key, val])
        
with open('wyniki/nusvc_wyniki.csv', 'w') as f:
    w = csv.writer(f)
    for key, val in score.cv_results_.items():
        w.writerow([key, val])
        