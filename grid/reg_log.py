import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics, preprocessing, decomposition

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
y_labels = ('Mężczyzna', 'Kobieta')

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
X = decomposition.PCA().fit_transform(X)

## UTWORZENIE OBIEKTU KLASYFIKATORA WRAZ Z CROSS-VALIDACJĄ
Cs = np.linspace(10, 12, 60)
clf = linear_model.LogisticRegressionCV(Cs = Cs, fit_intercept=True, max_iter=10000, n_jobs=-1).fit(X, y)
print(clf.score(X, y))
print(clf.scores_)
print(clf.C_)

## TWORZENIE CONFUSSION MATRICES
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Confusion matrices (not)normalized')

ax1.set_title('Nie znormalizowany')
conf_mat_disp = metrics.plot_confusion_matrix(clf, X, y, display_labels=y_labels, cmap=plt.cm.Blues, ax=ax1)
ax2.set_title('Znormalizowany względem wartości prawdziwej')
conf_mat_disp_normalized = metrics.plot_confusion_matrix(clf, X, y, display_labels=y_labels, normalize='true', cmap=plt.cm.Blues, ax=ax2)

plt.show()
