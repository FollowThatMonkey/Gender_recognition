import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
y_labels = ('Mężczyzna', 'Kobieta')

## UTWORZENIE OBIEKTU KLASYFIKATORA WRAZ Z CROSS-VALIDACJĄ
clf = linear_model.LogisticRegressionCV(solver='newton-cg').fit(X, y)
print(clf.score(X, y))
print(clf.scores_)

## TWORZENIE CONFUSSION MATRICES
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Confusion matrices (not)normalized')

ax1.set_title('Nie znormalizowany')
conf_mat_disp = metrics.plot_confusion_matrix(clf, X, y, display_labels=y_labels, cmap=plt.cm.Blues, ax=ax1)
ax2.set_title('Znormalizowany względem wartości prawdziwej')
conf_mat_disp_normalized = metrics.plot_confusion_matrix(clf, X, y, display_labels=y_labels, normalize='true', cmap=plt.cm.Blues, ax=ax2)

plt.show()