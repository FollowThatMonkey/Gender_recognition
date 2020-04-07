import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, model_selection, metrics

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
y_labels = ('Mężczyzna', 'Kobieta')

## UTWORZENIE OBIEKTU KLASYFIKATORA
clf = svm.SVC(C=2e4)

## CROSS-VALIDACJA
scores = model_selection.cross_validate(clf, X, y, return_estimator=True)
print(scores['test_score'])
print(scores['test_score'].mean())

## WYBRANIE NAJLEPSZEGO ESTYMATORA I PREDYKCJA DLA WSZYTKICH DANYCH
best_clf = scores['estimator'][np.argmax(scores['test_score'])]

## TWORZENIE CONFUSSION MATRICES
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Confusion matrices (not)normalized')

ax1.set_title('Nie znormalizowany')
conf_mat_disp = metrics.plot_confusion_matrix(best_clf, X, y, display_labels=y_labels, cmap=plt.cm.Blues, ax=ax1)
ax2.set_title('Znormalizowany względem wartości prawdziwej')
conf_mat_disp = metrics.plot_confusion_matrix(best_clf, X, y, display_labels=y_labels, normalize='true', cmap=plt.cm.Blues, ax=ax2)

plt.show()