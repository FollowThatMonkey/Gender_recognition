import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, model_selection, metrics, preprocessing

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
y_labels = ('Mężczyzna', 'Kobieta')

## PREPROCESSING
for row, val in enumerate(y):
    x0, y0 = X[row, 0], X[row, 1]
    
    for indx, val in enumerate(X[row, :]):
        if indx%2 == 0:
            X[row, indx] -= x0
        else:
            X[row, indx] -= y0
X = preprocessing.Normalizer().fit_transform(X)

## UTWORZENIE OBIEKTU KLASYFIKATORA
clf = linear_model.Perceptron(alpha=1e-5, n_jobs=-1)

## CROSS-VALIDACJA
scores = model_selection.cross_validate(clf, X, y, return_estimator=True, n_jobs=-1)
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