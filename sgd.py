import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, model_selection, metrics, preprocessing

## WCZYTANIE DANYCH
data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
y_labels = ('Mężczyzna', 'Kobieta')

## Liczba mężczyzn/kobiet
print("Liczba mężczyzn:", len(y[y==0]))
print("Liczba kobiet:", len(y[y==1]))

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

## Rozdzielenie danych do późniejszego liczenia 'accuracy' i 'confusion matrix'
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.1, stratify = y)

## UTWORZENIE OBIEKTU KLASYFIKATORA
clf = linear_model.SGDClassifier(loss='squared_hinge', alpha=3.6e-08, fit_intercept=False, n_jobs=-1)

## CROSS-VALIDACJA
scores = model_selection.cross_validate(clf, X_train, y_train, return_train_score=True, return_estimator=True, n_jobs=-1)
print('The score array for test scores on each cv split:', scores['test_score'])
print('Mean of above:', scores['test_score'].mean())

## WYBRANIE NAJLEPSZEGO ESTYMATORA I PREDYKCJA DLA WSZYTKICH DANYCH
best_clf = scores['estimator'][np.argmax(scores['test_score'])]
print('Accuracy on final set:', best_clf.score(X_test, y_test))

## TWORZENIE CONFUSSION MATRICES
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Confusion matrices')

ax1.set_title('Nie znormalizowany')
conf_mat_disp = metrics.plot_confusion_matrix(best_clf, X_test, y_test, display_labels=y_labels, cmap=plt.cm.Blues, ax=ax1)
ax2.set_title('Znormalizowany względem wartości prawdziwej')
conf_mat_disp = metrics.plot_confusion_matrix(best_clf, X_test, y_test, display_labels=y_labels, normalize='true', cmap=plt.cm.Blues, ax=ax2)

plt.show()