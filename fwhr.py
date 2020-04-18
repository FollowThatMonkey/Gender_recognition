import numpy as np
from sklearn import linear_model, metrics, model_selection
import matplotlib.pyplot as plt

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
    xa, xb = xx[72], xx[105] # źrenice - w celu wyznaczenia kąta rotacji
    ya, yb = yy[72], yy[105]

    theta = -np.arctan((ya-yb)/(xa-xb))
    if (ya-yb)/(xa-xb) < -8: # jedna z twarzy jest pod +/-kątem prostym w lewo - ten warunek (trochę bezmyślny) pozwala poprawnie tę twarz przekręcić
        theta -= np.pi
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    xx, yy = np.dot(R, [xx, yy])
## ponowne przypisanie do X
    X[row, ::2] = xx
    X[row, 1::2] = yy

# Liczenie FWHR
width, height = np.zeros((len(y), 1)), np.zeros((len(y), 1))
for row, val in enumerate(y):
    xx, yy = X[row, ::2], X[row, 1::2]
    meanup = (yy[34]+yy[41])/2
    meandown = (yy[39]+yy[26])/2
    
    meanleft = (xx[10]+xx[9]+xx[8])/3
    meanright = (xx[15]+xx[14]+xx[17])/3
    
    width[row] = meanright-meanleft
    height[row] = meanup-meandown
fwhr = width/height

# usuwanie twarzyczki gdy fwhr <1.5 lub >3.15
row = 0
while row < len(y):
    if fwhr[row] < 1.5 or fwhr[row] > 3.15:
        y = np.delete(y, (row), axis=0)
        fwhr = np.delete(fwhr, (row), axis=0)
        width = np.delete(width, (row), axis=0)
        height = np.delete(height, (row), axis=0)
        row -= 1
    row += 1

# rozdzielenie danych po równo
fwhr_y = np.hstack((fwhr, y.reshape((len(y), 1))))
fwhr_men, fwhr_women = fwhr_y[y==0], fwhr_y[y==1]
print('fwhr men:', fwhr_men.shape)
print('fwhr women:', fwhr_women.shape)
for i in range(5): np.random.shuffle(fwhr_women)
fwhr_women = fwhr_women[:fwhr_men.shape[0], :]
fwhr = np.vstack((fwhr_men, fwhr_women))
for i in range(5): np.random.shuffle(fwhr)
fwhr, y = fwhr[:,0], fwhr[:,1]
fwhr.shape = len(fwhr), 1

## Rozdzielenie danych do późniejszego liczenia 'accuracy', 'confusion matrix' oraz 'ROC'
X_train, X_test, y_train, y_test = model_selection.train_test_split(fwhr, y, test_size = 0.1, stratify = y) #stratify żeby starał się po równo rozdzielić klasy

## UTWORZENIE OBIEKTU KLASYFIKATORA WRAZ Z CROSS-VALIDACJĄ
Cs = np.linspace(1e-10, 1, 50)
clf = linear_model.LogisticRegressionCV(Cs = Cs, max_iter=10000, n_jobs=-1).fit(X_train, y_train)
print('Accuracy:', clf.score(X_test, y_test))

## CONFUSION MATRICES
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Confusion matrices')

ax1.set_title('Nie znormalizowany')
conf_mat_disp = metrics.plot_confusion_matrix(clf, X_test, y_test, display_labels=y_labels, cmap=plt.cm.Blues, ax=ax1)
ax2.set_title('Znormalizowany względem wartości prawdziwej')
conf_mat_disp_normalized = metrics.plot_confusion_matrix(clf, X_test, y_test, display_labels=y_labels, normalize='true', cmap=plt.cm.Blues, ax=ax2)
plt.show()
# prawdopodobnie przypisuje wszystko kobietom, bo ich danych jest więcej!?

## KRZYWA ROC
roc = metrics.plot_roc_curve(clf, X_test, y_test)
plt.show()
# kształt TPR=FPR wskazuje na zupełnie losową klasyfikację