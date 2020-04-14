import numpy as np
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt

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

# usuwanie twarzyczki gdy oczko < delta
row = 0
delta = 0.035
while row < len(y):
    xx, yy = X[row, ::2], X[row, 1::2]
    if yy[34]-yy[3] < delta or yy[41]-yy[43] < delta:
        X = np.delete(X, (row), axis=0)
        y = np.delete(y, (row), axis=0)
        row -= 1
    row += 1

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

# usuwanie twarzyczki gdy fwhr <3.1 lub >5.5
row = 0
while row < len(y):
    if fwhr[row] < 3.1 or fwhr[row] > 5.5:
        y = np.delete(y, (row), axis=0)
        fwhr = np.delete(fwhr, (row), axis=0)
        width = np.delete(width, (row), axis=0)
        height = np.delete(height, (row), axis=0)
        row -= 1
    row += 1

for row, val in enumerate(y):
    plt.plot(X[row,::2], X[row,1::2], 'bo')
    plt.show()