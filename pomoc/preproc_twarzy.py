import numpy as np
import matplotlib.pyplot as plt

data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]

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

    plt.plot(xx, yy, 'bo', ms=0.1)
plt.gcf().set_size_inches(8, 9)
plt.savefig('obrazki/wszystkie_twarze.png')
plt.show()