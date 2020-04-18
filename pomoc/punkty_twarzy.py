import numpy as np
import matplotlib.pyplot as plt

data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]
## usunięcie złej twarzyczki
X = np.delete(X, (8656), axis=0)
y = np.delete(y, (8656), axis=0)

row = 5

xx, yy = X[row, ::2], X[row, 1::2]
plt.plot(xx, yy, 'bo')

for indx in range(len(xx)):
    plt.annotate(str(indx), (xx[indx], yy[indx]), fontweight = 'book')
plt.gcf().set_size_inches(8, 9)
plt.savefig('obrazki/twarz.png')

plt.show()