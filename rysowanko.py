import numpy as np
import matplotlib.pyplot as plt

data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]

fig, (ax1, ax2) = plt.subplots(2)

ax1.set_title("Mężczyźni")
ax2.set_title("Kobiety")

for indx, gender in enumerate(y[:2]):
    x = []
    y = []

    for indx2, val in enumerate(X[indx, :]):
        if indx2%2 == 0:
            x.append(val)
        else:
            y.append(val)
    
    if gender == 0:
        ax1.plot(x, y, 'ro')
    else:
        ax2.plot(x, y, 'bo')

plt.show()