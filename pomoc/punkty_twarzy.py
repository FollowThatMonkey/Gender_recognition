import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = np.load('dane/pure_landmarks_gender.npy')
X, y = data[:, :-1], data[:, -1]

colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title('Bez preprocessingu')
ax2.set_title('Po preprocessingu - preprocessing.Normalizer()')

for row, val in enumerate(y):
    x0, y0 = X[row, 0], X[row, 1]
    
    for indx, val in enumerate(X[row, :]):
        if indx%2 == 0:
            X[row, indx] -= x0
        else:
            X[row, indx] -= y0

X_pre = preprocessing.Normalizer().fit_transform(X)
for row, val in enumerate(y[8655:8657]):
    xx_bez, yy_bez = [], []
    xx_po, yy_po = [], []

    for indx, val in enumerate(X[row, :]):
        if indx%2 == 0:
            xx_bez.append(X[row, indx])
            xx_po.append(X_pre[row, indx])
        else:
            yy_bez.append(X[row, indx])
            yy_po.append(X_pre[row, indx])

    ax1.plot(xx_bez, yy_bez, colors[row%len(colors)])
    ax2.plot(xx_po, yy_po, colors[row%len(colors)])

    for indx, val in enumerate(xx_bez):
        if indx == 105 or indx == 72:
            ax1.annotate(str(indx), (xx_bez[indx], yy_bez[indx]), c='r')
            ax2.annotate(str(indx), (xx_po[indx], yy_po[indx]), c='r')
        else:
            ax1.annotate(str(indx), (xx_bez[indx], yy_bez[indx]))
            ax2.annotate(str(indx), (xx_po[indx], yy_po[indx]))


plt.show()