# import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

labels = [[0, 0],
          [1, 1],
          [2, 2],
          [3, 6],
          [4, 4],
          [5, 5],
          [6, 6],
          [7, 9],
          [8, 8],
          [9, 12],
          [10, 10],
          [11, 11],
          [12, 12],
          [13, 17],
          [14, 14],
          [15, 15],
          [16, 16],
          [17, 17]]
targets = [0, 1, 2, 3, 4, 5, 6, 7, 7, 9, 9, 11, 11, 13, 15, 15, 16, 17]

plt.clf()
plt.plot(labels, targets, 'b.', markersize=6)
plt.plot(labels, y_, 'g.-', markersize=6)

reg = linear_model.LinearRegression()
print(reg.fit(labels, targets))

X_test = [[0, 1], [1, 2], [2, 2]]
predictions = reg.predict(X_test)

print(reg.coef_)