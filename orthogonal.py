import numpy as np



U = 1/3 * np.array([
    [2,-2,1], [1,2,2],[2,1,-2]
])


Xp = np.array([
    [1, 2, 3],
    [2, 4, 5]
])

Xm = np.array([
    [1, 1, 2],
    [0, 2, 4]
])

Um = np.array([
    [1, 0.5, 2]
])
A = np.array([[0.7, 0.1], [0.2, 0.4]])
B = np.array([[0.5], [0.1]])


D = Xp - A @ Xm - B  @ Um
DU = D @ U
import pdb
pdb.set_trace()