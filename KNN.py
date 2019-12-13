import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

'''
    Self defined function, used to calculate prediction errors.
    Arg: 
        1. p => predict value from the model trained.
        2. y => true value from original data.
    Return: error
'''


def get_err(p, y):
    # Count all errors:
    count = 0
    for k in range(len(p)):
        if p[k] != y[k]:
            count += 1
    return count / len(p)


# Import data:
trn_data = np.loadtxt(
    'wdbc_trn.csv', delimiter=',')
val_data = np.loadtxt(
    'wdbc_val.csv', delimiter=',')
tst_data = np.loadtxt(
    'wdbc_tst.csv', delimiter=',')

# Prepare data;
y_trn = trn_data[:, 0]
y_val = val_data[:, 0]
y_tst = tst_data[:, 0]
X_trn = trn_data[:, 1:]
X_val = val_data[:, 1:]
X_tst = tst_data[:, 1:]

# Candidate parameters:
ks = np.array([1, 5, 11, 15, 21])

# Training, predicating:
trn_err = np.array([])
val_err = np.array([])
for k in ks:
    # Train:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_trn, y_trn)

    # Predicate train and its error:
    pred_trn = neigh.predict(X_trn)
    trn_err = np.append(trn_err, get_err(pred_trn, y_trn))

    # predicate validation and its error:
    pred_val = neigh.predict(X_val)
    val_err = np.append(val_err, get_err(pred_val, y_val))

fig = plt.figure()
plt.title('Test/Validation Error vs K for KNN')
plt.plot(ks, trn_err, label='TrnErr', marker='o')
plt.plot(ks, val_err, label='ValErr', marker='o')
plt.xlabel('K')
plt.ylabel('Errors')
plt.legend()
plt.show()

# Choose K = 5 to do test:
neigh = KNeighborsClassifier(n_neighbors=5)
neigh = neigh.fit(X_trn, y_trn)
pred_tst = neigh.predict(X_tst)
tst_err = get_err(pred_tst, y_tst)

print('Test Accuracy (K = 5):', 1 - tst_err)