import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    # Generate a non-linear data set
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30)  # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m,))
    y[ind] = 1 - y[ind]

    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state = 42)

    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)


def visualize(models, param, X, y):
    # Initialize plotting
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])

    # Create a mesh
    xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
    yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), np.arange(yMin, yMax, 0.01))

    for i, (p, clf) in enumerate(models.items()):
        # if i > 0:
        # break
        r, c = np.divmod(i, 3)
        ax = axes[r, c]

        # Plot contours
        zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)
        if (param == 'C' and p > 0.0) or (param == 'gamma'):
            ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        # Plot data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('{0} = {1}'.format(param, p))


# Generate the data
n_samples = 300  # Total size of data set
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

# Training:
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)
models = dict()
trnErr = dict()
valErr = dict()
for G in gamma_values:
    # Fitting:
    clf = SVC(10, gamma=G, kernel='rbf')
    clf.fit((X_trn, y_trn)[0], (X_trn, y_trn)[1])

    # Analyzing:
    pred_val = clf.predict((X_val, y_val)[0])
    wrong = 0
    for i in range(len(pred_val)):
        if pred_val[i] != (X_val, y_val)[1][i]:
            wrong += 1
    valErr[G] = wrong / len(pred_val)
    pred_trn = clf.predict((X_trn, y_trn)[0])
    wrong = 0
    for i in range(len(pred_trn)):
        if pred_trn[i] != (X_trn, y_trn)[1][i]:
            wrong += 1
    trnErr[G] = wrong / len(pred_trn)

print(trnErr)
print(valErr)


# Plot:
cs = np.array([])
trn_err = np.array([])
val_err = np.array([])
for G in gamma_values:
    tmp = trnErr[G]
    trn_err = np.append(trn_err, tmp)
    tmp = valErr[G]
    val_err = np.append(val_err, tmp)

fig = plt.figure()
plt.title('Test/Validation Error vs gamma')
plt.plot(gamma_range, trn_err, label='TrnErr', marker='o')
plt.plot(gamma_range, val_err, label='ValErr', marker='o')
plt.xlabel('Log(gamma)')
plt.ylabel('Errors')
plt.legend()
plt.show()

# Testing:
# Choose gamma = 10
clf = SVC(10, gamma=10, kernel='rbf')
clf.fit((X_trn, y_trn)[0], (X_trn, y_trn)[1])
tstErr = 0
pred_tst = clf.predict((X_tst, y_tst)[0])
wrong = 0
for i in range(len(pred_tst)):
    if pred_tst[i] != (X_tst, y_tst)[1][i]:
        wrong += 1
tstErr = wrong / len(pred_tst)

print('Testing Accuracy:', 1 - tstErr)
