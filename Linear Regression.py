import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Q1:
# 1.a: Transform x to vandemonde matrix:
def polynomial_transform(X, d):
    return np.vander(X, (d + 1))

# 1.b: Function to get weights:
def train_model(phi, y):
    y0 = np.array(y)
    return np.linalg.inv(phi.T @ phi) @ phi.T @ y0.T

# 1.c:
def evaluate_model(Phi, y, w):
    sum = 0;
    size = len(y)
    for i in range(len(y)):
        sum += ((y[i] - w @ Phi[i].T) ** 2)
    return (sum / size)

# 1.d: Discussion is at the end.
def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
    return y

###################################
n = 750                                 # Number of data points
X = np.random.uniform(-7.5, 7.5, n)     # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)       # Random Gaussian noise
y = f_true(X) + e                       # True labels with noise

plt.figure()
# Plot the data
plt.scatter(X, y, 12, marker='o')
# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# Validation and Test:
tst_frac = 0.3 # Fraction of examples to sample for the test set
val_frac = 0.1 # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

w = {} # Dictionary to store all the trained models
validationErr = {} # Validation error of the models
testErr = {} # Test error of all the models

for d in range(3, 25, 3): # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d) # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn) # Learn model ontraining data
    Phi_val = polynomial_transform(X_val, d) # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d]) # Evaluate model on validation data
    Phi_tst = polynomial_transform(X_tst, d) # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d]) # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

plt.figure()
plt.title('Polynomial_Basis_Model vs True')
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# My Discussion:
# From plots of Validation Error and Test Error, we can see that d = 15 is the best choice.
# Because from d = 15, both validation error and test error becomes stable and are consistent with each other.
# Furthermore, although d > 15 performs almost same and even a little better, d = 15 is the lowest degree we can
# have, which means it's more simpler than others.



# Q2:
# 2.a: Generate kernel:
def radial_basis_transform(X, B, gamma = 0.1):
    kernel = np.empty(shape=[len(X), len(B)])
    for i in range(len(X)):
        for j in range(len(B)):
            kernel[i, j] = np.exp(-gamma * ((X[i] - B[j]) ** 2))
    return kernel

# 2.b: Calculate w:
def train_ridge_model(Phi, y, lam):
    dim = len(y)
    return (np.linalg.inv(Phi.T @ Phi + lam * np.identity(dim))) @ Phi.T @ y.T

# 2.c Try every lamba:
def evaluate_model_with_penalty(Phi, y, w, lam):
    sum = 0;
    size = len(y)
    for i in range(len(y)):
        sum += ((y[i] - w @ Phi[i].T) ** 2)
    # norm = 0
    # for i in w:
        # norm += (i ** 2)
    # sum += (lam * norm)
    # return sum/2

w = {} # Dictionary to store all the trained models
validationErr = {} # Validation error of the models
testErr = {} # Test error of all the models
lam = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for l in lam:
    # Take log(lam):
    lgl = np.log10(l)

    # Train:
    Phi_trn = radial_basis_transform(X_trn, y_trn)
    w[l] = train_ridge_model(Phi_trn, y_trn, l)

    # Validation:
    Phi_val = radial_basis_transform(X_val, y_trn)
    validationErr[l] = evaluate_model_with_penalty(Phi_val, y_val, w[l], l)

    # Test:
    Phi_tst = radial_basis_transform(X_tst, y_trn)
    testErr[l] = evaluate_model_with_penalty(Phi_tst, y_tst, w[l], l)

# Plot:
plt.figure()
plt.title('ValidationErr and TestErr with Penalty')
loglam = np.log10(list(validationErr.keys()))
plt.plot(loglam, list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(loglam, list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Log of Lamba Chosen', fontsize=16)
plt.ylabel('Validation/Test error with penalty', fontsize=16)
plt.xticks(loglam, fontsize=12)
plt.legend(['Validation Error with penalty', 'Test Error with penalty'], fontsize=16)

# 2.d: Plot true versus model:
plt.figure()
plt.title('Radial_Basis_Model vs True')
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

Phi_tru = radial_basis_transform(x_true, y_true)
for l in lam:
    w[l] = train_ridge_model(Phi_tru, y_true, l)
    y_l = Phi_tru @ w[l]
    plt.plot(x_true, y_l, marker='None', linewidth=2)

# Plot on screen:
plt.show()

# As we can see, linearity increases with lammba increases.
