import numpy as np
def problem_1a (A, B):
    return A + B
def problem_1b (A, B, C):
    return (A @ B) -C
def problem_1c (A, B, C):
    return (A * B) + np.transpose(C)
def problem_1d (x, y):
    return np.transpose(x) * y
def problem_1e (A, x):
    return np.linalg.solve(A,x)
def problem_1f (A, x):
    return np.linalg.solve(A.transpose(), x.transpose()).transpose()
def problem_1g (A, i):
    return np.sum(A, axis = 1, where = (np.arange(len(A)) % 2) >0)[i]
def problem_1h (A, c, d):
    return np.mean(A[np.nonzero(c<=A)][np.nonzero(A[np.nonzero(c<=A)]<=d)])
def problem_1i (A, k):
    w, v = np.linalg.eig(A)
    w = w[np.argsort(w)]
    v = v[:,np.argsort(w)]
    m=np.arange(len(A)-k,k+1,1)
    X = v[:,m]
    return X
def problem_1j (x, k, m, s):
    return np.transpose(np.random.multivariate_normal(x+m*z,s*np.identity(len(x)),k))
def problem_1k (A):
    return np.random.shuffle(A)
def problem_1l (x):
    return ((x - (np.mean(x))) / np.std(x))
def problem_1m (x, k):
    return np.repeat(A[: , np.newaxis], k, axis =1)
def problem_1n (X):
    a = np.repeat(X[np.newaxis,:], len(X), 0)
    b = x.transpose()
    squ=np.square(a-b)
    sqr=np.sqrt(squ)
    return np.sum(sqr, axis=0)

import numpy as np
def linear_regression (X_tr, y_tr): 
    X = np.transpose(X_tr)
    return np.linalg.solve(X @ np.transpose(X), X @ y_tr)
                          
def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")
    print(len(X_tr))

    w = linear_regression(X_tr, ytr)
    print((0.5 / len(X_tr)) * np.sum((X_tr @ w - ytr) ** 2))
    print((0.5 / len(X_te)) * np.sum((X_te @ w - yte) ** 2))
