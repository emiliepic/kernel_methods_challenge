import numpy as np


class KernelRidgeRegression:
    def __init__(self, lambd, kernel, epsilon=1e-3):
        self.lambd = lambd
        self.kernel = kernel
        self.alpha = None

    def fit(self, X, y):
        # print(np.unique(y))
        K = self.kernel(X, X)
        n = X.shape[0]
        self.alpha = np.linalg.inv(K + self.lambd * np.eye(n)) @ y
        # support_indices = np.where(self.alpha > 1e-5)[0]
        self.support = X
        # self.alpha = self.alpha[support_indices]
        # self.support_y = y[support_indices]

    def predict(self, X):
        """ Predict the y values for the input data in {-1, 1}"""
        # print("alpha", self.alpha)
        # print("support", self.support)
        K = self.kernel(X, self.support)
        # print("K", K)
        f_values = K @ self.alpha
        # print("f values ", f_values)
        return 2 * (f_values > 0) - 1


class WeightedKernelRidgeRegression:
    def __init__(self, lambd, kernel, W, epsilon=1e-3):
        self.lambd = lambd
        self.kernel = kernel
        self.alpha = None
        self.W = W

    def fit(self, X, y):
        K = self.kernel(X, X)
        n = X.shape[0]
        W_sqrt = np.sqrt(self.W)
        inv = np.linalg.inv(W_sqrt @ K @ W_sqrt + self.lambd * np.eye(n))
        self.all_alpha = W_sqrt @ inv @ W_sqrt @ y
        # support_indices = np.where(self.all_alpha > 1e-5)[0]
        self.support = X #[support_indices]
        self.alpha = self.all_alpha #[support_indices]
        self.support_y = y #[support_indices]
    
    def predict(self, X):
        """ Predict the y values for the input data in {-1, 1}"""
        K = self.kernel(X, self.support)
        f_values = K @ self.alpha
        return 2 * (f_values > 0) - 1



# def solve_krr(lambd, X, y, kernel_type='linear', sigma=1.):
#     if kernel_type == 'linear':
#         kernel = Linear().kernel
#     elif kernel_type == 'rbf':
#         kernel = RBF(sigma=sigma).kernel
#     else:
#         raise ValueError("Invalid kernel type")
#     K = kernel(X, X)
#     n = X.shape[0]
#     inv = np.linalg.inv(K + lambd * np.eye(n))
#     alpha = inv @ y
#     return alpha

# def solve_wkrr(lambd, W, X, y, kernel_type='linear', sigma=1.):
#     if kernel_type == 'linear':
#         kernel = Linear().kernel
#     elif kernel_type == 'rbf':
#         kernel = RBF(sigma=sigma).kernel
#     else:
#         raise ValueError("Invalid kernel type")
#     K = kernel(X, X)
#     n = X.shape[0]
#     W_sqrt = np.sqrt(W)
#     inv = np.linalg.inv(W_sqrt @ K @ W_sqrt + lambd * np.eye(n))
#     alpha = W_sqrt @ inv @ W_sqrt @ y
#     return alpha


    