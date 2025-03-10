import numpy as np
from scipy.special import expit as sigmoid  # Fonction sigmoïde
from krr import KernelRidgeRegression, WeightedKernelRidgeRegression

class KernelLogisticRegression:
    def __init__(self, kernel, lambd, max_iter=100, tol=1e-6):
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.lambd = lambd

    def fit(self, X, y, alpha_init=None):
        self.X_train = X
        self.y_train = y
        n = len(y)
        K = self.kernel(X, X)
        if alpha_init is None:
            # Initialisation aléatoire de alpha
            # alpha = np.random.randn(n)
            alpha = np.zeros(n)
        else:
            alpha = alpha_init

        for t in range(self.max_iter):
            # Mise à jour de m_i
            # print(K.shape, alpha.shape)
            # print("Iteration", t)
            # print("K ok", K.shape)
            # print("alpha ok", alpha.shape)
            m = K @ alpha  # m = K @ alpha^t
            # print("multiplication ok", m.shape)
            
            # Calcul de P^t_i et W^t_i
            # P = -sigmoid(-y * m)  # P_i = -sigmoid(-y_i * m_i)
            W = np.diag(sigmoid(m) * sigmoid(-m))  # W_i = sigma(m_i) * sigma(-m_i)
            # print("W ok", W.shape)
            
            # Mise à jour de z^t_i
            z = m + y / sigmoid(m)  # z_i = m_i + y_i / sigma(m_i)
            # print("z ok", z.shape)
            
            # Résolution de WKRR pour mettre à jour alpha
            model = WeightedKernelRidgeRegression(lambd=self.lambd, kernel=self.kernel, W=W)
            # print("model ok")
            model.fit(X, z)
            alpha_new = model.all_alpha
            # print("alpha_new ok", alpha_new.shape)
            # alpha_new = solve_wkrr(K, W, z)
            
            # Vérification de la convergence
            if np.linalg.norm(alpha_new - alpha) < self.tol:
                print(f"Convergence atteinte à l'itération {t+1}")
                break
            
            alpha = alpha_new
        
        self.alpha = alpha
    
    def predict(self, X):
        K = self.kernel(X, self.X_train)
        m = K @ self.alpha
        return 2 * (m > 0) - 1
