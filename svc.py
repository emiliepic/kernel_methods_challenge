import numpy as np
from scipy import optimize
import subprocess
import json

# Chemin vers le fichier Julia
julia_script_path = "solve_quad.jl"


class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X, X)

        # Lagrange dual problem
        def loss(alpha):
            first_term = np.sum(alpha)
    
            y_prod = y[:, np.newaxis] * y[np.newaxis, :]
            alpha_prod = alpha[:, np.newaxis] * alpha[np.newaxis, :]
            
            # Compute the double sum
            second_term = np.sum(y_prod * alpha_prod * K)
            
            return 0.5 * second_term - first_term #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            y_alpha_product = y * alpha
            sc_prod =  K @ y_alpha_product


            return y * sc_prod - 1 # '''----------------partial derivative of the dual loss wrt alpha -----------------'''


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: np.dot(alpha, y) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha: y  #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: np.concatenate([self.C - alpha, alpha])  # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha: np.concatenate([-np.eye(N), np.eye(N)])  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        support_indices = np.where(self.alpha > self.epsilon)[0]
        self.support = X[support_indices]
        self.support_labels = y[support_indices]
        self.alpha_support_vectors = self.alpha[support_indices]
        
        self.b = (np.sum(self.support_labels)
                  - np.sum(self.alpha_support_vectors * self.support_labels
                           * self.kernel(self.support, self.support))) / len(self.support)

        self.norm_f = np.sum(self.alpha_support_vectors
                             * self.support_labels * self.alpha_support_vectors)

    def fit_julia(self, X, y):

        n = len(y)
        K = self.kernel(X, X)
        # save to json file
        dict_params = {"n": n, "K": K.tolist(), "y": y.tolist(), "C": self.C}
        json_path = "params.json"
        with open(json_path, "w") as f:
            json.dump(dict_params, f)
        alpha_path = "alpha.csv"
        

        # Exécution du script Julia avec des arguments
        try:
            result = subprocess.run(
                ["julia", julia_script_path, json_path, alpha_path],
                check=True, capture_output=True, text=True
            )
            print("Sortie du script Julia :")
            print(result.stdout)
            print("Erreurs du script Julia :")
            print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script Julia : {e}")
            print("Sortie d'erreur :")
            print(e.stderr)  # Affiche les erreurs du script Julia



        # Load the alpha values without the header
        self.alpha = np.genfromtxt(alpha_path, delimiter=",", skip_header=1)
        self.support = X
        self.support_labels = y
        self.alpha_support_vectors = self.alpha
        
        self.b = (np.sum(self.support_labels)
                  - np.sum(self.alpha_support_vectors * self.support_labels
                           * self.kernel(self.support, self.support))) / len(self.support)

        self.norm_f = np.sum(self.alpha_support_vectors
                             * self.support_labels * self.alpha_support_vectors)


    ### Implementation of the separting function $f$ 
    def separating_function(self, X):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N

        kernel_values = self.kernel(X, self.support)
        f_values = np.sum(self.alpha_support_vectors * self.support_labels * kernel_values, axis=1)

        return f_values
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1