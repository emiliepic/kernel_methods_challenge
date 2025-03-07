import numpy as np
from kernels import RBF, Linear
from svc import KernelSVC
from losses import f1_score_m1_1
import time


def k_fold_cross_validation(X, y, model_class, param_grid, k=5, using_julia=False):
    """
    Effectue une validation croisée K-Fold pour déterminer les meilleurs paramètres pour un modèle.
    
    X : array-like, données d'entrée
    y : array-like, étiquettes de sortie
    model_class : classe du modèle (par exemple, LogisticRegression, SVC)
    param_grid : dictionnaire contenant les hyperparamètres à tester
    k : int, nombre de folds (par défaut 5)
    scoring : str, la métrique à utiliser ('accuracy', 'mse', etc.)
    
    Retourne le meilleur ensemble de paramètres et la meilleure performance.
    """
    
    def split_data(X, y, k):
        """ Fonction pour diviser les données en k folds """
        indices = np.random.permutation(len(X))
        fold_size = len(X) // k
        folds = []
        
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)
            folds.append((train_indices, test_indices))
        
        return folds
    
    def evaluate_model(X_train, y_train, X_test, y_test, model_class, params, using_julia=False):
        """ Fonction pour entraîner un modèle et évaluer sa performance """
        if model_class == "gaussian":
            kernel = RBF(sigma=params['sigma']).kernel
        elif model_class == "linear":
            kernel = Linear().kernel
        else:
            raise ValueError("Invalid model class")
        model = KernelSVC(C=params['C'], kernel=kernel, epsilon=1e-14)
        if using_julia:
            model.fit_julia(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        return f1_score_m1_1(y_test, y_pred)
    
    folds = split_data(X, y, k)
    best_score = -np.inf
    best_params = None
    
    # Boucle à travers les combinaisons de paramètres
    for params in param_grid:
        scores = []
        start_time_params = time.time()
        # Pour chaque fold, on effectue un entraînement et une validation
        for train_indices, test_indices in folds:
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            start_time = time.time()
            score = evaluate_model(X_train, y_train, X_test, y_test, model_class, params, using_julia=using_julia)
            print(f"Time for one fold: {time.time() - start_time}")
            scores.append(score)
        print(f"Time for one set of params: {time.time() - start_time_params}")
        # Calculer la performance moyenne pour ces paramètres
        mean_score = np.mean(scores)
        
        # Si la performance est meilleure, on met à jour les meilleurs paramètres
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score