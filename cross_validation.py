import numpy as np
from kernels import RBF, Linear, Polynomial, Sigmoid
from svc import KernelSVC
from krr import KernelRidgeRegression, WeightedKernelRidgeRegression
from log_reg import KernelLogisticRegression
from losses import f1_score_m1_1
from dataloader import load_data


def k_fold_cross_validation(X, y, kernel_type, model_class, param_grid, k=5, using_julia=False):

    def split_data(X, y, k):
        indices = np.random.permutation(len(X))
        fold_size = len(X) // k
        folds = []
        
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)
            folds.append((train_indices, test_indices))
        
        return folds
    
    def evaluate_model(X_train, y_train, X_test, y_test, kernel_type, model_class, params, using_julia=False):

        if kernel_type == "gaussian":
            kernel = RBF(sigma=params['sigma']).kernel
        elif kernel_type == "linear":
            kernel = Linear().kernel
        elif kernel_type == "Poly":
            kernel = Polynomial(degree=params['degree'], r=params['r']).kernel
        elif kernel_type == "Sigmoid":
            kernel = Sigmoid(sigma=params['sigma'], r=params['r']).kernel
        else:
            raise ValueError("Invalid kernel type")
        
        if model_class == "svc":
            model = KernelSVC(C=params['C'], kernel=kernel, epsilon=1e-14)
        elif model_class == "krr":
            model = KernelRidgeRegression(lambd=params['lambd'], kernel=kernel, epsilon=1e-14)
        elif model_class == "log_reg":
            model = KernelLogisticRegression(kernel=kernel, lambd=params['lambd'])

        else:
            raise ValueError("Invalid model class")
        
        if using_julia:
            model.fit_julia(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        return f1_score_m1_1(y_test, y_pred)
    
    folds = split_data(X, y, k)
    best_score = -np.inf
    best_params = None
    

    for params in param_grid:
        scores = []
        # start_time_params = time.time()

        for train_indices, test_indices in folds:
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            # start_time = time.time()
            score = evaluate_model(X_train, y_train, X_test, y_test, kernel_type, model_class, params, using_julia=using_julia)
            # print(f"Time for one fold: {time.time() - start_time}")
            scores.append(score)
        # print(f"Time for one set of params: {time.time() - start_time_params}")

        mean_score = np.mean(scores)
        

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score

def kfold(num_dataset, params_list, k=5):
    best_score = -np.inf
    best_params = None
    for p, params in enumerate(params_list):

        scores = []
        model_class = params['model']
        kernel_type = params['kernel']
        encoding = params['encoding']
        using_julia = params['using_julia']
        params_model = params['params']

        X_train, y_train = load_data(num_dataset, "train", encoding=encoding)

        for i in range(k):

            train_indices = np.where(np.arange(len(X_train)) % k != i)[0]
            test_indices = np.where(np.arange(len(X_train)) % k == i)[0]
            X_train_fold, X_test_fold = X_train[train_indices], X_train[test_indices]
            y_train_fold, y_test_fold = y_train[train_indices], y_train[test_indices]
            if kernel_type == "gaussian":
                kernel = RBF(sigma=params_model['sigma']).kernel
            elif kernel_type == "linear":
                kernel = Linear().kernel
            elif kernel_type == "Poly":
                kernel = Polynomial(degree=params_model['degree'], r=params_model['r']).kernel
            elif kernel_type == "Sigmoid":
                kernel = Sigmoid(sigma=params_model['sigma'], r=params_model['r']).kernel
            else:
                raise ValueError("Invalid kernel type")
            if model_class == "svc":
                model = KernelSVC(C=params_model['C'], kernel=kernel, epsilon=1e-14)
            elif model_class == "krr":
                model = KernelRidgeRegression(lambd=params_model['lambd'], kernel=kernel, epsilon=1e-14)
            elif model_class == "log_reg":
                model = KernelLogisticRegression(kernel=kernel, lambd=params_model['lambd'])
            else:
                raise ValueError("Invalid model class")
            
            if using_julia:
                model.fit_julia(X_train_fold, y_train_fold)
            else:
                model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            scores.append(f1_score_m1_1(y_test_fold, y_pred))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score