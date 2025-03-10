import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import json
import itertools

from cross_validation import k_fold_cross_validation
from dataloader import load_data
from kernels import RBF, Linear, Polynomial, Sigmoid
from losses import f1_score_m1_1
from svc import KernelSVC
from krr import KernelRidgeRegression, WeightedKernelRidgeRegression
from log_reg import KernelLogisticRegression

do_pca = False
using_julia = True
best_params_tot = {}
text_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
model_class = "svc"
encoding = "int"

kernel_params = {
    'gaussian': {'sigma': [0.01, 0.1, 1, 10]},
    'linear': {},
    'Poly': {'degree': [2, 3, 4], 'r': [0, 1, 2]},
    'Sigmoid': {'sigma': [0.01, 0.1, 1, 10], 'r': [0, 1, 2]}
}

param_map = {
    "svc": {'C': [0.1, 1, 10]},
    "krr": {'lambd': [0.1, 1, 10]},
    "wkrr": {'lambd': [0.1, 1, 10], 'W': [0.1, 2, 10]},
    "log_reg": {'lambd': [0.1, 1, 10]}
}

for kernel_type in ['linear', 'gaussian', 'Poly', 'Sigmoid']:  # 'linear', 'gaussian', 'Poly', 'Sigmoid'
    best_params_tot[kernel_type] = {}
    Y_pred_tot = []
    score_val_tot = 0

    # Set model-specific params
    params = param_map.get(model_class)
    if not params:
        raise ValueError("Invalid model class")

    # Set kernel-specific params
    kernel_params_for_type = kernel_params.get(kernel_type)
    if kernel_params_for_type:
        params.update(kernel_params_for_type)
    
    # Generate parameter grid
    param_grid_list = []
    for param, values in params.items():
        param_grid_list.append(values)
        
    params_grid = [dict(zip(params.keys(), comb)) for comb in itertools.product(*param_grid_list)]


    for k in range(3):
        X_numpy, Y_numpy = load_data(k, "train", encoding=encoding)
        # X_numpy = X_numpy[:200]
        # Y_numpy = Y_numpy[:200]
        X_test = load_data(k, "test", encoding=encoding)
        
        # PCA on the data
        if do_pca:
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_numpy)
            print("PCA done")
            X_te_pca = pca.transform(X_test)
        else:
            X_train_pca = X_numpy
            X_te_pca = X_test
    
        
        best_params, best_score = k_fold_cross_validation(X_train_pca, Y_numpy, kernel_type, model_class, params_grid, k=3, using_julia=using_julia)
        best_params_tot[kernel_type][k] = best_params
        print(f"Best parameters for model {kernel_type} on dataset {k} : {best_params}")
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_pca, Y_numpy, test_size=0.2, random_state=42)
        if kernel_type == 'gaussian':
            kernel = RBF(sigma=best_params['sigma']).kernel
        elif kernel_type == "linear":
            kernel = Linear().kernel
        elif kernel_type == "Poly":
            kernel = Polynomial(degree=best_params['degree'], r=best_params['r']).kernel
        elif kernel_type == "Sigmoid":
            kernel = Sigmoid(sigma=best_params['sigma'], r=best_params['r']).kernel
        
        if model_class == "svc":
            model = KernelSVC(C=best_params['C'], kernel=kernel, epsilon=1e-14)
        elif model_class == "krr":
            model = KernelRidgeRegression(lambd=best_params['lambd'], kernel=kernel, epsilon=1e-14)
        elif model_class == "wkrr":
            n = len(Y_train)
            W_matrix = np.diag([best_params['W'] for _ in range(n)])
            model = WeightedKernelRidgeRegression(lambd=best_params['lambd'], kernel=kernel, W=W_matrix, epsilon=1e-14)
        elif model_class == "log_reg":
            model = KernelLogisticRegression(kernel=kernel, lambd=best_params['lambd'])


        print("Model created, fitting started")
        if using_julia:
            model.fit_julia(X_train, Y_train)
        else:
            model.fit(X_train, Y_train)
        print("Model fitted")
        Y_pred_train = model.predict(X_train)
        Y_pred_val = model.predict(X_val)
        Y_pred_test = model.predict(X_te_pca)
        Y_pred_tot.append(Y_pred_test)
        print(f'F1 score for fold {k} on train is {f1_score_m1_1(Y_train, Y_pred_train)}')
        print(f'F1 score for fold {k} on val is {f1_score_m1_1(Y_val, Y_pred_val)}')
        score_val_tot += f1_score_m1_1(Y_val, Y_pred_val)

    print(f'Average F1 score on validation set : {score_val_tot/3}')
    score_val_tot /= 3
    Y_pred_tot = np.array(Y_pred_tot)
    # transformer les -1 en 0
    Y_pred_tot[Y_pred_tot == -1] = 0
    Y_pred_tot = Y_pred_tot.ravel()
    # save to csv file with header Id,Bound
    df = pd.DataFrame({'Id': np.arange(len(Y_pred_tot)), 'Bound': Y_pred_tot})
    # text_params = '_'.join([f'{key}_{value}' for key, value in best_params.items()])
    df.to_csv(f'results/{text_date}_Yte_{encoding}_{model_class}_pca{do_pca}_{kernel_type}_{score_val_tot:.3f}.csv', index=False)

# save best params to json file

with open(f'results/{text_date}_best_params_tot.json', 'w') as f:
    json.dump(best_params_tot, f)


# for kernel_type in ['linear']: #'gaussian', 'Poly', 'Sigmoid'
#     best_params_tot[kernel_type] = {}
#     Y_pred_tot = []
#     score_val_tot = 0
#     if model_class == "svc":
#         params = {'C': [0.1, 1, 10]}
#     elif model_class == "krr":
#         params = {'lambd': [0.1, 1, 10]}
#     elif model_class == "wkrr":
#         params = {'lambd': [0.1, 1, 10], 'W': [0.1, 2, 10]}
#     elif model_class == "log_reg":
#         params = {'lambd': [0.1, 1, 10]}
#     else:
#         raise ValueError("Invalid model class")
    
#     if kernel_type == 'gaussian':
#         params['sigma'] = [0.01, 0.1, 1, 10]
#         if model_class == "svc":
#             params_grid = [{'C': C, 'sigma': sigma} for C in params['C'] for sigma in params['sigma']]
#         elif model_class == "krr":
#             params_grid = [{'lambd': lambd, 'sigma': sigma} for lambd in params['lambd'] for sigma in params['sigma']]
#         elif model_class == "wkrr":
#             params_grid = [{'lambd': lambd, 'sigma': sigma, 'W': W} for lambd in params['lambd'] for sigma in params['sigma'] for W in params['W']]
#         elif model_class == "log_reg":
#             params_grid = [{'lambd': lambd, 'sigma': sigma} for lambd in params['lambd'] for sigma in params['sigma']]

#     elif kernel_type == 'linear':
#         if model_class == "svc":
#             params_grid = [{'C': C} for C in params['C']]
#         elif model_class == "krr":
#             params_grid = [{'lambd': lambd} for lambd in params['lambd']]
#         elif model_class == "wkrr":
#             params_grid = [{'lambd': lambd, 'W': W} for lambd in params['lambd'] for W in params['W']]
#         elif model_class == "log_reg":
#             params_grid = [{'lambd': lambd} for lambd in params['lambd']]

#     elif kernel_type == 'Poly':
#         params['degree'] = [2, 3, 4]
#         params['r'] = [0, 1, 2]
#         if model_class == "svc":
#             params_grid = [{'C': C, 'degree': degree, 'r': r} for C in params['C'] for degree in params['degree'] for r in params['r']]
#         elif model_class == "krr":
#             params_grid = [{'lambd': lambd, 'degree': degree, 'r': r} for lambd in params['lambd'] for degree in params['degree'] for r in params['r']]
#         elif model_class == "wkrr":
#             params_grid = [{'lambd': lambd, 'degree': degree, 'W': W, 'r': r} for lambd in params['lambd'] for degree in params['degree'] for W in params['W'] for r in params['r']]
#         elif model_class == "log_reg":
#             params_grid = [{'lambd': lambd, 'degree': degree, 'r': r} for lambd in params['lambd'] for degree in params['degree'] for r in params['r']]

#     elif kernel_type == "Sigmoid":
#         params['sigma'] = [0.01, 0.1, 1, 10]
#         params['r'] = [0, 1, 2]
#         if model_class == "svc":
#             params_grid = [{'C': C, 'sigma': sigma, 'r': r} for C in params['C'] for sigma in params['sigma'] for r in params['r']]
#         elif model_class == "krr":
#             params_grid = [{'lambd': lambd, 'sigma': sigma, 'r': r} for lambd in params['lambd'] for sigma in params['sigma'] for r in params['r']]
#         elif model_class == "wkrr":
#             params_grid = [{'lambd': lambd, 'sigma': sigma, 'W': W, 'r': r} for lambd in params['lambd'] for sigma in params['sigma'] for W in params['W'] for r in params['r']]
#         elif model_class == "log_reg":
#             params_grid = [{'lambd': lambd, 'sigma': sigma, 'r': r} for lambd in params['lambd'] for sigma in params['sigma'] for r in params['r']]

#     else:
#         raise ValueError("Invalid kernel type")

