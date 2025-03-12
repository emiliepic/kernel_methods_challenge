import datetime
import numpy as np
import pandas as pd
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



kernel_params = {
    'gaussian': {'sigma': [0.01, 0.1, 1, 10]},
    'linear': {},
    'Poly': {'degree': [2, 3, 4], 'r': [0, 1, 2]},
    'Sigmoid': {'sigma': [0.01, 0.1, 1, 10], 'r': [0, 1, 2]}
}

param_map = {
    "svc": {'C': [0.1, 1, 10]},
    "krr": {'lambd': [0.1, 1, 10]},
    "log_reg": {'lambd': [0.1, 1, 10]}
}



for model_class in ['svc', 'krr', 'log_reg']:  # 'svc', 'krr', 'log_reg'
    for encoding in ['bin' , 'int', 'one_hot', 'mat100']: # 'bin' , 'int', 'one_hot'
        best_params_tot = {}
        text_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        if model_class == "svc":
            using_julia = True
        for kernel_type in ['Sigmoid','linear', 'gaussian', 'Poly']:  # 'linear', 'gaussian', 'Poly', 'Sigmoid'
            best_params_tot[kernel_type] = {}
            Y_pred_tot = []
            score_val_tot = 0

            # Generate parameter grid
            params = param_map.get(model_class).copy()
            if not params:
                raise ValueError("Invalid model class")
            kernel_params_for_type = kernel_params.get(kernel_type).copy()
            if kernel_params_for_type:
                params.update(kernel_params_for_type)
            param_grid_list = []
            for param, values in params.items():
                param_grid_list.append(values) 
            params_grid = [dict(zip(params.keys(), comb)) for comb in itertools.product(*param_grid_list)]


            for k in range(3):
                X_train, Y_train = load_data(k, "train", encoding=encoding)
                
                X_test = load_data(k, "test", encoding=encoding)   
                
                best_params, best_score = k_fold_cross_validation(X_train, Y_train, kernel_type, model_class, params_grid, k=3, using_julia=using_julia)
                best_params_tot[kernel_type]["score"] = best_score
                best_params_tot[kernel_type][k] = best_params
                print(f"Best parameters for model {kernel_type} on dataset {k} : {best_params}")
                
                X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
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
                Y_pred_test = model.predict(X_test)
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

            df.to_csv(f'results/{text_date}_Yte_{encoding}_{model_class}_{kernel_type}_{score_val_tot:.3f}.csv', index=False)

        # save best params to json file
        with open(f'results/{text_date}_best_params_tot.json', 'w') as f:
            json.dump(best_params_tot, f)

