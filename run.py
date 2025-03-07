import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import json

from cross_validation import k_fold_cross_validation
from dataloader import load_data
from kernels import RBF, Linear
from losses import f1_score_m1_1
from svc import KernelSVC

do_pca = False
using_julia = True
best_params_tot = {}
text_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

for model_class in ['gaussian', 'linear']:
    best_params_tot[model_class] = {}
    Y_pred_tot = []
    score_val_tot = 0
    if model_class == 'gaussian':
        params = {'C': [0.1, 1, 10], 'sigma': [0.01, 0.1, 1, 10]}
        params_grid = [{'C': C, 'sigma': sigma} for C in params['C'] for sigma in params['sigma']]
    elif model_class == 'linear':
        params = {'C': [0.1, 1, 10]}
        params_grid = [{'C': C} for C in params['C']]

    for k in range(3):
        X_numpy, Y_numpy = load_data(k, "train")
        # X_numpy = X_numpy[:200]
        # Y_numpy = Y_numpy[:200]
        X_test = load_data(k, "test")
        
        # PCA on the data
        if do_pca:
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_numpy)
            print("PCA done")
            X_te_pca = pca.transform(X_test)
        else:
            X_train_pca = X_numpy
            X_te_pca = X_test
    
        
        best_params, best_score = k_fold_cross_validation(X_train_pca, Y_numpy, model_class, params_grid, k=3, using_julia=using_julia)
        best_params_tot[model_class][k] = best_params
        print(f"Best parameters for model {model_class} on dataset {k} : {best_params}")
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_pca, Y_numpy, test_size=0.2, random_state=42)
        if model_class == 'gaussian':
            kernel = RBF(sigma=best_params['sigma']).kernel
        else:
            kernel = Linear().kernel
        model = KernelSVC(C=best_params['C'], kernel=kernel, epsilon=1e-14)

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
    df.to_csv(f'{text_date}_Yte_pca{do_pca}_{model_class}_{score_val_tot:.3f}.csv', index=False)

# save best params to json file

with open(f'{text_date}_best_params_tot.json', 'w') as f:
    json.dump(best_params_tot, f)
