import datetime
import numpy as np
import pandas as pd
from dataloader import load_data
from kernels import RBF, Linear, Polynomial, Sigmoid
from losses import f1_score_m1_1
from svc import KernelSVC
from krr import KernelRidgeRegression, WeightedKernelRidgeRegression
from log_reg import KernelLogisticRegression


# dict_best_params = {0: {"model": "krr", "kernel": "Poly", "encoding": "one_hot", "using_julia": False,
#                           "params": {"lambd": 0.1, "sigma": 0.01, "r": 0, "degree": 4}},
#                     1: {"model": "krr", "kernel": "gaussian", "encoding": "mat100", "using_julia": False,
#                           "params": {"lambd": 0.1, "sigma": 0.1}},
#                     2: {"model": "krr", "kernel": "gaussian", "encoding": "mat100", "using_julia": False,
#                           "params": {"lambd": 0.1, "sigma": 0.01}}}

dict_best_params = {0: {'model': 'svc', 'kernel': 'Poly', 'encoding': 'one_hot', 'using_julia': True,
                        'params': {'C': 0.1, 'degree': 4, 'r': 0}},
                    1: {'model': 'svc', 'kernel': 'gaussian', 'encoding': 'mat100', 'using_julia': True,
                        'params': {'C': 10, 'sigma': 0.1}},
                    2: {'model': 'svc', 'kernel': 'gaussian', 'encoding': 'mat100', 'using_julia': True,
                        'params': {'C': 10, 'sigma': 0.1}}}


best_params_tot = {}
Y_pred_tot = []
train_loss_tot = 0
for num_dataset in range(3):
    params = dict_best_params[num_dataset]
    model_class = params['model']
    kernel_type = params['kernel']
    encoding = params['encoding']
    using_julia = params['using_julia']
    params_model = params['params']

    X_train, Y_train = load_data(num_dataset, "train", encoding=encoding)
    X_test = load_data(num_dataset, "test", encoding=encoding)

    if kernel_type == 'gaussian':
        kernel = RBF(sigma=params_model['sigma']).kernel
    elif kernel_type == "linear":
        kernel = Linear().kernel
    elif kernel_type == "Poly":
        kernel = Polynomial(degree=params_model['degree'], r=params_model['r']).kernel
    elif kernel_type == "Sigmoid":
        kernel = Sigmoid(sigma=params_model['sigma'], r=params_model['r']).kernel

    if model_class == "svc":
        model = KernelSVC(C=params_model['C'], kernel=kernel, epsilon=1e-14)
    elif model_class == "krr":
        model = KernelRidgeRegression(lambd=params_model['lambd'], kernel=kernel, epsilon=1e-14)
    elif model_class == "log_reg":
        model = KernelLogisticRegression(kernel=kernel, lambd=params_model['lambd'])

    if using_julia:
        model.fit_julia(X_train, Y_train)
    else:
        model.fit(X_train, Y_train)
    
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    Y_pred_tot.append(Y_pred_test)
    train_loss = f1_score_m1_1(Y_train, Y_pred_train)  
    train_loss_tot += train_loss  

Y_pred_tot = np.array(Y_pred_tot)
Y_pred_tot[Y_pred_tot == -1] = 0
Y_pred_tot = Y_pred_tot.ravel()
train_loss_tot /= 3
print(f'Train loss : {train_loss_tot}')

# Save the predictions
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M")

# save to csv file with header Id,Bound
df = pd.DataFrame({'Id': np.arange(len(Y_pred_tot)), 'Bound': Y_pred_tot})

df.to_csv(f'results/{now}_Yte_{train_loss_tot:.3f}.csv', index=False)

        