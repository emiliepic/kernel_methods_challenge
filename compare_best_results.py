import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import json
import itertools
from cross_validation import kfold




params_set = {
    0 : [],
    1 : [],
    2 : []
}

#### CHANGE WITH THE CORRECT PATHS ####

#log_reg
# bin
path_log_reg_bin = "results/2025-03-11_01-27_best_params_tot.json"
# int
path_log_reg_int = "results/2025-03-11_01-59_best_params_tot.json"
# one_hot
path_log_reg_one_hot = "results/2025-03-11_02-57_best_params_tot.json"
# mat100
path_log_reg_mat100 = "results/2025-03-11_03-40_best_params_tot.json"

# krr
# bin
path_krr_bin = "results/2025-03-11_09-41_best_params_tot.json"
# int
path_krr_int = "results/2025-03-11_10-01_best_params_tot.json"
# one_hot
path_krr_one_hot = "results/2025-03-11_10-22_best_params_tot.json"
# mat100
path_krr_mat100 = "results/2025-03-11_14-20_best_params_tot.json"

# svc
# bin
path_svc_bin = "results/2025-03-11_18-42_best_params_tot.json"
# int
path_svc_int = "results/2025-03-11_20-40_best_params_tot.json"
# one_hot
path_svc_one_hot = "results/2025-03-11_22-31_best_params_tot.json"
# mat100
path_svc_mat100 = "results/2025-03-12_00-42_best_params_tot.json"


json_paths_log_reg = {'int': path_log_reg_int, 'bin': path_log_reg_bin, 'one_hot': path_log_reg_one_hot, 'mat100': path_log_reg_mat100}
json_paths_krr = {'int': path_krr_int, 'bin': path_krr_bin, 'one_hot': path_krr_one_hot, 'mat100': path_krr_mat100}
json_paths_svc = {'int': path_svc_int, 'bin': path_svc_bin, 'one_hot': path_svc_one_hot, 'mat100': path_svc_mat100}

json_paths = {'log_reg': json_paths_log_reg, 'krr': json_paths_krr, 'svc': json_paths_svc}

for model_class, paths in json_paths.items():
    for key, value in json_paths.items():
        with open(value, 'r') as f:
            dict_params = json.load(f)
        for k in range(3):
            for kernel_type in ['Sigmoid', 'linear', 'gaussian', 'Poly']:
                print(f"Best parameters for model {kernel_type} on dataset {k} : {dict_params[kernel_type][str(k)]}")
                params_set[k].append({"model": model_class, "kernel": kernel_type, "encoding": key, "using_julia": False, "params": dict_params[kernel_type][str(k)]})
    

best_params_tot = {}

for num_dataset in range(3):
    list_params = params_set[num_dataset]
    best_params, best_score = kfold(num_dataset, list_params, k=5)
    best_params_tot[num_dataset] = best_params

print(best_params_tot)

with open(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_best_params_fin.json", 'w') as f:
    json.dump(best_params_tot, f)
        