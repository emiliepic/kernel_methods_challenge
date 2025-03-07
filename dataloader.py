import pandas as pd
import numpy as np

def load_data(k, type='train'):
    file_Y = open(f'data/Ytr{k}.csv', 'rb')
    
    if type == 'train':
        X_numpy = np.genfromtxt(f'data/Xtr{k}_mat100.csv', delimiter=' ')
        Y_pd = pd.read_csv(file_Y)
        Y_numpy = Y_pd['Bound'].to_numpy().ravel()
        Y_numpy[Y_numpy == 0] = -1
        return X_numpy, Y_numpy
    else:
        X_numpy = np.genfromtxt(f'data/Xte{k}_mat100.csv', delimiter=' ')
        return X_numpy
