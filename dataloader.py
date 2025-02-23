import pandas as pd
data_path = 'data/'

def load_data(k):
    X_train = pd.read_csv(data_path + f'Xtr{k}.csv')
    X_test = pd.read_csv(data_path + f'Xte{k}.csv')
    y_train = pd.read_csv(data_path + f'Ytr{k}.csv')
    return X_train, y_train, X_test

