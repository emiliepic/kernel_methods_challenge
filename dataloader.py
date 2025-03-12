import pandas as pd
import numpy as np
import os



def encode_data(k, type='train',encoding="one_hot"):
    if type == 'train':
        file_path = f'data/Xtr{k}.csv'
    elif type == 'test':
        file_path = f'data/Xte{k}.csv'
    else:
        raise ValueError("Invalid type")
    df = pd.read_csv(file_path)
    X = df['seq'].to_numpy()
    if encoding == 'one_hot':
        X_encoded = np.array([encode_sequence_one_hot(seq) for seq in X])
    elif encoding == 'int':
        X_encoded = np.array([encode_sequence_int(seq) for seq in X])
    elif encoding == 'bin':
        X_encoded = np.array([encode_sequence_bin(seq) for seq in X])
    else:
        raise ValueError("Invalid encoding")
    return X_encoded

def encode_sequence_one_hot(sequence):
    """ Encode a DNA sequence into a flatten matrix """
    encoding = np.zeros((len(sequence), 4))
    for i, letter in enumerate(sequence):
        if letter == 'A':
            encoding[i, 0] = 1
        elif letter == 'C':
            encoding[i, 1] = 1
        elif letter == 'G':
            encoding[i, 2] = 1
        elif letter == 'T':
            encoding[i, 3] = 1
        else:
            raise ValueError("Invalid letter in sequence")
    return encoding.flatten()

def encode_sequence_int(sequence):
    encoding = np.zeros(len(sequence))
    for i, letter in enumerate(sequence):
        if letter == 'A':
            encoding[i] = 0
        elif letter == 'C':
            encoding[i] = 1
        elif letter == 'G':
            encoding[i] = 2
        elif letter == 'T':
            encoding[i] = 3
        else:
            raise ValueError("Invalid letter in sequence")
    return encoding

def encode_sequence_bin(sequence):
    encoding = np.zeros((len(sequence),2))
    for i, letter in enumerate(sequence):
        if letter == 'A':
            encoding[i, 0] = 0
            encoding[i, 1] = 0
        elif letter == 'C':
            encoding[i, 0] = 0
            encoding[i, 1] = 1
        elif letter == 'G':
            encoding[i, 0] = 1
            encoding[i, 1] = 0
        elif letter == 'T':
            encoding[i, 0] = 1
            encoding[i, 1] = 1
        else:
            raise ValueError("Invalid letter in sequence")
    return encoding.flatten()


def save_encoded_data():
    for k in range(3):
        for type in ['train', 'test']:
            for encoding in ['one_hot', 'int', 'bin']:
                if type == 'train':
                    path = f'data/Xtr{k}_{encoding}.csv'
                elif type == 'test':
                    path = f'data/Xte{k}_{encoding}.csv'
                else:
                    raise ValueError("Invalid type")
                X = encode_data(k, type, encoding)
                np.savetxt(path, X, delimiter=' ')


def load_data(k, type='train', encoding='mat100'):
    file_Y = open(f'data/Ytr{k}.csv', 'rb')
    
    if type == 'train':
        file_path = f'data/Xtr{k}_{encoding}.csv'
        if not os.path.exists(file_path):
            save_encoded_data()
        X_numpy = np.genfromtxt(file_path, delimiter=' ')
        Y_pd = pd.read_csv(file_Y)
        Y_numpy = Y_pd['Bound'].to_numpy().ravel()
        Y_numpy[Y_numpy == 0] = -1
        return X_numpy, Y_numpy
    else:
        file_path = f'data/Xte{k}_{encoding}.csv'
        if not os.path.exists(file_path):
            save_encoded_data()
        X_numpy = np.genfromtxt(file_path, delimiter=' ')
        return X_numpy
