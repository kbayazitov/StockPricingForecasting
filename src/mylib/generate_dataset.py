import numpy as np
import torch

def train_test_split(t, y, split=0.8):
    ind_split = int(split * len(y))

    t_train = t[:ind_split]
    y_train = y[:ind_split]

    t_test = t[ind_split:len(y)]
    y_test = y[ind_split:len(y)]
    
    return t_train, y_train, t_test, y_test

def windowed_dataset(y, input_window=30, output_window=10, stride=1, num_features=1):
    num_samples = (len(y) - input_window - output_window) // stride + 1

    X = np.zeros([num_samples, input_window, num_features])
    Y = np.zeros([num_samples, output_window, num_features])

    for f in range(num_features):
        for i in range(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[i, :, f] = y[start_x:end_x, f]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[i, :, f] = y[start_y:end_y, f]

    return X, Y

def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch

def get_datasets(data, iw=30, ow=1, s=1, nf=5, a=0, b=1, shift1=5, shift2=1):
    train_datasets = []
    test_datasets = []

    for method in ('base', 'noise', 'auto'):
        data[f'value_{method}_diff_1'] = data[f'value_{method}'] - data[f'value_{method}'].shift(shift1)
        data[f'value_{method}_diff_2'] = data[f'value_{method}_diff_1'] - data[f'value_{method}_diff_1'].shift(shift2)

        data['combined'] = data.apply(lambda x: list([x['open_diff_2'],
                                                      x['high_diff_2'],
                                                      x['low_diff_2'],
                                                      x['close_diff_2'],
                                                      x[f'value_{method}_diff_2'],
                                                      ]), axis=1)

        y = np.array(data['combined'].tolist())[shift1+shift2:]
        t = np.array(data['time'])[shift1+shift2:]

        if method != 'base':
            num = 5
        else:
            num = 4

        for i in range(num):
            y[:, i] = ((y[:, i] - np.min(y[:, i])) * (b - a) / (np.max(y[:, i]) - np.min(y[:, i]))) + a

        t_train, y_train, t_test, y_test = train_test_split(t, y, split=0.75)

        Xtrain, Ytrain = windowed_dataset(y_train, input_window=iw, output_window=ow, stride=s, num_features=nf)
        Xtest, Ytest = windowed_dataset(y_test, input_window=iw, output_window=ow, stride=s, num_features=nf)
        X_train, Y_train, X_test, Y_test = numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

        train_datasets.append(torch.utils.data.TensorDataset(X_train, Y_train))
        test_datasets.append(torch.utils.data.TensorDataset(X_test, Y_test))

    return train_datasets, test_datasets
