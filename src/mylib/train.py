import numpy as np
import torch
import torch.nn as nn
import random
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

def train_test_split(t, y, split=0.8):
    ind_split = int(split * len(y))

    t_train = t[:ind_split]
    y_train = y[:ind_split]
    #y_train = y_train.reshape(-1, 1)

    t_test = t[ind_split:len(y)]
    y_test = y[ind_split:len(y)]
    #y_test = y_test.reshape(-1, 1)

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

def makeplots(losses, labels, accs=None, from_n=0, colors=['blue', 'green', 'red', 'orange', 'purple', 'pink', 'black']):
    """
    Draws a plots of losses and accuracies
    Args:
        losses: list - losses
        labels: list - labels of plots
        accs: list - accuracies
        colors: list - colors of plots
        task: ['Classification', 'Regression'] - task name
    Returns:
        plot
    Example:
        >>>
    """
    if (accs is not None):
        for acc, color, label in zip(accs, colors, labels):
            mean = np.array(acc).mean(0)
            std = np.array(acc).std(0)
            x_axis = np.arange(0, len(mean))
            plt.plot(x_axis, mean, color=color, label=label)
            plt.fill_between(x_axis, mean-std, mean+std, alpha=0.3, color=color)
        plt.xlabel('Epochs', fontsize=30)
        plt.ylabel('Acc', fontsize=30)
        plt.grid()
        plt.legend(loc='best')
        plt.show()

    for loss, color, label in zip(losses, colors, labels):
        '''
        mean = np.array(loss).mean(0)
        std = np.array(loss).std(0)
        x_axis = np.arange(0, len(mean))
        '''
        loss = [loss[i][from_n:] for i in range(len(loss))]
        mean = np.array(loss).mean(0)
        std = np.array(loss).std(0)
        x_axis = np.arange(from_n, len(mean) + from_n)
        plt.plot(x_axis, mean, color=color, label=label)
        plt.fill_between(x_axis, mean-std, mean+std, alpha=0.3, color=color)
    plt.xlabel('Epochs', fontsize=30)
    plt.ylabel('MSE', fontsize=30)
    plt.grid()
    plt.legend(loc='best')
    plt.show()

def train_classifier_model(model, train_data, test_data, epochs=25, batch_size=32, attempts=2, SEED=42):

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    list_of_train_acc = []
    list_of_test_acc = []
    list_of_train_losses = []
    list_of_test_losses = []

    for attempt in tqdm(range(attempts)):

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_function = torch.nn.CrossEntropyLoss()
        #loss_function = nn.BCELoss()

        train_acc = []
        test_acc = []
        train_losses = []
        test_losses = []

        for epoch in tqdm(range(epochs), leave=False):
            train_generator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            train_true = 0
            train_loss = 0
            for x, y in tqdm(train_generator, leave=False):
                optimizer.zero_grad()
                x = x.to(model.device)
                y = y.to(model.device)
                #output = model(x, y)
                output = model(x)

                loss = loss_function(output, y.squeeze(1))
                #loss = loss_function(output, y.view(y.shape[0]))

                loss.backward()
                optimizer.step()
                #train_true += mean_squared_error(y.cpu(), output.cpu())
                #train_true += metrics.accuracy_score(y.cpu(), torch.argmax(output, axis=1).cpu())
                train_true += torch.sum(y.squeeze(1).cpu() == torch.argmax(output, axis=1).cpu()).item()
                train_loss += loss.cpu().item()

            test_generator = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
            test_true = 0
            test_loss = 0
            for x, y in tqdm(test_generator, leave=False):
                x = x.to(model.device)
                y = y.to(model.device)
                output = model(x)

                loss = loss_function(output, y.squeeze(1))
                #loss = loss_function(output, y.view(y.shape[0]))

                #test_true += mean_squared_error(y.cpu().squeeze(), output.cpu().squeeze())
                #test_true += metrics.accuracy_score(y.cpu(), torch.argmax(output, axis=1).cpu())
                test_true += torch.sum(y.squeeze(1).cpu() == torch.argmax(output, axis=1).cpu()).item()
                test_loss += loss.cpu().item()

            train_acc.append(train_true / len(train_data))
            test_acc.append(test_true / len(test_data))
            #train_acc.append(train_true*batch_size/len(train_data))
            #test_acc.append(test_true*batch_size/len(test_data))
            train_losses.append(train_loss*batch_size/len(train_data))
            test_losses.append(test_loss*batch_size/len(test_data))

        list_of_train_acc.append(train_acc)
        list_of_test_acc.append(test_acc)
        list_of_train_losses.append(train_losses)
        list_of_test_losses.append(test_losses)

    #return list_of_test_rmse, list_of_test_losses, list_of_train_rmse, list_of_train_losses
    return list_of_test_acc, list_of_test_losses, list_of_train_acc, list_of_train_losses

class EncoderClassifier(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, input_size, hidden_size, output_size=2, num_layers=2, dropout=0.3, bidirectional=True):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(EncoderClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        #self.classifier = nn.Linear(hidden_size * 2, 2)
        self.classifier = nn.Linear(2 * num_layers * int(bidirectional + 1) * hidden_size, output_size)

    def forward(self, input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        #input = input.unsqueeze(2)
        output, (h, c) = self.lstm(input)

        act = torch.cat([h, c], dim=0).transpose(0, 1)
        act = act.reshape(len(input), -1)
        act = self.classifier(act)
        return act

class CustomMSE(nn.Module):
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, predicted, target):
        #squared_error = [((len(predicted) - i) / len(predicted)) * torch.square(predicted[i] - target[i]) for i in range(len(predicted))]
        squared_error = [((i + 1) / len(predicted)) * torch.square(predicted[i] - target[i]) for i in range(len(predicted))]
        loss = torch.sum(torch.stack(squared_error), axis=1) / (predicted.size()[1])
        #loss = torch.sum(torch.square(predicted - target) , axis= 1) / (predicted.size()[1])
        loss = torch.sum(loss) / loss.shape[0]
        return loss

def train_forecasting_model(model, train_data, test_data, epochs=30, batch_size=32, lr=1e-3, loss_func='MSE', SEED=42):

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    #list_of_train_rmse = []
    #list_of_test_rmse = []
    list_of_train_losses = []
    list_of_test_losses = []

    attempts = 2

    for attempt in tqdm(range(attempts)):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if loss_func == 'MSE':
            loss_function = nn.MSELoss()
        elif loss_func == 'CustomMSE':
            loss_function = CustomMSE()

        #train_rmse = []
        #test_rmse = []
        train_losses = []
        test_losses = []

        for epoch in tqdm(range(epochs), leave=False):
            train_generator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            train_true = 0
            train_loss = 0
            for x, y in tqdm(train_generator, leave=False):
                optimizer.zero_grad()
                x = x.to(model.device)
                y = y.to(model.device)
                output = model(x, y)

                loss = loss_function(output, y[:,:,3])

                loss.backward()
                optimizer.step()
                #train_true += mean_squared_error(y.cpu(), output.cpu())
                train_loss += loss.cpu().item()

            test_generator = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
            test_true = 0
            test_loss = 0
            for x, y in tqdm(test_generator, leave=False):
                x = x.to(model.device)
                y = y.to(model.device)
                output = model(x)

                loss = loss_function(output, y[:,:,3])

                #test_true += mean_squared_error(y.cpu().squeeze(), output.cpu().squeeze())
                test_loss += loss.cpu().item()

            #train_rmse.append(train_true*32/len(train_data))
            #test_rmse.append(test_true*32/len(test_data))
            train_losses.append(train_loss*batch_size/len(train_data))
            test_losses.append(test_loss*batch_size/len(test_data))

        #list_of_train_rmse.append(train_rmse)
        #list_of_test_rmse.append(test_rmse)
        list_of_train_losses.append(train_losses)
        list_of_test_losses.append(test_losses)

    #return list_of_test_rmse, list_of_test_losses, list_of_train_rmse, list_of_train_losses
    return list_of_test_losses, list_of_train_losses

class Encoder(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        _, hidden = self.lstm(input)

        return _, hidden

class Decoder(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, input, encoder_hidden):

        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        output, hidden = self.lstm(input, encoder_hidden)
        output = self.linear(output)

        return output, hidden

class seq2seq(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.4, bidirectional=False, teacher_forcing_ratio=0, target_len=10):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.target_len = target_len

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input, true_input=None):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''
        # encode input_tensor
        #input = input.unsqueeze(1)     # add in batch size of 1 [batch_size,]
        _, encoder_hidden = self.encoder(input) # [batch_size, input_size, hidden_size]

        #decoder_input = input
        decoder_hidden = encoder_hidden

        # initialize tensor for predictions
        outputs = torch.zeros(len(input), target_len, 1).to(self.device) # [batch_size, target_len, input_size]

        # decode input_tensor
        decoder_input = input[:, -1, :].view(input.shape[0], 1, input.shape[2])

        #decoder_input = outputs[:, 0, :].view(input.shape[0], 1, input.shape[2])
        #decoder_input = torch.zeros(len(input), 1).view(input.shape[0], 1, input.shape[2]).to(device)

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t:t+1] = self.linear(decoder_output)
            teacher_force = random.random() < self.teacher_forcing_ratio

            if true_input is not None:
                decoder_input = true_input[:, t].unsqueeze(1) if teacher_force else decoder_output
            else:
                decoder_input = decoder_output

        return outputs.squeeze(2)

def plot_test_results(model, test_dataset, input_size=30, output_size=10, num_rows=5):

    fig, ax = plt.subplots(num_rows, figsize = (13, 15))

    # plot test predictions
    for i in range(num_rows):
        x_input = test_dataset[i * 5][0].unsqueeze(0)
        y_input = test_dataset[i * 5][1].unsqueeze(0)

        pred = model(x_input.to(model.device)).cpu().detach().numpy()
        ax[i].plot(np.linspace(0, input_size, input_size), x_input[0,:,3].tolist(), color='b')
        ax[i].plot(np.linspace(input_size, input_size+output_size+1, output_size+1), [x_input[0,-1,3].item()]+y_input[0,:,3].tolist(), color='g')
        ax[i].plot(np.linspace(input_size, input_size+output_size+1, output_size+1), [x_input[0,-1,3].item()]+pred[0].tolist(), color='r')

    plt.show()
