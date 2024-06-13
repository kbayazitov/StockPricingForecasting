import torch
from torch import nn
import random

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

class LSTM_seq2seq(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.4, bidirectional=False, teacher_forcing_ratio=0, target_len=10):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(LSTM_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.target_len = target_len

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout, bidirectional=False)
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input, true_input=None):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''

        batch_size = input.size(0)

        # encode input_tensor
        #input = input.unsqueeze(1)     # add in batch size of 1 [batch_size,]
        _, encoder_hidden = self.encoder(input) # [batch_size, input_size, hidden_size]

        #decoder_input = input
        decoder_hidden = encoder_hidden

        # initialize tensor for predictions
        outputs = torch.zeros(len(input), self.target_len, 1).to(self.device) # [batch_size, target_len, input_size]


        if self.bidirectional:
            hidden = decoder_hidden[0].view(self.num_layers, 2, batch_size, self.hidden_size)
            cell = decoder_hidden[1].view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = hidden.sum(dim=1)  # Sum bidirectional outputs
            cell = cell.sum(dim=1)      # Sum bidirectional outputs

            decoder_hidden = (hidden, cell)

        # decode input_tensor
        decoder_input = input[:, -1, :].view(input.shape[0], 1, input.shape[2])

        #decoder_input = outputs[:, 0, :].view(input.shape[0], 1, input.shape[2])
        #decoder_input = torch.zeros(len(input), 1).view(input.shape[0], 1, input.shape[2]).to(device)

        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t:t+1] = self.linear(decoder_output)
            teacher_force = random.random() < self.teacher_forcing_ratio

            if true_input is not None:
                decoder_input = true_input[:, t].unsqueeze(1) if teacher_force else decoder_output
            else:
                decoder_input = decoder_output

        return outputs.squeeze(2)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
