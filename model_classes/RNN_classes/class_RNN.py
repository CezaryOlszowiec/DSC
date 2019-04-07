# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:58:55 2019

@author: Cezary Olszowiec
"""
#-------------------------
import config
#-------------------------
import logging
logging.basicConfig(level=config.logging_level)
#-------------------------
import torch
torch.set_default_tensor_type(config.default_tensor_type)

#---------------------------------------------------------------------------------------------------------

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, num_layers, batch_size, sequence_length):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0) #input_size = dimension of state_space, hidden_size = output from LSTM, batch_first = batch goes first
        self.decoder = torch.nn.Linear(hidden_size, output_size, bias=False)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._num_classes = num_classes
        self._num_layers = num_layers
        self._batch_size = batch_size
        self._sequence_length = sequence_length 
        self._initrange = 0.1
        self.init_weights_decoder()
        #self.softmax = torch.nn.Softmax(dim=1) #LogSoftmax
        
    def init_weights_decoder(self):
        self.decoder.weight.data.uniform_(-self._initrange, self._initrange)

    def init_hidden(self):
        return torch.zeros(self._num_layers, self._batch_size, self._hidden_size)    

    def forward(self, input_layer, hidden_layer):
        input_layer = input_layer.view(self._batch_size, self._sequence_length, self._input_size)
        output, hidden_layer = self.rnn(input_layer, hidden_layer)
        output = self.decoder(output)
        return output, hidden_layer
          
#---------------------------------------------------------------------------------------------------------