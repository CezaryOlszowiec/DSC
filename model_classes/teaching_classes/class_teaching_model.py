# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:09:06 2019

@author: Cezary Olszowiec
"""
#-------------------------
from functools import partial 
#-------------------------
from Data_classes.class_Data import Splitted_Data
#-------------------------
import config
#-------------------------
import logging
logging.basicConfig(level=config.logging_level)
#-------------------------
import torch
torch.set_default_tensor_type(config.default_tensor_type)
#-------------------------


#---------------------------------------------------------------------------------------------------------

class Teaching(object):
    def __init__(self, epochs, batch_size):
        self._batch_size = batch_size 
        self._epochs = epochs 
        
    def run(self, rnn, training_data: Splitted_Data, criterion, optimizer, num_workers):
        partial_evaluate = partial(self.evaluate_one_batch, rnn, criterion)
        averaged_training_losses = []
        validation_losses = []
        
        for iter in range(self._epochs):
            total_training_loss = 0
            dataloader = training_data.get_train_loader(self._batch_size, num_workers)
            index_of_the_last_batch = len(dataloader) - 1
            for i_batch, sample_batched in enumerate(dataloader):
                loss = partial_evaluate(sample_batched['trajectory_category'], sample_batched['trajectory_in_numpy'])
                total_training_loss += loss.item()
                if i_batch < index_of_the_last_batch:
                    loss.backward()
                    optimizer.step()
                else:
                    validation_losses.append(loss.item())
            averaged_training_losses.append(total_training_loss/len(dataloader))
        
        return averaged_training_losses, validation_losses

    def evaluate_one_batch(self, rnn, criterion, category_tensor, input_tensor):        
        hidden_layer = rnn.init_hidden()   
        rnn.zero_grad()
        output_layer, next_hidden = rnn(input_tensor, hidden_layer)
        loss_tensor_output = output_layer[:,-1,:]
        loss_tensor_category = category_tensor[:,0,:]
        loss_tensor_category = loss_tensor_category.type('torch.LongTensor')
        loss = criterion(loss_tensor_output, torch.max(loss_tensor_category , 1)[1])   
        return loss        
        
#---------------------------------------------------------------------------------------------------------