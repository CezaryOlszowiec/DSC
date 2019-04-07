# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:42:55 2019

@author: Cezary Olszowiec
"""

#-------------------------
import logging
logging_level = logging.DEBUG
#-------------------------
default_tensor_type = 'torch.FloatTensor'
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')


#---------------------------------------------------------------------------------------------------------
parameters_data = {
        'generate_data': False,
        'how_many_files': 100,
        'data_directory': r'../Dynamical_Systems_Classifier/data/',
        'shuffle': True, 
        'num_workers': 0,
        'percentage_of_training_data': 0.8
    }
#---------------------------------------------------------------------------------------------------------
parameters_ode = {
        'dimension': 2,
        'order': 1,
        'list_of_invariant_objects': ['sink', 'source', 'hyperbolic equilibrium'],
        't_start': 0,
        't_end': 1,
        'number_of_intermediate_points': 100,
        'initial_value_domain': (-1,1)
    }
#---------------------------------------------------------------------------------------------------------
parameters_rnn = {
        'batch_size': 20,
        'input_size': 2, #dimension of phase space
        'hidden_size': 4, #dimension of hidden layer
        'num_classes': 3, #same as n_categories
        'output_size': 3, # IT IS THE SAME NUMBER num_classes
        'num_layers': 3, #how many hidden layers?
        'sequence_length': 100, # == parameters_ode['number_of_intermediate_points'] == length of a single trajectory     
    }
#---------------------------------------------------------------------------------------------------------
parameters_teaching = {
        'num_workers': 0,
        'epochs': 2,
        'learning_rate': 0.001,
    }