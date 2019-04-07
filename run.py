# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:17:41 2019

@author: Cezary Olszowiec
"""
#-------------------------
import torch
from torchvision import transforms
#-------------------------
from model_classes.teaching_classes.class_teaching_model import Teaching
from model_classes.RNN_classes.class_RNN import RNN
from Data_classes.class_Data import ToTensor, Dataset_Trajectories, Splitted_Data
from Data_classes.Generation_of_data.class_generate_data import Generate_Data
#-------------------------
import config
#-------------------------
import logging
logging.basicConfig(level=config.logging_level)
#-------------------------
torch.set_default_tensor_type(config.default_tensor_type)
#-------------------------


#---------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    if config.parameters_data['generate_data']:
        generate_data = Generate_Data(config.parameters_data['data_directory'], 
                                      config.parameters_ode['list_of_invariant_objects'], 
                                      config.parameters_ode['initial_value_domain'], 
                                      config.parameters_data['how_many_files'], 
                                      config.parameters_ode['dimension'], 
                                      config.parameters_ode['order'], 
                                      config.parameters_ode['t_start'], 
                                      config.parameters_ode['t_end'],
                                      config.parameters_ode['number_of_intermediate_points']
                                      )
        generate_data.generate()
      
        
    dataset = Dataset_Trajectories(directory = config.parameters_data['data_directory'], 
                                   transform = transforms.Compose([ToTensor()]) )

    splitted_data = Splitted_Data(dataset, 
                                  config.parameters_data['percentage_of_training_data'])
    
    rnn = RNN(config.parameters_rnn['input_size'], 
              config.parameters_rnn['hidden_size'], 
              config.parameters_rnn['output_size'], 
              config.parameters_rnn['num_classes'], 
              config.parameters_rnn['num_layers'], 
              config.parameters_rnn['batch_size'], 
              config.parameters_rnn['sequence_length'])

    
    teaching = Teaching(config.parameters_teaching['epochs'], 
                        config.parameters_rnn['batch_size']) 


    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=config.parameters_teaching['learning_rate'])  
    
    averaged_training_losses, validation_losses = teaching.run(rnn, splitted_data, criterion, optimizer, config.parameters_teaching['num_workers'])
    
    print('averaged_training_losses', averaged_training_losses) 
          
    print('validation_losses', validation_losses)

