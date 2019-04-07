# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:23:35 2019

@author: Cezary Olszowiec
"""

#-------------------------
import numpy as np
import os
#-------------------------
from ODE_classes.Invariant_Objects_classes.class_Invariant_Objects import Invariant_Object
#-------------------------
import config
#-------------------------
import logging
logging.basicConfig(level=config.logging_level)
#-------------------------
import torch
from torch.utils.data import Dataset, DataLoader, sampler
torch.set_default_tensor_type(config.default_tensor_type)

#---------------------------------------------------------------------------------------------------------

class Dataset_Trajectories(Dataset):
    def __init__(self, directory, transform=None):
        self.trajectories_in_numpy = np.array([np.loadtxt(os.path.join(directory, filename), dtype= 'float32' ) for filename in os.listdir(directory)])
        self.trajectories_name_files = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        self.trajectories_categories = np.array([ Invariant_Object.assign_category_to_file_name(filename) for filename in os.listdir(directory)])
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.trajectories_in_numpy)
    
    def __getitem__(self, nr):
        sample = {'trajectory_category': self.trajectories_categories[nr], 'trajectory_in_numpy': self.trajectories_in_numpy[nr]}
        if self.transform:
            sample = self.transform(sample)
        return sample

#---------------------------------------------------------------------------------------------------------

class ToTensor(object):
    def __call__(self, sample):
        trajectory_category, trajectory_in_numpy = sample['trajectory_category'], sample['trajectory_in_numpy']
        trajectory_tensor = torch.zeros(len(trajectory_in_numpy), len(trajectory_in_numpy[0]))
        trajectory_category_tensor = torch.zeros(1, len(trajectory_category))     
        for j in range(len(trajectory_in_numpy)):
            trajectory_tensor[j,:] = torch.Tensor(trajectory_in_numpy[j])
        trajectory_category_tensor[0,:] = torch.Tensor(trajectory_category)
            
        return {'trajectory_category': trajectory_category_tensor,
                'trajectory_in_numpy': trajectory_tensor}
        
#---------------------------------------------------------------------------------------------------------

class Splitted_Data(object):
    def __init__(self, dataset: Dataset_Trajectories, percentage_of_training_data: float): 
        self._dataset = dataset
        self._train_sampler = []
        self._test_sampler = []
        self._percentage_of_training_data = percentage_of_training_data 
        self.split()
        
    def split(self): #nice method taken from stackoverflow -@Aldream
        indices = list(range(len(self._dataset)))
        split = int(np.floor(self._percentage_of_training_data * len(self._dataset)))
        np.random.seed(1)
        np.random.shuffle(indices)
        self._train_sampler = sampler.SubsetRandomSampler(indices[:split])
        self._test_sampler = sampler.SubsetRandomSampler(indices[split:])
        
    def get_train_loader(self, batch_size, num_workers):
        return DataLoader(self._dataset, batch_size = batch_size, sampler = self._train_sampler, num_workers = num_workers)

    def get_test_sampler(self, batch_size, num_workers):        
        return DataLoader(self._dataset, batch_size = batch_size, sampler = self._test_sampler, num_workers = num_workers)
        
#---------------------------------------------------------------------------------------------------------
        