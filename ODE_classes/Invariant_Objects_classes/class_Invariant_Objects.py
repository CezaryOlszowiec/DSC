# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:16:39 2019

@author: Cezary Olszowiec
"""
#-------------------------
import os
import numpy as np
#-------------------------
import config
#-------------------------
import logging
logging.basicConfig(level=config.logging_level)


#---------------------------------------------------------------------------------------------------------

class Invariant_Object(object):
    
    def __init__(self, type_of_invariant_object: str):
        self._type_of_invariant_object = type_of_invariant_object
        
    @staticmethod
    def assign_category_to_file_name(filename: str) -> np.array:
        idx = config.parameters_ode['list_of_invariant_objects'].index( ''.join(filter(config.whitelist.__contains__, filename)) )
        one2hot = np.zeros(len(config.parameters_ode['list_of_invariant_objects']), dtype = 'float32')
        one2hot[idx] = 1
        return one2hot
 
#---------------------------------------------------------------------------------------------------------
       
class Trajectory(Invariant_Object):
    
    def __init__(self, type_of_invariant_object_corresponding_to_the_trajectory: str, ode_trajectory: np.array, initial_point: np.array, time_range: np.array):
        super(Trajectory, self).__init__(type_of_invariant_object_corresponding_to_the_trajectory)
        self._ode_trajectory = ode_trajectory
        self._time_range = time_range
        self._initial_point = initial_point
    
    def save(self, root_dir: str, number_of_the_trajectory: int):
        try:
            path = os.path.join(root_dir, self._type_of_invariant_object + '_' + str(number_of_the_trajectory))
            np.savetxt(path, self._ode_trajectory, fmt='%f')
        except Exception as e:
            logging.error(e)
            
    @staticmethod
    def load(type_of_invariant_object: str, number_of_the_trajectory: int) -> np.array:
        return np.loadtxt(type_of_invariant_object + '_' + str(number_of_the_trajectory), dtype=float)

    def load_and_set_ode_trajectory(self, type_of_invariant_object: str, number_of_the_trajectory: int):
        try:
            self._ode_trajectory = self.load(type_of_invariant_object, number_of_the_trajectory)
        except Exception as e:
            logging.error(e)

    def plot_trajectory(self):
        pass
    #TODO
        
#---------------------------------------------------------------------------------------------------------        
        