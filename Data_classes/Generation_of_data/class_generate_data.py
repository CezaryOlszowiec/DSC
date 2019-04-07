# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:15:52 2019

@author: Cezary Olszowiec
"""
#-------------------------
import os
import numpy as np
from multiprocessing import cpu_count, pool
from functools import partial
#-------------------------
import config
#-------------------------
import logging
logging.basicConfig(level=config.logging_level)
#-------------------------
from ODE_classes.class_ODE import IVP
from ODE_classes.Vector_Fields_classes.class_Vector_Fields import Vector_Fields, Eigensystem
#-------------------------

#---------------------------------------------------------------------------------------------------------
    
class Generate_Data(object):
    def __init__(self, directory, types, initial_value_domain, how_many_files, dimension, order, t_start, t_end, number_of_intermediate_points):
        self._directory = directory  
        self._types = types
        self._initial_value_domain = initial_value_domain  
        self._how_many_files = how_many_files 
        self._dimension = dimension 
        self._order = order 
        self._t_start = t_start 
        self._t_end = t_end 
        self._number_of_intermediate_points = number_of_intermediate_points 
        
    def generate(self):        
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        time_range = np.linspace(self._t_start, self._t_end, self._number_of_intermediate_points)
        
        pool_ = pool.Pool(cpu_count())
            
        for type_of_invariant_object_corresponding_to_the_trajectory in self._types: 
            partial_generate_one_type = partial(Generate_Data.generate_one_type, 
                                                self._directory, 
                                                type_of_invariant_object_corresponding_to_the_trajectory, 
                                                self._initial_value_domain, 
                                                self._dimension, 
                                                self._order, 
                                                time_range) 
            
            pool_.map(partial_generate_one_type, range(self._how_many_files))
    
    @staticmethod
    def generate_one_type(directory, type_of_invariant_object_corresponding_to_the_trajectory, initial_value_domain, dimension, order, time_range, j):
        vector_fields_defined = Vector_Fields(order, dimension)
        eigenvalues_or_matrix = Eigensystem.generate_random_eigenvalues_or_matrix_entries(dimension, type_of_invariant_object_corresponding_to_the_trajectory) 
        vector_field = vector_fields_defined.types_of_vector_fields[type_of_invariant_object_corresponding_to_the_trajectory](eigenvalues_or_matrix)
        initial_value = np.random.uniform(initial_value_domain[0], initial_value_domain[1], dimension)
        sink_ivp = IVP(vector_field, order, dimension, initial_value, time_range)
        sink_ivp.solve_ivp(type_of_invariant_object_corresponding_to_the_trajectory)
        trajectory = sink_ivp.solution
        trajectory.save(directory, j)

#---------------------------------------------------------------------------------------------------------
