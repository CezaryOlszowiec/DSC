# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:09:14 2019

@author: Cezary Olszowiec
"""
#-------------------------
import scipy.integrate
import numpy as np
#-------------------------
from ODE_classes.Invariant_Objects_classes.class_Invariant_Objects import Trajectory

#---------------------------------------------------------------------------------------------------------

class ODE(object):
    
    def __init__(self, vector_field, order: int, dimension: int):
        self._vector_field = vector_field
        self._order = order
        self._dimension = dimension
        
    def integrate_ode(self, initial_value, time_range) -> np.array:
        return scipy.integrate.odeint(self._vector_field, initial_value, time_range)

#---------------------------------------------------------------------------------------------------------
        
class IVP(ODE):
    
    def __init__(self, vector_field, order: int, dimension: int, initial_value: np.array, time_range: np.array):
        super(IVP, self).__init__(vector_field, order, dimension)
        self._solution = None #Trajectory
        self._initial_value = initial_value
        self._time_range = time_range
    
    def solve_ivp(self, type_of_invariant_object_corresponding_to_the_trajectory: str):
        self._solution = Trajectory(type_of_invariant_object_corresponding_to_the_trajectory, self.integrate_ode(self._initial_value, self._time_range), self._initial_value, self._time_range)
        
    @property
    def solution(self) -> Trajectory:
        return self._solution 

#---------------------------------------------------------------------------------------------------------