# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:18:00 2019

@author: Cezary Olszowiec
"""
#-------------------------
import numpy as np
#-------------------------
from typing import Dict

#---------------------------------------------------------------------------------------------------------

class Vector_Fields(object):
    def __init__(self, order: int, dimension: int):
        self._types_of_vector_fields = {'sink': lambda eigenvalues: lambda x, t: np.matmul([[eigenvalues[0],0],[0, eigenvalues[1]]], x), #TODO for higher dimensions
                                        'source': lambda eigenvalues: lambda x, t: np.matmul([[eigenvalues[0],0],[0, eigenvalues[1]]], x), #TODO for higher dimensions 
                                        'hyperbolic equilibrium': lambda eigenvalues: lambda x, t: np.matmul([[eigenvalues[0],0],[0, eigenvalues[1]]], x), #TODO for higher dimensions
                                        'complex saddle': lambda matrix: lambda x, t: np.matmul(matrix, x), #TODO in higher dimensions
                                        'spiralling sink': lambda matrix: lambda x, t: np.matmul(matrix, x), #TODO in higher dimensions
                                        'spiralling source': lambda matrix: lambda x, t: np.matmul(matrix, x), #TODO in higher dimensions
                                        'neutral equilibrium': lambda a: lambda x, t: np.matmul([[a,-1.0],[1.0, -a]], x) #TODO in higher dimensions
                                            
                                        #'periodic orbit': 1, #TODO
                                        #'heteroclinic orbit': 1, #TODO
                                        #'homoclinic orbit': 1  #TODO
                                        } #TODO
        
    @property
    def types_of_vector_fields(self) -> Dict:
        return self._types_of_vector_fields

#TODO creation of vector fields for any dimension

#---------------------------------------------------------------------------------------------------------
        
    
class Eigensystem(object):
    def __init__(self, dimension: int, eigenvalues = None, eigenvectors = None):
        self._eigenvalues = eigenvalues
        self._eigenvectors= eigenvectors
        self._dimension = dimension

    @staticmethod
    def generate_random_eigenvalues_or_matrix_entries(dimension: int, type_of_invariant_object: str) -> np.array:
        return {
            'sink': -np.absolute(np.random.random_sample(dimension,)),
            'source': np.absolute(np.random.random_sample(dimension,)),
            'hyperbolic equilibrium': np.random.permutation( np.concatenate( (np.absolute(np.random.random_sample(1,)), -np.absolute(np.random.random_sample(1,)), np.random.random_sample(dimension - 2,) ), axis=None) ),
            'complex saddle': 1, #TODO in higher dimensions
            'spiralling sink': [[-0.5*np.random.random_sample(), -1.0],[1.0, 0]], #TODO in higher dimensions
            'spiralling source': [[0.5*np.random.random_sample(), -1.0],[1.0, 0]], #TODO in higher dimensions
            'neutral equilibrium': np.random.random_sample(), #TODO in higher dimensions
            'periodic orbit': 1, 
            'heteroclinic orbit': 1, 
            'homoclinic orbit': 1
        }.get(type_of_invariant_object, None)

#---------------------------------------------------------------------------------------------------------
