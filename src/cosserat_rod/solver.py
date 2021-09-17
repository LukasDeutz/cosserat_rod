'''
Created on 13 Sept 2021

@author: lukas
'''
import numpy as np

class Solver():

    def __init__(self, 
                 linearization_method = 'simple', 
                 method_parameters = {}, 
                 fenics_solver = {}):        
        
        self.linearization_method = linearization_method
        self.method_parameters = {}
        
        if self.linearization_method == 'simple':
            pass
        elif self.linearization_method == 'picard_iteration':            
            self._init_picard_solver(method_parameters)
        elif self.linearization_method == 'newton':
            self._init_newton_solver(method_parameters)
                          
        self._init_fenics_solver(fenics_solver)
                          
    def _init_picard_solver(self, method_parameters):
        
        tol = 1e-5 # tolerance
        maxiter = 25 # maximum iterations
        _ord = np.inf # error norm
                
        self.method_parameters['tol'] = tol
        self.method_parameters['maxiter'] = maxiter
        self.method_parameters['ord'] = _ord
        
        self.method_parameters.update(method_parameters)
        
    def _init_newton_solver(self, kwargs):
                
        pass
                
    def _init_fenics_solver(self, fenics_solver):
        
        self.fenics_solver = {}
        self.fenics_solver['linear_solver'] = 'superlu'
        self.fenics_solver['preconditioner'] = 'ilu'
        
        self.fenics_solver.update(fenics_solver)
        
        return
        
        
        
        
        
        
        
    
        