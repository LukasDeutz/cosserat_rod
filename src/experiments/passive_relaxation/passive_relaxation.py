'''
Created on 2 Nov 2021

@author: lukas
'''

# Build-in imports
from os.path import abspath

# Third party imports
from fenics import Function, Expression, sqrt, dot, assemble, dx
import numpy as np
import pickle 
import matplotlib.pyplot as plt

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod, grad
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver


data_path = '../../data/experiments/passive_relaxation/'
fig_path = '../../fig/experiments/passive_relaxation/'

# Parameter
N = 100

def zero_controls(worm):

    Omega_expr = Expression(('0', '0', '0'), degree = 1)    
    sigma_expr = Expression(('0', '0', '0'), degree = 1)
    
    Omega_pref = Function(worm.function_spaces['Omega'])
    sigma_pref = Function(worm.function_spaces['sigma'])
    
    Omega_pref.assign(Omega_expr)
    sigma_pref.assign(sigma_expr)

    C  = ControlsFenics(Omega_pref, sigma_pref)
    
    return C

def semi_circle_controls(worm):
    
    L0 = 1 # Natural length in rescalled coordinates
    
    # Choose radius such that worm curles into semi-circle        
    R = L0/np.pi 
    
    Ome1 = 1./R
    
    Omega_expr = Expression(('Ome1', '0', '0'), degree=1, Ome1 = Ome1)    
    sigma_expr = Expression(('0', '0', '0'), degree = 1)
    
    Omega_pref = Function(worm.function_spaces['Omega'])
    sigma_pref = Function(worm.function_spaces['sigma'])
    
    Omega_pref.assign(Omega_expr)
    sigma_pref.assign(sigma_expr)

    C  = ControlsFenics(Omega_pref, sigma_pref)
    
    return C
    
def init_configuration():
    
    T = 2.5 
    dt = 0.001
    
    model_parameters = ModelParameters(external_force = 'linear_drag', 
                                       B_ast = 0.1*np.identity(3), 
                                       S_ast = 0.1*np.identity(3))
            
    solver = Solver(linearization_method = 'picard_iteration')
            
    n = int(T/dt)        
    
    worm = Rod(N, dt, model_parameters = model_parameters, solver = solver)    
    C = semi_circle_controls(worm)
    
    CS = ControlSequenceFenics(C, n_timesteps=n)
    
    FS = worm.solve(T, CS)    
    F0 = FS[-1]
        
    filepath = data_path + f'semi_circle_F0_N={N}.dat'    
    pickle.dump(F0.to_numpy(), open(data_path + filepath, 'wb'))
    
    print(f'Done! Saved file to {abspath(filepath)}')
        
def passive_relaxation_model_parameters():

    # Buffer-solution
    mu = 1.0 # viscosity
    # Ratio between normal and tangental drag-coefficients
    K = 2.0
        
    alpha = r/L  
        
    C_t = 2*np.pi*mu/np.log(L/r)
    


    B_ast = 0.1*np.identity(3), 
    S_ast = 0.1*np.identity(3))


    model_parameters = ModelParameters(external_force = 'resistive_force', 

    return
            
def passive_relaxation_from_semi_circle():

    T = 2.5 
    dt = 0.001


    
        
    filepath = data_path + f'semi_circle_F0_N={N}.dat'    
    F0 = pickle.load(open(data_path + filepath, 'wb'))     
    F0 = F0.to_fenics(rod)
    
    
    
    

# TODOs:



# Relax into bended configuration
# Set preferred controls to zero
# Relax back to straight configuration
# Measure time scale of relaxation (How would we do this?)

