'''
Created on 19 Sept 2021

@author: lukas
'''

# third party imports
from fenics import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

# local imports
from cosserat_rod.rod import Rod
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.util import f2n
from cosserat_rod.plot3d import generate_interactive_scatter_clip, plot_strains
from cosserat_rod.solver import Solver
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics

N  = 50
dt = 0.01

model_parameters = ModelParameters(external_force = 'linear_drag', B_ast = 0.1*np.identity(3), S_ast = 0.1*np.identity(3))
solver = Solver(linearization_method = 'picard_iteration')

def test_constant_to_zero_controls(stretch = True, shear = True):

    Omega_expr = Expression(("2.0*sin(3*pi*x[0]/2)", 
                             "3.0*cos(3*pi*x[0]/2)",
                             "5.0*cos(2*pi*x[0])"), 
                             degree=1)    

    if stretch:
        nu = Expression('1 + 0.5*cos(2*pi*pi*x[0])', degree = 1)        
    else:
        nu = 1.0        
    if shear:
        theta_max = 10.0/360 * 2 * np.pi        
        theta = Expression('theta_max*(1 - sin(2*pi*x[0]))', degree = 1, theta_max = theta_max)
        phi = Expression('2*pi*x[0]', degree = 1)        
    else:           
        phi = 0.0    
        theta = 0.0
    
    sigma_expr = Expression(('-nu*cos(phi)*sin(theta)', '-nu*sin(phi)*sin(theta)', '1 - nu*cos(theta)'), 
                       degree = 1,
                       nu = nu,
                       phi = phi,
                       theta = theta)
    
    if not stretch and not shear:
        sigma_expr = Expression(('0', '0', '0'), degree = 1)
    
    # Simulate deformation from straight configuration to constant controls     
    T = 1.5

    n = int(T/dt)        
    rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)

    Omega_pref = Function(rod.function_spaces['Omega'])
    sigma_pref = Function(rod.function_spaces['sigma'])
    
    Omega_pref.assign(Omega_expr)
    sigma_pref.assign(sigma_expr)

    C  = ControlsFenics(Omega_pref, sigma_pref)
    CS = ControlSequenceFenics(C, n_timesteps=n)
    
    FS_0 = rod.solve(T, CS)        
    
    # Simulate deformation from deformed configuration to zero controls         
    F = FS_0[-1]
    
    Omega_expr = Expression(('0', '0', '0'), degree = 1)
    sigma_expr = Expression(('0', '0', '0'), degree = 1)
    
    Omega_pref.assign(Omega_expr)
    sigma_pref.assign(Omega_expr)

    C  = ControlsFenics(Omega_pref, sigma_pref)
    CS = ControlSequenceFenics(C, n_timesteps=n)
    
    FS_1 = rod.solve(T, CS, F0 = F)        

    FS = FS_0 + FS_1

    generate_interactive_scatter_clip(FS.to_numpy(), 500)
    
if __name__ == '__main__':
    
    test_constant_to_zero_controls(stretch = True, shear = True)
    
    
