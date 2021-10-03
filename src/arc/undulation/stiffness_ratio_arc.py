'''
Created on 3 Oct 2021

@author: lukas
'''

# Build-in imports
from os.path import abspath
import argparse

# Third party imports
from fenics import Expression, Function
import numpy as np
import pickle 
import matplotlib.pyplot as plt

# Local imports
from experiments.undulation.undulation import sim_undulation, initial_posture
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from cosserat_rod.plot3d import plot_controls_FS_vs_CS

data_path = '../../../data/stiffness_ratio/'

def undulation_parameter(A = 1.0, Q = 5.5):
    ''' 
    A = wave-vector amplitude ratio
    B = worm length times q
    '''    
    L = 1.0 #     
    k = Q/L # wave vector    
    Amp = k*A # amplitude
    w = 3.5 # angular velocity
       
    return Amp, k, w
    

def simulate_2d_undulations(c, N = 50, dt = 0.01, T = 5.0):

    # model parameter
    K = 40
    B = np.identity(3)
    B_ast = 0.1*B    
    K_rot = B

    model_parameters = ModelParameters(external_force = 'resistive_force', 
                                       K = K,
                                       K_rot = K_rot,
                                       B = B,
                                       B_ast = B_ast)

    
    # undulation parameters    
    Amp, k, w = undulation_parameter()
           
    Omega_expr = Expression(("A*cos(k*x[0] - w*t)", 
                             "0",
                             "0"), 
                             degree=1,
                             t = 0,
                             A = Amp,
                             k = k,
                             w = w)    
    
    solver = Solver(linearization_method = 'picard_iteration')
    
    # start from initial posture on the limit cycle
    F0 = initial_posture(N, dt, solver, Omega_expr)
         
    print(f'Simulate undulation for stiffness ratio c={c}')
    
    S = c*B
    S_ast = 0.1*S
    model_parameters.S = S
    model_parameters.S_ast = S_ast
                        
    output = sim_undulation(N, dt, T, solver, model_parameters, Omega_expr, F0)        
                    
    filename = f'undulation_N={N}_dt={dt}_T={T}_c={c}.dat'    
    pickle.dump(output, open(data_path + filename, 'wb'))                                 
    print(f'Finished! Saved file to {abspath(data_path + filename)}')    
                
    return
      
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('dt', type=float)
    parser.add_argument('T', type=float)
    parser.add_argument('c', type=float)
    
    args = parser.parse_args()
    N  = args.N
    dt = args.dt
    T  = args.T
    c  = args.c
        
    simulate_2d_undulations(c, N, dt, T)     
    
    