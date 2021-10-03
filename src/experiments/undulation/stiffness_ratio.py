'''
Created on 3 Oct 2021

@author: lukas
'''

# Build-in imports
from os.path import abspath

# Third party imports
from fenics import Expression, Function
import numpy as np
import pickle 
import matplotlib.pyplot as plt

# Local imports
from experiments.undulation.undulation import sim_undulation, initial_posture, data_path, fig_path
from experiments.undulation.plot_undulation import plot_head_trajectory
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from cosserat_rod.plot3d import plot_controls_CS_vs_FS

data_path = data_path + 'stiffness_ratio/'
fig_path = fig_path + 'stiffness_ratio/'

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
    

def simulate_2d_undulations(C_arr, N = 50, dt = 0.01, T = 5.0):

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
    
    for c in C_arr:
        
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

def plot_all_undulations(C_arr, N, dt, T):

    print('Plot undulations')

    for c in C_arr:
            
        # Get file        
        filename = f'undulation_N={N}_dt={dt}_T={T}_c={c}.dat'    
        data = pickle.load(open(data_path + filename, 'rb'))                                 
    
        # Get data     
        FS = data['FS'] 
        CS = data['CS'] 
        
        # plot head trajectory
        fig = plot_head_trajectory(FS)
        plt.savefig(fig_path + f'head_trajectory_N={N}_dt={dt}_T={T}_c={c}.pdf')
        plt.close(fig)
        
        # plot controls vs strains
        fig = plot_controls_CS_vs_FS(CS, FS)
        plt.savefig(fig_path + f'CS_vs_FS_N={N}_dt={dt}_T={T}_c={c}.pdf')
        plt.close(fig)

    print(f'Finished!')    
      
    return
    
      
if __name__ == "__main__":
    
    N = 100
    dt = 0.001
    T = 5
    
    C_arr = [1.0, 2.5, 5.0]
    
    simulate_2d_undulations(C_arr, N, dt, T)     
    plot_all_undulations(C_arr, N, dt, T)
    
    