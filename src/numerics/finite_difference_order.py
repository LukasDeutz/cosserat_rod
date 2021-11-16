'''
Created on 2 Nov 2021

@author: lukas
'''
from dolfin.fem.assembling import assemble

'''
Created on 2 Nov 2021

@author: lukas
'''

# Build-in imports
import os

# Third-party imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import product

# Local imports
from cosserat_rod.rod import Rod
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from cosserat_rod.controls import ControlSequenceFenics
from tests.constant_controls_test import constant_control

data_path = '../../data/numerics/finite_difference_order/'
fig_path = '../../fig/numerics/finite_difference_order/'

# Parameter
N  = 100
T  = 2.5
dt = 0.1

N_arr = [50, 100, 200, 400]
dt_arr = [0.1, 0.01, 0.001, 0.0001]

lm = 'picard_iteration' # linearization_method

model_parameters = ModelParameters(external_force = 'linear_drag', B_ast = 0.1*np.identity(3), S_ast = 0.1*np.identity(3))
solver = Solver(linearization_method = lm)

max_ord = 5

#TODO: Make a plot of the final centerlines

# def plot_comparison_discretization_schemes():
#
#     for dt in dt_arr:
#
#         filepath_1 = data_path + f'FS_N={N}_dt={dt}_T={T}_lm={lm}_ds=1obd.dat'        
#         filepath_2 = data_path + f'FS_N={N}_dt={dt}_T={T}_lm={lm}_ds=2obd.dat'
#
#         FS_1 = pickle.load(open(filepath_1, 'rb'))        
#         FS_2 = pickle.load(open(filepath_2, 'rb'))
#
#         X1 = FS_1.x
#         X2 = FS_2.x
#         err = np.mean(np.sqrt(np.sum((X1 - X2)**2, axis = 1)), axis = 1)
#
#         plt.semilogy(FS_1.t_arr, err, label = f'dt={dt}')
#
#     lfz = 18
#     plt.xlabel('$t$', fontsize = lfz)
#     plt.ylabel('err', fontsize = lfz)
#     plt.legend()
#     plt.show()
#
#     return
#
# def plot_error_to_smallest_timestep():
#
#     # Take FS for smallest time step as ground truth
#     dt_GT = dt_arr[-1]
#     filepath = data_path + f'FS_N={N}_dt={dt_GT}_T={T}_lm={lm}_ds={ds}.dat'        
#     FS_GT = pickle.load(open(filepath, 'rb'))
#
#     for dt in dt_arr:
#
#         skip = int(dt/dt_GT)        
#         filepath = data_path + f'FS_N={N}_dt={dt}_T={T}_lm={lm}_ds={ds}.dat'        
#         FS = pickle.load(open(filepath, 'rb'))
#
#         X0 = FS.x
#         X1 = FS_GT.x[::skip, :, :]
#         err = np.mean(np.sqrt(np.sum((X0 - X1)**2, axis = 1)), axis = 1)
#
#         plt.semilogy(FS.t_arr, err, label = f'dt={dt}')
#
#     lfz = 18
#     plt.xlabel('$t$', fontsize = lfz)
#     plt.ylabel('err', fontsize = lfz)
#     plt.legend()
#     plt.show()
#
#     return
    
def plot_error_for_difference_orders(N, dt, lm):
    
    # Take FS for smallest time step as ground truth
    fig = plt.figure()
    
    filepath = data_path + f'FS_N={N}_dt={dt}_T={T}_fdo={max_ord}_lm={lm}.dat'        
    FS_m_ord = pickle.load(open(filepath, 'rb'))
    
    t_arr = FS_m_ord.t_arr
    
    X = FS_m_ord.x

    for k in range(1, max_ord):

        filepath = data_path + f'FS_N={N}_dt={dt}_T={T}_fdo={k}_lm={lm}.dat'        
        FS_k = pickle.load(open(filepath, 'rb'))        
        X_k = FS_k.x
        err = np.mean(np.sqrt(np.sum((X - X_k)**2, axis = 1)), axis = 1)
                                                             
        plt.semilogy(t_arr, err, label = f'order={k}')
        
    lfz = 18
    plt.xlabel('$t$', fontsize = lfz)
    plt.ylabel('err', fontsize = lfz)
    plt.legend()
    
    return


def simulate_diffent_order_batch():
    
    for N, dt in product(N_arr, dt_arr):
        
        simulate_different_orders(N, dt)
        
    return
        
def simulate_different_orders(N, dt):

    print('Simulate relaxation to constant controls for different time steps: \n')

    n = int(T/dt)        
    
    solver.finite_difference_order[1] = max_ord
    rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)                    

    C = constant_control(rod, stretch = True, shear = True)
    CS = ControlSequenceFenics(C, n_timesteps=n)
    FS = rod.solve(T, CS)

    filepath = data_path + f'FS_N={N}_dt={dt}_T={T}_fdo={max_ord}_lm={lm}.dat'    
    pickle.dump(FS.to_numpy(), open(filepath, 'wb'))
                        
    for order in range(1, max_ord):
    
        # Simulate
        solver.finite_difference_order[1] = order
        rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)                    
        C = constant_control(rod, stretch = True, shear = True)
        CS = ControlSequenceFenics(C, n_timesteps=n)
        FS = rod.solve(T, CS)
        
        filepath = data_path + f'FS_N={N}_dt={dt}_T={T}_fdo={order}_lm={lm}.dat'    
        pickle.dump(FS.to_numpy(), open(filepath, 'wb'))
        print(f'Done! Saved file to {os.path.abspath(filepath)}')

    print('Finished!')    
    #compute_error(FS_list, dt_arr)
        
    return
    
if __name__ == '__main__':    
    #test_convergence()         
    #simulate_diffent_order_batch()    
    #simulate_different_orders(N, dt)    
    
    lm = 'picard_iteration'
    plot_error_for_difference_orders(N, dt, lm)
    lm = 'simple'
    plot_error_for_difference_orders(N, dt, lm)
    
    plt.show()
    












