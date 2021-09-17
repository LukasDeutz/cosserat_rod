# Build-in imports
from os.path import abspath

# Third party imports
from fenics import Expression, sqrt, dot, assemble, dx
import numpy as np
import pickle 
from itertools import product

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod, grad
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from tests import constant_controls

# N_arr  = [50, 100, 250, 500]
# dt_arr = [1e-1, 1e-2, 1e-3, 1e-4]

data_path = '../../data/tests/constant_controls/'
fig_path = '../../fig/tests/constant_controls/'

N  = 50
dt = 1e-2

T = 2.0

model_parameters = ModelParameters(external_force = 'linear_drag', B_ast = 0.1*np.identity(3), S_ast = 0.1*np.identity(3))
solver = Solver(linearization_method = 'picard_iteration')

def calculate_elastic_energy(FS, CS, rod):

    print('Calculate elastic energy from simulation results.')

    E = np.zeros(len(FS))
    E_Omega_arr = np.zeros_like(E) 
    E_sigma_arr = np.zeros_like(E)
        
    for i, (F, C) in enumerate(zip(FS, CS)):
                
        x = F.x
        xu = sqrt(dot(grad(x), grad(x)))
        sigma = F.sigma
        Omega = F.Omega
        
        Omega_pref = C.Omega
        sigma_pref = C.sigma
        
        E_Omega = assemble((dot(Omega - Omega_pref, Omega - Omega_pref) * xu * dx()))
        E_sigma = assemble((dot(sigma - sigma_pref, Omega - Omega_pref) * xu * dx()))
        
        E_Omega_arr[i] = E_Omega
        E_sigma_arr[i] = E_sigma
            
    E = E_Omega_arr + E_sigma_arr

    N  = rod.N
    dt = rod.dt

    all_E = {}
    all_E['E'] = E
    all_E['E_Omega'] = E
    all_E['E_sigma'] = E
    all_E['N']  = N
    all_E['dt'] = dt
    
    file_path = data_path + f'elastic_energies_zero_sigma_N={N}_dt_{dt}.dat'
    
    pickle.dump(all_E, open(file_path, 'wb'))
    
    print(f'Done!: Save file to {abspath(file_path)} \n')
    
    return

def test_constant_controls():
    
    print('Test constant controls for different combinations of N and dt')
    
    Omega_pref = Expression(("2.0*sin(3*pi*x[0]/2)", 
                             "3.0*cos(3*pi*x[0]/2)",
                             "5.0*cos(2*pi*x[0])"), 
                             degree=1)    
    
    sigma_pref = Expression(('0', '0', '0'), degree = 1)
    
    print(f'Simulate constant controls for N={N} and dt={dt}')
            
    n = int(T/dt)        
    rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)

    C = ControlsFenics(Omega_pref, sigma_pref)
    CS = ControlSequenceFenics(C, n_timesteps=n)
    
    FS = rod.solve(T, CS)        
    print('Done!')
    
    calculate_elastic_energy(FS, CS, rod)
    
    print('Finished all simulations!')
        
    return

if __name__ == '__main__':
    
    test_constant_controls()







