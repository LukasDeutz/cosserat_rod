# Build-in imports
from os.path import abspath
import argparse

# Third party imports
from fenics import Function, Expression, sqrt, dot, assemble, dx
import numpy as np
import pickle 

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod, grad
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver

data_path = '../../../data/constant_controls/'

model_parameters = ModelParameters(external_force = 'linear_drag', B_ast = 0.1*np.identity(3), S_ast = 0.1*np.identity(3))
solver = Solver(linearization_method = 'picard_iteration')

def calculate_elastic_energy(FS, CS, rod, shear, stretch):

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
    
    shear = str(shear)[0]
    stretch = str(stretch)[0]
    
    file_path = data_path + f'elastic_energies_stretch={stretch}_shear={shear}_N={N}_dt_{dt}.dat'
    
    pickle.dump(all_E, open(file_path, 'wb'))
    
    print(f'Done!: Save file to {abspath(file_path)} \n')
    
    return


def simulate_constant_controls(N, dt, T, stretch = False, shear = False):
    
    print(f'Simulate constant controls for N={N} and dt={dt}')
        
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
                
    n = int(T/dt)        
    rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)

    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)
    
    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps=n)
    
    FS = rod.solve(T, CS)        
        
    calculate_elastic_energy(FS, CS, rod, stretch, shear)
    
    print('Finished simulations!')
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('dt', type=float)
    parser.add_argument('T', type=float)
    
    args = parser.parse_args()
    N  = args.N
    dt = args.dt
    T  = args.T
    
    simulate_constant_controls(N, dt, T, stretch = True, shear=True)
    
    






