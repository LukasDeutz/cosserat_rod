'''
Created on 2 Nov 2021

@author: lukas
'''

# Third party imports
from fenics import Function, Expression, sqrt, dot, assemble, dx
import numpy as np
from itertools import product

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod, grad
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from sympy.physics.mechanics.functions import inertia

# Parameter
N  = 101
T  = 2.5
dt = 0.01

model_parameters = ModelParameters(external_force = 'linear_drag', 
                                   B_ast = 0.1*np.identity(3), 
                                   S_ast = 0.1*np.identity(3), 
                                   J = 0.0*np.identity(3),
                                   rho = 1e-2)

# linearization method
solver = Solver(linearization_method = 'picard_iteration', finite_difference_order={1:1, 2:1})

def constant_control(rod, stretch, shear):
    
    print('Test constant controls for different combinations of N and dt')
    
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
                
    Omega_pref = Function(rod.function_spaces['Omega'])
    sigma_pref = Function(rod.function_spaces['sigma'])
        
    Omega_pref.assign(Omega_expr)
    sigma_pref.assign(sigma_expr)
    
    C  = ControlsFenics(Omega_pref, sigma_pref)
    
    return C

def calculate_elastic_energy(FS, CS, rod):

    print('Calculate elastic energy from simulation results.')

    E = np.zeros(len(FS))
        
    for i, (F, C) in enumerate(zip(FS, CS)):
                
        x = F.x
        xu = sqrt(dot(grad(x), grad(x)))
        sigma = F.sigma
        Omega = F.Omega
        
        Omega_pref = C.Omega
        sigma_pref = C.sigma
        
        E_Omega = assemble((dot(Omega - Omega_pref, Omega - Omega_pref) * xu * dx()))
        E_sigma = assemble((dot(sigma - sigma_pref, Omega - Omega_pref) * xu * dx()))
            
        E[i]= E_Omega + E_sigma
    
    return E

def test_constant_controls():
        
        for inertia in [True]:
                    
            model_parameters.inertia = inertia
                
            n = int(T/dt)        
            rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)
                    
            C = constant_control(rod, stretch = True, shear = True)
            CS = ControlSequenceFenics(C, n_timesteps=n)
    
            FS = rod.solve(T, CS)        
                    
            E = calculate_elastic_energy(FS, CS, rod)
            
            tol = 1e-5                        
            eps = 1e-10
            
            # tests
            assert np.abs(E[-1]) <= tol, 'E did not converged to zero within tolerance: {tol}'
            assert np.all(np.diff(E) <= 0.0 + eps), 'E is not monotonically decreasing'
            
            print(f'Passed test inertia={inertia}')
        
if __name__ == '__main__':
    
    test_constant_controls()
        


        
    
    


