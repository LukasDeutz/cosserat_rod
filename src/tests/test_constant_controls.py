'''
Created on 29 Sept 2021

@author: lukas
'''
# Build-in imports

# Third party imports
from fenics import Function, Expression, sqrt, dot, assemble, dx
import numpy as np

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import CosseratRod, InextensibleKirchhoffRod, grad
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver

N  = 50 # number of vertices
dt = 1e-2 # time step
T = 3.0 # simulation time
n = int(T/dt) # number if time steps
tol = 1e-3 # Elastic energy should not be larger than tolerance

# Test constrols controls for
# Cosserat rod 
# Kirchhoff rod
# Simple linearization
# Picard iteration
 
def constant_curvature():
    
    Omega_expr = Expression(("2.0*sin(3*pi*x[0]/2)", 
                             "3.0*cos(3*pi*x[0]/2)",
                             "5.0*cos(2*pi*x[0])"), 
                             degree=1)    
    
    return Omega_expr
    
def constant_strain_vector():

    nu = Expression('1 + 0.5*cos(2*pi*pi*x[0])', degree = 1)        
    theta_max = 10.0/360 * 2 * np.pi        
    theta = Expression('theta_max*(1 - sin(2*pi*x[0]))', degree = 1, theta_max = theta_max)
    phi = Expression('2*pi*x[0]', degree = 1)            
    
    sigma_expr = Expression(('-nu*cos(phi)*sin(theta)', '-nu*sin(phi)*sin(theta)', '1 - nu*cos(theta)'), 
                       degree = 1,
                       nu = nu,
                       phi = phi,
                       theta = theta)    
 
    return sigma_expr
 
def calculate_elastice_energy(F, C, rod):

    x = F.x
    xu = sqrt(dot(grad(x), grad(x)))
    Omega = F.Omega
    
    Omega_pref = C.Omega

    E = assemble((dot(Omega - Omega_pref, Omega - Omega_pref) * xu * dx()))
    
    if rod.rod_type == 'Cosserat_rod':
        
        sigma = F.sigma
        sigma_pref = C.sigma        
        E_sigma = assemble((dot(sigma - sigma_pref, sigma - sigma_pref) * xu * dx()))
     
        E += E_sigma
     
    return E
      
def test_constant_controls_cosserat_rod():

    model_parameters = ModelParameters(external_force = 'linear_drag', 
                                       B = np.identity(3),
                                       S = np.identity(3),
                                       B_tilde = 0.1*np.identity(3), 
                                       S_tilde = 0.1*np.identity(3))
    
    solver = Solver(linearization_method = 'picard_iteration')
    
    rod = CosseratRod(N, dt, model_parameters, solver)

    #TODO: This should be done in the Control class
    Omega_expr = constant_curvature()
    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma_expr = constant_strain_vector()
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)

    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps = n)
    
    FS = rod.solve(T, CS)
    
    E = calculate_elastice_energy(FS[-1], C, rod)
    
    assert E<tol, f'Elastic energy E={E} is larger than tolerance {tol}'
    
    return

def test_constant_controls_inextensible_kirchhoff_rod():

    model_parameters = ModelParameters(external_force = 'linear_drag', 
                                       B = np.identity(3),
                                       B_tilde = 0.1*np.identity(3))    
    
    solver = Solver(linearization_method = 'picard_iteration')
    
    rod = InextensibleKirchhoffRod(N, dt, model_parameters, solver)

    #TODO: This should be done in the Control class
    Omega_expr = constant_curvature()
    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    C = ControlsFenics(Omega)
    CS = ControlSequenceFenics(C, n_timesteps = n)
    
    FS = rod.solve(T, CS)
    
    E = calculate_elastice_energy(FS[-1], C, rod)
    
    print(E)
    
    assert E>tol
    
    return

if __name__ == '__main__':
    
    #test_constant_controls_cosserat_rod()
    test_constant_controls_inextensible_kirchhoff_rod()
    
        










