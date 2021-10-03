# Build-in imports
from os.path import abspath

# Third party imports
from fenics import Expression, Function
import numpy as np
import pickle 
import matplotlib.pyplot as plt

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from cosserat_rod.plot3d import generate_interactive_scatter_clip, plot_controls_FS_vs_CS,\
    plot_strains
from cosserat_rod.frame import FrameFenics
from tests.constant_controls import model_parameters


# Paths undulation
data_path = "../../../data/undulation/"
fig_path = "../../../fig/undulation/"


def initial_posture(N, dt, solver, Omega_expr):
    
    # Relax to initial posture 
    
    T_init = 2.0
    n_init = int(T_init/dt) 

    model_parameters = ModelParameters()

    rod = Rod(N, dt, 
              model_parameters=model_parameters, 
              solver=solver)
    
        
   
    sigma_expr = Expression(('0', '0', '0'), degree = 1)    

    Omega = Function(rod.function_spaces['Omega'])
    sigma = Function(rod.function_spaces['sigma'])
    
    
    Omega.assign(Omega_expr)
    sigma.assign(sigma_expr)
    
    C_init = ControlsFenics(Omega, sigma)
    CS_init = ControlSequenceFenics(C_init, n_timesteps = n_init)
        
    FS = rod.solve(T_init, CS=CS_init)
    
    return FS[-1]
    
    
def sim_undulation(N, 
                   dt, 
                   T, 
                   solver, 
                   model_parameters,
                   Omega_expr,
                   F0 = None):
    
    print(f'Simulate undulation for N={N}, dt={dt}')

    solver.external_force = 'resistive_force'


    n = int(T/dt) 
    t_arr = np.linspace(0, T, n)

    # init worm            
    rod = Rod(N, dt, 
              model_parameters=model_parameters, 
              solver=solver)
                        
    sigma_expr = Expression(('0', '0', '0'), degree = 1)    
                                    
    CS = []
        
    for t in t_arr:
                                        
        Omega_expr.t = t        
        
        Omega = Function(rod.function_spaces['Omega'])        
        sigma = Function(rod.function_spaces['sigma'])
        
        Omega.assign(Omega_expr)
        sigma.assign(sigma_expr)
        
        C = ControlsFenics(Omega, sigma)

        CS.append(C)
            
    # Simulate unudlation
    CS = ControlSequenceFenics(CS)    

    FS = rod.solve(T, CS=CS, F0 = F0)
    
    output = {}
    output['FS'] = FS.to_numpy()
    output['CS'] = CS.to_numpy()
    output['N']  = N
    output['dt'] = dt
    output['T'] = T

    return output
            
def stiffness_matrices():
    
    # Assume we rescaled the length dimension such that the rod has unit length.    
    # The ratio between length and radius, 
    beta = 0.05
    # Thus, in the scaled length dimension
    r = beta
    # Second are moment of intertia 
    I  = np.pi/4 * r**4    
    # If we want the bending ridgities want to be in the order of one then choose Young's modulus    
    E = 1/I
    
    p = 0.4 # Poisson's ratio    
    G = E/(2*(1 + p)) # Shear modulus
                        
    A = np.pi * r**2
    
    B = np.identity(3)
    B[0,0] = E*I
    B[1,1] = E*I
    B[2,2] = G*I
    
    S = np.identity(3)
    S[0,0] = G*A
    S[1,1] = G*A
    S[2,2] = E*A
    
    print(f'Bending and twisting stiffness matrix B \n: {B}')
    print(f'Shear and stretch stiffness matrix S \n: {S}')
    
    return

def dimless_stiffness_matrices():

    T = 1 # characteristic time scale in s
    E = 10*1e6 # Young's modulus in Pa
    p = 0.4 # Poisson's ratio
    G = E/(2*(1 + p)) # Shear modulus in Pa     
    L = 1*1e-3 # Worm length    
    r = 40*1e-6 # Worm radius in m
    Delta_r = 0.5*1e-6 # Width of the cuticle in m
    mu = 1*1e-3 # Viscosity in Pa
    a = 4./3 # Shear constant for disk

    K = 4*np.pi*mu/np.log(L/r) / T
    
    r1 = r
    r2 = r-Delta_r
        
    S = np.identity(3)
    
    A = np.pi*((r1/L)**2 - (r2/L)**2)
    
    S[0,0] = a * G * A/ K
    S[1,1] = S[0,0] 
    S[2,2] = E* A / K
    
    B = np.identity(3)
    
    I = 0.25*np.pi*((r1/L)**4 - (r2/L)**4)
    
    B[0,0] = E * I / K
    B[1,1] = B[0,0]
    B[2,2] = G * 2 * I / K
    
    print(f'Bending and twisting stiffness matrix B \n: {B}')
    print(f'Shear and stretch stiffness matrix S \n: {S}')
    
    # Viscosity matrices    
    S_ast = 0.1*S
    B_ast = 0.1*B
    
    #TODO
    K_rot = B
    K = 1./T
    
    return K, K_rot, S, S_ast, B, B_ast 

def dimless_undulation_parameters():
    
    T = 1
    L = 1 # worm length in mm
    A = 5
    lam = 1.54 # wave length in mm
    lam = lam/L    
    w = 1.76
    w = w/T

    k = 2*np.pi/lam
    
    return k,w,A


      
if __name__ == "__main__":
    
    pass
    
