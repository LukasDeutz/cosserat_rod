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

data_path = '../../data/tests/'

N  = 50
dt = 0.01
T = 0.75

model_parameters = ModelParameters(external_force = 'linear_drag', B_ast = 0.1*np.identity(3), S_ast = 0.1*np.identity(3))
solver = Solver(linearization_method = 'picard_iteration')

def test_zero_controls():

    print('Test zero controls')

    rod = Rod(N, dt, model_parameters, solver)

    n = int(T/dt)

    CS = {}

    Ome_list = []
    sig_list = []

    for i in range(n):
        
        Ome = Expression(("0", "0", "0"), degree=1)
        sig = Expression(("0", "0", "0"), degree=1)
        
        Ome_list.append(Ome)
        sig_list.append(sig)
        
    CS['Omega'] = Ome_list
    CS['sigma'] = sig_list

    FS = rod.solve(T, CS)
    generate_interactive_scatter_clip(FS.to_numpy(), fps = 100)
    
    print('Done!\n')
        
    return

def test_constant_stretch(nu = 1.5, F0 = None, plot = False):
    
    print('Test constant stretch')
    
    rod = Rod(N, dt, model_parameters, solver)

    n = int(T/dt)

    Omega_expr = Expression(("0", "0", "0"), degree=1)
        
    # shear angle in degree
    theta = 0.0
    # convert into radian
    theta = theta/360 * 2 * np.pi        
    # Orientation in the e1-e2 plane
    phi = 0. 
    # Stretch ratio
    nu = nu
            
    sigma_expr = Expression(('-nu*cos(phi)*sin(theta)', '-nu*sin(phi)*sin(theta)', '1 - nu*cos(theta)'), 
                            degree = 1,
                            nu = nu,
                            phi = phi,
                            theta = theta)

    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)
    
    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps = n)
        
    FS = rod.solve(T, CS, F0=F0)

    if plot:
        plot_strains(FS.to_numpy(), dt = dt)    
        plt.show()
        generate_interactive_scatter_clip(FS.to_numpy(), fps = 100)
    
    print('Done!\n')

    return FS

def test_constant_shear(phi = 0, F0 = None, plot = False):
    
    print('Test constant shear')
    
    rod = Rod(N, dt, model_parameters, solver)
    
    n = int(T/dt)
    
    Omega_expr = Expression(("0", "0", "0"), degree=1)
                
    # shear angle in degree
    theta = 30.0
    # convert into radian
    theta = theta/360 * 2 * np.pi        
    # Orientation in the e1-e2 plane    
    phi = phi
    # Stretch ratio
    nu = 1.0
        
    sigma_expr = Expression(('-nu*cos(phi)*sin(theta)', '-nu*sin(phi)*sin(theta)', '1 - nu*cos(theta)'), 
                       degree = 1,
                       nu = nu,
                       phi = phi,
                       theta = theta)
        
    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)
                
    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps = n)
                
    FS = rod.solve(T, CS, F0=F0)
    
    if plot:    
        plot_strains(FS.to_numpy(), dt = dt)
        plt.show()
        generate_interactive_scatter_clip(FS.to_numpy(), fps = 100)

    print('Done!\n')
    
    return FS
    
def test_constant_twist(F0 = None, plot = False):
    
    print('Test constant twitch')
    
    rod = Rod(N, dt, model_parameters, solver)

    n = int(T/dt)
        
    Omega3 = 2.*np.pi
                                
    Omega_expr = Expression(("0", "0", f"Omega3"), degree=1, Omega3 = Omega3)
    sigma_expr = Expression(("0", "0", "0"), degree=1)

    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)

                
    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps = n)

    FS = rod.solve(T, CS, F0 = F0)

    if plot:
        plot_strains(FS.to_numpy(), dt = dt)
        plt.show()     
        generate_interactive_scatter_clip(FS.to_numpy(), fps = 100)    
   
    print('Done!\n')
    
    return FS
    
def test_constant_bend(bend_dir = 1, F0 = None, plot = False):
    
    print('Test constant bend')
    
    rod = Rod(N, dt, model_parameters, solver)
        
    # Curvature
    Omega = 2.*np.pi
    # Pure axial stretch without shear
    n = int(T/dt)
                                
    if bend_dir == 1:                                
        Omega_expr = Expression(("Omega", "0", "0"), degree=1, Omega = Omega)
    if bend_dir == 2:                                
        Omega_expr = Expression(("0", "Omega", "0"), degree=1, Omega = Omega)
        
    sigma_expr = Expression(("0", "0", "0"), degree=1)

    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)
        
    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps = n)

    FS = rod.solve(T, CS, F0 = F0)
    
    # pickle.dump(FS, open(data_path + 'bending_test.dat', 'wb'))

    if plot:
        plot_strains(FS.to_numpy(), dt = dt)
        plt.show()
        generate_interactive_scatter_clip(FS.to_numpy(), fps = 100)    
    
    print('Done!\n')
    
    return FS

def calc_strain_vector_for_constant_bend():

    print('Test constant bend')
    
    rod = Rod(N, dt, model_parameters)

    n = int(T/dt)

    CS = {}

    Ome_list = []
    sig_list = []

    for i in range(n):
        
        # Stretch or compression ratio
        Ome1 = 2.*np.pi
        # Pure axial stretch without shear
                                
        Ome = Expression((f"{Ome1}", "0", "0"), degree=1)
        sig = Expression(("0", "0", "0"), degree=1)
        
        Ome_list.append(Ome)
        sig_list.append(sig)
        
    CS['Omega'] = Ome_list
    CS['sigma'] = sig_list

    FS = rod.solve(T, CS)

    sig = np.zeros((len(FS), 3, rod.N))
    sig_from_x = np.zeros_like(sig) 

    for i, F in enumerate(FS.frames):
                                
        x   = F.x
        e1  = F.e1
        e2  = F.e2
        e3  = F.e3
        
        Q = outer(e1, rod.E1) + outer(e2, rod.E2) + outer(e3, rod.E3) 
        
        sig_from_x_t = project(rod.E3 - Q.T * grad(x) / rod.xu_0, rod.V3)        
        sig_t = F.sigma
        
        sig_from_x_t = f2n(sig_from_x_t)
        sig_t = f2n(sig_t)
        
        sig[i, :, :] = sig_t
        sig_from_x[i, :, :] = sig_from_x_t
                
    output = {'sig': sig, 'sig_from_x': sig_from_x}
        
    pickle.dump(output, open(data_path + 'sigma.dat', 'wb'))

    return
    
def test_all(plot = False):
    
    rod = Rod(N, dt, model_parameters, solver)

    n = int(T/dt)
    
    Omega_expr = Expression(('0', '0', '0'), degree = 1)
    sigma_expr = Expression(('0', '0', '0'), degree = 1)
        
    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)
    
    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps = n)
    
    # Pure stretch    
    FS_all = test_constant_stretch(nu = 1.5)
    
    # Relaxation to zero controls    
    F0 = FS_all[-1]    
    FS = rod.solve(T, CS, F0=F0)    
    
    FS_all += FS
    
    #Pure compression
    F0 = FS_all[-1]
    FS = test_constant_stretch(0.5, F0)
    FS_all += FS
    
    #Relaxation to zero controls    
    F0 = FS_all[-1]
    FS = rod.solve(T, CS, F0=F0)
    FS_all += FS
    
    #Pure shear in e1 direction
    F0 = FS_all[-1]
    FS = test_constant_shear(phi=0, F0 = F0)
    FS_all += FS

    #Relaxation to zero controls    
    F0 = FS_all[-1]
    FS = rod.solve(T, CS, F0=F0)
    FS_all += FS

    #Pure shear in e2 direction
    F0 = FS_all[-1]
    FS = test_constant_shear(phi=np.pi/2, F0 = F0)
    FS_all += FS

    #Relaxation to zero controls    
    F0 = FS_all[-1]
    FS = rod.solve(T, CS, F0=F0)
    FS_all += FS

    #Pure bend in around e1
    F0 = FS_all[-1]
    FS = test_constant_bend(bend_dir = 1, F0 = F0)
    FS_all += FS

    #Relaxation to zero controls    
    F0 = FS_all[-1]
    FS = rod.solve(T, CS, F0=F0)
    FS_all += FS

    #Pure bend in around e2
    F0 = FS_all[-1]
    FS = test_constant_bend(bend_dir = 2, F0 = F0)
    FS_all += FS

    #Relaxation to zero controls    
    F0 = FS_all[-1]
    FS = rod.solve(T, CS, F0=F0)
    FS_all += FS

    #Pure twist
    F0 = FS_all[-1]
    FS = test_constant_twist(F0=F0)
    FS_all += FS

    #Relaxation to zero controls    
    F0 = FS_all[-1]
    FS = rod.solve(T, CS, F0=F0)
    FS_all += FS
                    
    pickle.dump(FS_all.to_numpy(), open(data_path + 'all_controls_FS.dat', 'wb'))                                 
    
    if plot:    
        generate_interactive_scatter_clip(FS_all.to_numpy(), 500)
    
    return
    
    
if __name__ == '__main__':
    
    print('Run test for the cosserat rod \n')    
    #test_zero_controls()        
    #test_constant_stretch()    
    #test_constant_twist()        
    #test_constant_bend()
    #test_constant_shear()
    test_all()
    
    #calc_strain_vector_for_constant_bend()
    
    print('Finished all tests!')
    