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

data_path = '../data/cosserat_worm/'

N  = 50
dt = 0.01
T = 2.0

model_parameters = ModelParameters(external_force = 'resistive_force')

def test_zero_controls():

    print('Test zero controls')

    rod = Rod(N, dt, model_parameters)

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

def test_constant_stretch():
    
    print('Test constant stretch')
    
    rod = Rod(N, dt, model_parameters)

    n = int(T/dt)

    CS = {}

    Ome_list = []
    sig_list = []

    for i in range(n):
        
        # Stretch or compression ratio
        nu = 1.5
        # Pure axial stretch without shear
        sig3 = 1. - nu
        
        Ome = Expression(("0", "0", "0"), degree=1)
        sig = Expression(("0", "0", f"{sig3}"), degree=1)
        
        Ome_list.append(Ome)
        sig_list.append(sig)
        
    CS['Omega'] = Ome_list
    CS['sigma'] = sig_list

    FS = rod.solve(T, CS)
    
    FS = FS.to_numpy()
    
    plot_strains(FS, dt = dt)
    plt.show()
    
    generate_interactive_scatter_clip(FS, fps = 100)

    print('Done!\n')

def test_constant_shear():
    
    print('Test constant shear')
    
    rod = Rod(N, dt, model_parameters)

    n = int(T/dt)

    CS = {}

    Ome_list = []
    sig_list = []

    for i in range(n):
        
        # Pure axial stretch without shear
        sig1 = 1.0 
        
        Ome = Expression(("0", "0", "0"), degree=1)
        sig = Expression((f"{sig1}", "0", "0"), degree=1)
        
        Ome_list.append(Ome)
        sig_list.append(sig)
        
    CS['Omega'] = Ome_list
    CS['sigma'] = sig_list

    FS = rod.solve(T, CS)
    generate_interactive_scatter_clip(FS, fps = 100)

    print('Done!\n')
    
def test_constant_twist():
    
    print('Test constant twitch')
    
    rod = Rod(N, dt)

    n = int(T/dt)

    CS = {}

    Ome_list = []
    sig_list = []

    for i in range(n):
        
        # Stretch or compression ratio
        Ome3 = 2.*np.pi
        # Pure axial stretch without shear
                                
        Ome = Expression(("0", "0", f"{Ome3}"), degree=1)
        sig = Expression(("0", "0", "0"), degree=1)
        
        Ome_list.append(Ome)
        sig_list.append(sig)
        
    CS['Omega'] = Ome_list
    CS['sigma'] = sig_list

    FS = rod.solve(T, CS)

    FS = FS.to_numpy()

    plot_strains(FS, dt = dt)
    plt.show()
 
    generate_interactive_scatter_clip(FS, fps = 100)    
   
    print('Done!\n')
    
def test_constant_bend():
    
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
    FS = FS.to_numpy()
    
    # pickle.dump(FS, open(data_path + 'bending_test.dat', 'wb'))

    plot_strains(FS, dt = dt)
    plt.show()
    
    generate_interactive_scatter_clip(FS, fps = 100)    
    
    print('Done!\n')

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
        
        sig_from_x_t = f2n(sig_from_x_t, 'e1')
        sig_t = f2n(sig_t,'sigma')
        
        sig[i, :, :] = sig_t
        sig_from_x[i, :, :] = sig_from_x_t
                
    output = {'sig': sig, 'sig_from_x': sig_from_x}
        
    #pickle.dump(output, open(data_path + 'sigma.dat', 'wb'))

    return
    
if __name__ == '__main__':
    
    print('Run test for the cosserat rod \n')    
    # test_zero_controls()    
    #test_constant_stretch()    
    #test_constant_twist()        
    test_constant_bend()
    #test_constant_shear()
    
    #calc_strain_vector_for_constant_bend()
    
    print('Finished all tests!')
    