# Build-in imports
from os.path import abspath

# Third party imports
from fenics import Expression, sqrt, dot, assemble, dx, Function
import numpy as np
import pickle 
import matplotlib.pyplot as plt

# Local imports
from cosserat_rod.controls import ControlsFenics, ControlSequenceFenics
from cosserat_rod.rod import Rod, grad
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from cosserat_rod.plot3d import generate_interactive_scatter_clip, plot_strains


# N_arr  = [50, 100, 250, 500]
# dt_arr = [1e-1, 1e-2, 1e-3, 1e-4]

data_path = '../../data/tests/constant_controls/'
fig_path = '../../fig/tests/constant_controls/'

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

def test_constant_controls(N, dt, T, stretch = False, shear = False):
    
    print(f'Simulate constant controls for N={N} and dt={dt}')
            
    n = int(T/dt)        
    rod = Rod(N, dt, model_parameters = model_parameters, solver = solver)
    
    Omega_expr = Expression(("2.0*sin(3*pi*x[0]/2)", 
                             "3.0*cos(3*pi*x[0]/2)",
                             "5.0*cos(2*pi*x[0])"), 
                             degree=1)    
    
    if stretch:
        nu = Expression('1 + 0.5*cos(2*pi*pi*x[0])', degree = 1)        
    else:
        nu = 1.0        
    if shear:
        theta = 10.0/360 * 2 * np.pi        
        theta = Expression('theta_max*(1 - sin(2*pi*x[0])', degree = 1, theta = theta)
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
        sigma_expr = Expression(('0', '0', '0'))


    Omega = Function(rod.function_spaces['Omega'])
    Omega.assign(Omega_expr)
    
    sigma = Function(rod.function_spaces['sigma'])
    sigma.assign(sigma_expr)

    C = ControlsFenics(Omega, sigma)
    CS = ControlSequenceFenics(C, n_timesteps=n)
    
    FS = rod.solve(T, CS)        
    
    generate_interactive_scatter_clip(FS.to_numpy(), 200)    
            
    calculate_elastic_energy(FS, CS, rod)
    
    print('Finished simulation!')
    
def plot_elastic_energies(N_arr, dt_arr, suffix = 'zero_sigma'):

    print('Plot elastic energies')

    lab_fz = 16
    lg_fz = 8

    # Plot for every fixed N, different dts    
    for N in N_arr:
        
        fig = plt.figure()#figsize = [6.4, 10])

        ax = plt.subplot(111)

        for dt in dt_arr:
            
            file_name = f'elastic_energies_{suffix}_N={N}_dt_{dt}.dat'
            all_E = pickle.load(open(data_path + file_name, 'rb'))
            
            E = all_E['E']
            
            n = int(T/dt)            
            t_arr = np.linspace(0, T, n)

            ax.semilogy(t_arr, E, label = f'dt={dt}')
            ax.set_xlabel(r'time $t$', fontsize = lab_fz)
            ax.set_ylabel(rf'Energy $E$ for N={N}', fontsize = lab_fz)
            ax.legend(fontsize = lg_fz)
            plt.tight_layout()
        
        fig.savefig(fig_path + f'elastic_energy_{suffix}_N={N}.pdf')
        plt.close(fig)
    
    # Plot for every fixed N, different dts    
    for dt in dt_arr:
        
        fig = plt.figure()#figsize = [6.4, 10])

        ax = plt.subplot(111)

        for N in N_arr:
            
            file_name = f'elastic_energies_{suffix}_N={N}_dt_{dt}.dat'
            all_E = pickle.load(open(data_path + file_name, 'rb'))
            
            E = all_E['E']
            
            n = int(T/dt)            
            t_arr = np.linspace(0, T, n)

            ax.semilogy(t_arr, E, label = f'N={N}')
            ax.set_xlabel(r'time $t$', fontsize = lab_fz)
            ax.set_ylabel(rf'Energy $E$ for dt={dt}', fontsize = lab_fz)
            ax.legend(fontsize = lg_fz)
            plt.tight_layout()
        
        fig.savefig(fig_path + f'elastic_energy_{suffix}_dt={dt}.pdf')
        plt.close(fig)
                    
    print('Finished plotting!\n')

if __name__ == '__main__':

    N = 50
    dt = 1e-2
    T = 3.0

    test_constant_controls(N, dt, T, stretch = True)
    
    #plot_elastic_energies()
    
        










