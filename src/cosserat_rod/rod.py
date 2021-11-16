# Third party imports
from fenics import *
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import gmres

# Local imports
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.solver import Solver
from cosserat_rod.frame import FrameFenics, FrameSequenceFenics


def grad(function): return Dx(function, 0)
    
class Rod():
    
    def __init__(self, N, dt, model_parameters = None, solver = None):
        
        self.N = N
        self.dt = dt
        
        if model_parameters is None:
            model_parameters = ModelParameters()
            
        if solver is None:
            solver = Solver()

        self.model_parameters = model_parameters
        self.solver = solver
        
        self._init_model_parameters()
        self._init_global_frame()
                    
        self._init_function_space()
        self._init_local_reference_frame()
        self._init_centerline()
        self._init_generalized_curvuture()
        self._init_strain_vector()
        self._init_controls()
                
        self._init_form()
        self._init_boundary_conditions()
        
    def _init_model_parameters(self):
        
        # Material parameters
        if type(self.model_parameters.B) == np.ndarray:
            self.B = Constant(self.model_parameters.B)
        elif type(self.model_parameters.B) == Expression:            
            self.B = self.model_parameters.B
            
        if type(self.model_parameters.S) == np.ndarray:        
            self.S = Constant(self.model_parameters.S)
        elif type(self.model_parameters.S) == Expression:
            self.S = self.model_parameters.S
            
        if type(self.model_parameters.B_ast) == np.ndarray:
            self.B_ast = Constant(self.model_parameters.B_ast)
        elif type(self.model_parameters.B_ast) == Expression:
            self.B_ast = self.model_parameters.B_ast

        if type(self.model_parameters.S_ast) == np.ndarray:
            self.S_ast = Constant(self.model_parameters.S_ast)
        elif type(self.model_parameters.S_ast) == Expression:
            self.S_ast = self.model_parameters.S_ast
                    
        # External moment
        self.K_rot = Constant(self.model_parameters.K_rot)
        
        # External force
        if self.model_parameters.external_force == 'linear_drag': # K is a diagonal matrix                
            self.K = Constant(self.model_parameters.K)
        elif self.model_parameters.external_force == 'resistive_force': # K is a float
            self.K = self.model_parameters.K
        
        # Only needed if inertia terms are included into the model
        if self.model_parameters.inertia:
            if type(self.model_parameters.J == np.ndarray):
                self.J = Constant(self.model_parameters.J)
            elif type(self.model_parameters.J) == Expression:
                self.I = self.model_parameters.J
        
        return

    def _init_function_space(self):
                
        self.mesh = UnitIntervalMesh(self.N-1)
    
        P1 = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
        P1_3 = MixedElement([P1] * 3) 
        
        self.V = FunctionSpace(self.mesh, P1)
        
        self.V3 = FunctionSpace(self.mesh, P1_3)  
        
        #F, M, x, Ome, sig, w         
        self.VV = [self.V3]*7         
        
        #F, M, x, Ome, sig, w
        self.W = FunctionSpace(self.mesh, MixedElement(7 * [P1_3]))
        
        # Create function space dictionary for easier access
        self.function_spaces = {}
        
        self.function_spaces['Omega'] = self.V3
        self.function_spaces['sigma'] = self.V3
        self.function_spaces['e123']  = self.V3
        
        # The time derivatives in the equations of motions are approximated
        # by finite backwards differences. Depending on the desired order 
        # of accuracy, the backwards differences need to include N=max(ord1, ord2+1)
        # previouse time points, where ord1 is the desired accuracy for the 
        # first order time derivatives and ord2 is the accuracy of the second 
        # order time derivatives.        
        ord1 = self.solver.finite_difference_order[1]
        
        if self.model_parameters.inertia:        
            ord2 = self.solver.finite_difference_order[2]
            N = np.max([ord1, ord2+1])
        else:
            N = ord1
                            
        self.u_arr = [Function(self.W) for _ in range(N)]
                       
        return 
    
    def _init_global_frame(self):
        
        self.E1 = Constant((1, 0, 0)) 
        self.E2 = Constant((0, 1, 0))
        self.E3 = Constant((0, 0, 1))

        return

    
    def _init_local_reference_frame(self):
        
        self.e1 = Function(self.function_spaces['e123'])
        self.e2 = Function(self.function_spaces['e123'])
        self.e3 = Function(self.function_spaces['e123'])
        
        return
    
    def _init_local_frame_vectors(self):
    
        e1_0_expr = Expression(('1', '0', '0'), degree=1)
        e2_0_expr = Expression(('0', '1', '0'), degree=1)
        e3_0_expr = Expression(('0', '0', '1'), degree=1)

        e1_0 = Function(self.function_spaces['e123'])
        e2_0 = Function(self.function_spaces['e123'])
        e3_0 = Function(self.function_spaces['e123'])

        e1_0.assign(e1_0_expr)
        e2_0.assign(e2_0_expr)
        e3_0.assign(e3_0_expr)

        return e1_0, e2_0, e3_0
                                            
    def _init_centerline(self):
    
        x0 = Expression(('0', '0', 'x[0]'), degree=1)                        
        x0  = project(x0, self.V3)

        # initial natural length element
        self.xu_0 = sqrt(dot(grad(x0), grad(x0))) 
                                                                                                                
        return x0

    def _init_generalized_curvuture(self):
                
        Ome0_expr = Expression(('0', '0', '0'), degree = 1)
        Ome0 = Function(self.V3)
        Ome0.assign(Ome0_expr) 
        
        return Ome0
        
    def _init_strain_vector(self):

        sig0_expr = Expression(('0', '0', '0'), degree = 1)        
        sig0 = Function(self.V3)
        sig0.assign(sig0_expr)
        
        return sig0
        
    def _init_angular_velocity(self):
        
        w0_expr = Expression(('0', '0', '0'), degree = 1)        
        w0 = Function(self.V3)
        w0.assign(w0_expr)
        
        return w0
         
    def _init_controls(self):
        
        Ome_pref = Expression(('0', '0', '0'), degree = 1)
        sig_pref = Expression(('0', '0', '0'), degree = 1)
        
        self.Ome_pref = project(Ome_pref, self.V3)
        self.sig_pref = project(sig_pref, self.V3)
        
        return
        
    def _assign_initial_values(self, F0, proj = False):
        
        fa = FunctionAssigner(self.W, self.VV)

        self.e1.assign(F0.e1)
        self.e2.assign(F0.e2)
        self.e3.assign(F0.e3)

        if not proj:                
            x = F0.x
            Omega = F0.Omega
            sigma = F0.sigma
            w = F0.w
        else:
            # TODO: Projects are needed if we pass initial frame 
            # F0 to solve. Don't know why
            x = project(F0.x, self.V3)
            Omega = project(F0.Omega, self.function_spaces['Omega'])
            sigma = project(F0.sigma, self.function_spaces['sigma'])
            w = project(F0.w, self.V3)
        
        for u_n in self.u_arr:
                        
            #K, F, M, x, Ome, sig, w
            fa.assign(u_n, 
                      [Function(self.V3),
                       Function(self.V3),
                       Function(self.V3),
                       x, 
                       Omega, 
                       sigma, 
                       w]) 
                             
        return

    def _init_frame(self):
        
        e1_0, e2_0, e3_0 = self._init_local_frame_vectors()
                                        
        x0   = self._init_centerline()    
        Ome0 = self._init_generalized_curvuture()
        sig0 = self._init_strain_vector()
        w0   = self._init_angular_velocity()

        F0 = FrameFenics(
            x = x0,
            e1 = e1_0,
            e2 = e2_0,
            e3 = e3_0,
            Omega = Ome0,
            sigma = sig0,
            w = w0,
            t = 0.0)
                
        return F0
    

    def _init_first_time_derivatives(self, u, tau_h):

        x_t   = self._finite_difference_approximation_first_derivative(u, 3)
        Ome_t = self._finite_difference_approximation_first_derivative(u, 4)
        sig_t = self._finite_difference_approximation_first_derivative(u, 5)
        w_t   = self._finite_difference_approximation_first_derivative(u, 6)

        xu_t  = dot(grad(x_t), tau_h)    
        
        return x_t, Ome_t, sig_t, w_t, xu_t
                        
    def _finite_difference_approximation_first_derivative(self, u, idx):
    
        o1 = self.solver.finite_difference_order[1]
        
        z = split(u)[idx]    
        z_arr = [split(u_n)[idx] for u_n in self.u_arr]
        
        if o1 == 1:                
            z_t = (-1*z_arr[-1] + 1*z) / self.dt
        elif o1 == 2:
            z_t = (1*z_arr[-2] - 4*z_arr[-1] + 3*z) / (2*self.dt)
        elif o1 == 3:
            z_t = (-2*z_arr[-3] + 9*z_arr[-2] - 18*z_arr[-1] + 11*z) / (6*self.dt)
        elif o1 == 4:
            z_t = (3*z_arr[-4] - 16*z_arr[-3] + 36*z_arr[-2] - 48*z_arr[-1] + 25*z)  / (12*self.dt)
        elif o1 == 5:
            z_t = (-12*z_arr[-5] + 75*z_arr[-4] - 200*z_arr[-3] + 300*z_arr[-2] -300*z_arr[-1] + 137*z)  / (60*self.dt)
        
        return z_t
    
    def _finite_difference_approximation_second_derivative(self, u, idx):
            
        o2 = self.solver.finite_difference_order[2]
    
        z = split(u)[idx]    
        z_arr = [split(u_n)[idx] for u_n in self.u_arr]
        
        if o2 == 1:                
            z_tt = (1*z_arr[-2] - 2*z_arr[-1] + 1*z) / self.dt**2
        elif o2 == 2:
            z_tt = (-1*z_arr[-3] + 4*z_arr[-2] - 5*z_arr[-1] + 2*z) / self.dt**2
        elif o2 == 3:
            z_tt = (11*z_arr[-4] - 56*z_arr[-3] + 114*z_arr[-2] - 104*z_arr[-1] + 35*z) / (12*self.dt**2)
        elif o2 == 4:
            z_tt = (-10*z_arr[-5] + 61*z_arr[-4] - 156*z_arr[-3] + 214*z_arr[-2] - 154*z_arr[-1] + 45*z) / (12*self.dt**2)
        elif o2 == 5:
            z_tt = (137*z_arr[-6] - 972*z_arr[-5] + 2970*z_arr[-4] - 5080*z_arr[-3] + 5265*z_arr[-2] - 3132*z_arr[-1] + 812*z)  / (180*self.dt**2)
        
        return z_tt
                                       
    def _init_form(self):
                
        # Update approximation of functions at the current time step to be the solution 
        # of the last iteration step in the picard iteration  
        
        if self.solver.linearization_method == 'simple':        
            self.u_h = self.u_arr[-1]
        elif self.solver.linearization_method == 'picard_iteration':
            self.u_h = Function(self.W) 
                                                              
        _, _, _, x_h, Ome_h, _, w_h = split(self.u_h) # u_head
                                                              
        xu_h  = sqrt(inner(grad(x_h), grad(x_h)))
        tau_h = grad(x_h) / xu_h
        mu_h = xu_h/self.xu_0 # stretch/compression ratio
               
        # Trial and test function
        u = TrialFunction(self.W)
        v = TestFunction(self.W)

        K, F, M, x, Ome, sig, w = split(u)
        phi_K, phi_F, phi_M, phi_x, phi_Ome, phi_sig, phi_w = split(v)
        
        if self.model_parameters.external_force == 'linear_drag':            
            KK_h = self.K         
        elif self.model_parameters.external_force == 'resistive_force':                         
            # The scalar K is the ratio between normal and tangential drift coefficient
            tautau_h = outer(tau_h, tau_h)                        
            KK_h = tautau_h + self.K*(Identity(3) - tautau_h)

        if self.model_parameters.inertia:
            rho = self.model_parameters.rho

        self.Q = outer(self.e1, self.E1) + outer(self.e2, self.E2) + outer(self.e3, self.E3)
       
        dx = Measure('dx', self.mesh)

        x_t, Ome_t, sig_t, w_t, xu_t = self._init_first_time_derivatives(u, tau_h)
                    
        if self.model_parameters.inertia:
            # second derivative
            x_tt = self._finite_difference_approximation_second_derivative(u, 3)
                                                                                                                                        
        eq_K = dot(K, phi_K) * dx - dot(KK_h * x_t, phi_K) * dx         
        eq_F = dot(F, phi_F) * dx - dot(self.S * (sig - self.sig_pref), phi_F) * dx - dot(self.S_ast * sig_t, phi_F) * dx
        eq_M = dot(M, phi_M) * dx - dot(self.B * (Ome - self.Ome_pref), phi_M) * dx - dot(self.B_ast * Ome_t, phi_M) * dx
                                    
        if not self.model_parameters.inertia:                        
            eq_x   = dot(K, phi_x) * xu_h * dx - dot(self.Q * F, grad(phi_x)) * dx        
            eq_Ome = dot(-self.K_rot * w, phi_Ome) * xu_h * dx + dot(cross(self.Q.T * tau_h, F), phi_Ome) * xu_h * dx + dot(cross(Ome_h, M), phi_Ome) * dx - dot(M, grad(phi_Ome)) * dx          
            #eq_Ome = dot(-self.Q * self.K_rot * w, phi_Ome) * xu_h * dx + dot(cross(tau_n, self.Q * F), phi_Ome) * xu_h * dx - dot(self.Q * M, grad(phi_Ome)) * dx  
        
        else:                                                
            eq_x   = dot(K, phi_x) * xu_h * dx - dot(self.Q * F, grad(phi_x)) * dx - rho * dot(x_tt, phi_x)  * self.xu_0 * dx                          
            eq_Ome = - dot(self.K_rot * w, phi_Ome) * xu_h * dx - dot(M, grad(phi_Ome)) * dx + dot(cross(Ome_h, M), phi_Ome) * xu_h * dx  + dot(cross(self.Q.T * tau_h, F), phi_Ome) * xu_h * dx \
                   + 1 / mu_h * dot(cross(self.J * w_h, w), phi_Ome) * self.xu_0 * dx + 1 / mu_h**2 * dot(self.J * w_h, phi_Ome) * xu_t * dx \
                   - 1 / mu_h * dot(self.J * w_t, phi_Ome) * self.xu_0 * dx
                                                          
        eq_sig = dot(sig, phi_sig) * dx - dot(self.E3, phi_sig) * dx + dot(self.Q.T * grad(x), phi_sig) / self.xu_0 * dx                 
        
        eq_w = dot(grad(w) / xu_h - Ome_t - cross(w, Ome_h), phi_w) * dx - dot(xu_t / xu_h * Ome_h, phi_w) * dx
        
        EQ = eq_K + eq_F + eq_M + eq_x + eq_Ome + eq_sig + eq_w 
        
        self.F_op, self.L = lhs(EQ), rhs(EQ)
                                                                    
    def _init_boundary_conditions(self):
        
        if self.model_parameters.bc:        
            Ome_space = self.W.split()[3]
            self.Ome_b = project(self.Ome_pref, self.V3)    
            self.bc = [DirichletBC(Ome_space, self.Ome_b, lambda x, o: o)]
        else:
            self.bc = []
        
        return
        
    def update_local_reference_frame(self, w):
        
        w = self.Q * w 
        
        # Avoid nans if w=0 due to division by zero by adding DOLFIN_EPS        
        k = w / sqrt(dot(w, w) + DOLFIN_EPS) 
                
        theta = sqrt(dot(w, w)) * self.dt
        
        e3 = self.e3 * cos(theta) + cross(k, self.e3) * sin(theta) + k * dot(k, self.e3) * (1 - cos(theta))        
        e1 = self.e1 * cos(theta) + cross(k, self.e1) * sin(theta) + k * dot(k, self.e1) * (1 - cos(theta))

        e3 = project(e3 / sqrt(dot(e3, e3)), self.V3)
                
        e1 = e1 - dot(e1, e3) / dot(e3, e3) * e3
        e1 = project(e1 / sqrt(dot(e1, e1)), self.V3)

        e2 = cross(e3, e1)
        e2 = project(e2 / sqrt(dot(e2, e2)), self.V3)
        
        self.e3.assign(e3)
        self.e1.assign(e1)
        self.e2.assign(e2)
                
        # check if frame contains nans
        assert not np.isnan(self.e3.vector().get_local()).any()
        assert not np.isnan(self.e1.vector().get_local()).any()
        assert not np.isnan(self.e2.vector().get_local()).any()
        
        return 

    def _solve_picard_iteration(self):
        
        u = Function(self.W)
        eps = 1.0
        tol = self.solver.linearization_parameters['tol']
        maxiter = self.solver.linearization_parameters['maxiter']
        _ord = self.solver.linearization_parameters['ord']

        # Approximate u_head by the solution from the previous time step 
        self.u_h.assign(self.u_arr[-1])
                                
        _iter = 0
                        
        print('Start picard iteration')

        while eps > tol and _iter < maxiter:

            solve(self.F_op == self.L, u, solver_parameters = self.solver.fenics_solver, bcs=self.bc)
            diff = u.vector().get_local() - self.u_h.vector().get_local()
            eps = np.linalg.norm(diff, ord = _ord)
            
            _iter += 1
            
            self.u_h.assign(u)
                                            
        if _iter < maxiter:
            print(f'Picard iteration converged after {_iter} iterations: norm={eps}')             
        else:
            print(f'Picard iteration did not converge: norm={eps}')             
            
        return u    
                                
    def solve(self, T, CS, F0 = None, cb = None):

        self.t = 0.0
        self.n_timesteps = int(T / self.dt) 

        assert len(CS) == self.n_timesteps, 'Controls not available for every simulation step.'
 
        if F0 is None:            
            F0 = self._init_frame()
            self._assign_initial_values(F0)        
        else:
            self._assign_initial_values(F0, proj= True)        

        # Assign fenics functions stored in frame to u_n and e1,e2,e3
      
        # Get inital w to update frame
        w = F0.w
        
        self.F = F0.clone()

        FS = []
                            
        f_s = ("{:.%if}" % len(str(self.dt).split('.')[-1])) # format string accounts for relevant number of decimals of time step dt                        
        print(f'Solve forward (t={f_s.format(self.t)}..{f_s.format(self.t + T)} / n_steps={self.n_timesteps})')
        print(f't={f_s.format(self.t)}')
                
        for C in CS:

            self.t += self.dt
                                                
            self.update_local_reference_frame(w)            

            # Update boundary condition
            if self.model_parameters.bc:            
                self.Ome_b.assign(self.Ome_pref)
            
            # Update preferred controls
            self.Ome_pref.assign(C.Omega)
            self.sig_pref.assign(C.sigma)
                        
            # Compute and update solution            
            if self.solver.linearization_method == 'simple':            
                u = Function(self.W)                                                
                solve(self.F_op == self.L, u, bcs=self.bc, solver_parameters = self.solver.fenics_solver)            
            elif self.solver.linearization_method == 'picard_iteration':
                u = self._solve_picard_iteration()

            self.check_for_nans(u)

            _, _, _, x, Ome, sig, w = u.split()

            # Update u
            for n, u_n in enumerate(self.u_arr[:-1]):
                u_n.assign(self.u_arr[n+1])
            
            self.u_arr[-1].assign(u)                                                                                           
            
            # Update frame 
            self.F.update(x, self.e1, self.e2, self.e3, Ome, sig, w, t = self.t)
            FS.append(self.F.clone())
            
            if cb is not None:
                cb(self.F)
            
            print(f't={f_s.format(self.t)}')

        FS = FrameSequenceFenics(FS)            
        
        return FS

    def check_for_nans(self, u):
        
        f_names = ['K', 'F', 'M', 'x', 'Ome', 'sig', 'w']
        
        for i, f in enumerate(split(u)):
            
            f = project(f, self.V3)
            
            if np.isnan(f.vector().sum()):
                    raise RuntimeError(f'{f_names[i]} contains NaNs')
                                    
        return


    
    
    
        
    



