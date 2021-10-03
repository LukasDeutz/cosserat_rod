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
        elif type(self.B) == Expression:            
            self.B = self.model_parameters.B
            
        if type(self.model_parameters.S) == np.ndarray:        
            self.S = Constant(self.model_parameters.S)
        elif type(self.S) == Expression:
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
        
        return

    def _init_function_space(self):
                
        self.mesh = UnitIntervalMesh(self.N-1)
    
        P1 = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
        P1_3 = MixedElement([P1] * 3) 
        
        self.V3 = FunctionSpace(self.mesh, P1_3)  
        
        #F, M, x, Ome, sig, w         
        self.VV = [self.V3, self.V3, self.V3, self.V3, self.V3, self.V3]         
        
        #F, M, x, Ome, sig, w
        self.W = FunctionSpace(self.mesh, MixedElement(6 * [P1_3]))
        
        # Create function space dictionary for easier access
        self.function_spaces = {}
        
        self.function_spaces['Omega'] = self.V3
        self.function_spaces['sigma'] = self.V3
        self.function_spaces['e123']  = self.V3
    
        self.u_n = Function(self.W)
    
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
                        
        #F, M, x, Ome, sig, w
        fa.assign(self.u_n, 
                  [Function(self.V3),
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

    def _init_form_simple(self):
                    
        # Solution from previous time step        
        _, _, x_n, Ome_n, _, _ = split(self.u_n)
        
        xu_n  = sqrt(inner(grad(x_n), grad(x_n))) # Length element
        tau_n = grad(x_n) / xu_n # Unit tangent vector
               
        # Trial and test functions               
        u = TrialFunction(self.W)
        v = TestFunction(self.W)

        F, M, x, Ome, sig, w = split(u)
        phi_F, phi_M, phi_x, phi_Ome, phi_sig, phi_w = split(v)
        
        # External force
        if self.model_parameters.external_force == 'linear_drag':            
            KK = self.K         
        elif self.model_parameters.external_force == 'resistive_force':             
            tautau = outer(tau_n, tau_n)
            P = Identity(3) - tautau            
            KK = self.K*P + tautau

        self.Q = outer(self.e1, self.E1) + outer(self.e2, self.E2) + outer(self.e3, self.E3)
       
        dx = Measure('dx', self.mesh)

        eq_F = dot(F, phi_F) * dx - dot(self.S * (sig - self.sig_pref), phi_F) * dx - dot(self.S_ast * (sig - self.sig_n) / self.dt, phi_F) * dx
        eq_M = dot(M, phi_M) * dx - dot(self.B * (Ome - self.Ome_pref), phi_M) * dx - dot(self.B_ast * (Ome - self.Ome_n) / self.dt, phi_M) * dx
                        
        eq_x   = dot(KK * (x - self.x_n), phi_x) / self.dt * xu_n * dx - dot(self.Q * F, grad(phi_x)) * dx        
        
        #eq_Ome = dot(-self.Q * self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(tau_n, self.Q * F), phi_Ome) * xu_n * dx - dot(self.Q * M, grad(phi_Ome)) * dx  
        eq_Ome = dot(-self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(self.Q.T * tau_n, F), phi_Ome) * xu_n * dx + dot(cross(Ome_n, M), phi_Ome) * dx - dot(M, grad(phi_Ome)) * dx          
        
        eq_sig = dot(sig, phi_sig) * dx - dot(self.E3, phi_sig) * dx + dot(self.Q.T * grad(x), phi_sig) / self.xu_0 * dx                 
        
        eq_w = dot(grad(w) / xu_n - (Ome - self.Ome_n) / self.dt - cross(w, Ome_n), phi_w) * dx \
              - dot(dot(grad(x) - grad(self.x_n), tau_n) / (self.dt * xu_n) * Ome_n, phi_w) * dx
        
        EQ = eq_F + eq_M + eq_x + eq_Ome + eq_sig + eq_w 
        
        self.F_op, self.L = lhs(EQ), rhs(EQ)
        
        return
        
    def _init_form_picard(self):
        
        
        # These are the functions from the previous time step which are used in the discretized 
        # time derivatives. They will not be updated during picard iteration!                
        self.x_tilde_n   = Function(self.V3)
        self.sig_tilde_n = Function(self.V3) 
        self.Ome_tilde_n = Function(self.V3)
        
        # Update approximation of functions at the current time step to be the solution 
        # of the last iteration step in the picard iteration
        _, _, x_n, Ome_n, _, _ = split(self.u_n)
                
        xu_n  = sqrt(inner(grad(x_n), grad(x_n)))
        tau_n = grad(x_n) / xu_n
               
        # Trial and test function
        u = TrialFunction(self.W)
        v = TestFunction(self.W)

        F, M, x, Ome, sig, w = split(u)
        phi_F, phi_M, phi_x, phi_Ome, phi_sig, phi_w = split(v)
        

        if self.model_parameters.external_force == 'linear_drag':            
            KK = self.K         
        elif self.model_parameters.external_force == 'resistive_force':                         
            tautau_n = outer(tau_n, tau_n)
            P_n = Identity(3) - tautau_n            
            KK = self.K*P_n + tautau_n

        self.Q = outer(self.e1, self.E1) + outer(self.e2, self.E2) + outer(self.e3, self.E3)
       
        dx = Measure('dx', self.mesh)

        eq_F = dot(F, phi_F) * dx - dot(self.S * (sig - self.sig_pref), phi_F) * dx - dot(self.S_ast * (sig - self.sig_tilde_n) / self.dt, phi_F) * dx
        eq_M = dot(M, phi_M) * dx - dot(self.B * (Ome - self.Ome_pref), phi_M) * dx - dot(self.B_ast * (Ome - self.Ome_tilde_n) / self.dt, phi_M) * dx
                        
        eq_x   = dot(KK * (x - self.x_tilde_n), phi_x) / self.dt * xu_n * dx - dot(self.Q * F, grad(phi_x)) * dx        
        
        #eq_Ome = dot(-self.Q * self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(tau_n, self.Q * F), phi_Ome) * xu_n * dx - dot(self.Q * M, grad(phi_Ome)) * dx  
        eq_Ome = dot(-self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(self.Q.T * tau_n, F), phi_Ome) * xu_n * dx + dot(cross(Ome_n, M), phi_Ome) * dx - dot(M, grad(phi_Ome)) * dx          
        
        eq_sig = dot(sig, phi_sig) * dx - dot(self.E3, phi_sig) * dx + dot(self.Q.T * grad(x), phi_sig) / self.xu_0 * dx                 
        
        eq_w = dot(grad(w) / xu_n - (Ome - self.Ome_tilde_n) / self.dt - cross(w, Ome_n), phi_w) * dx \
              - dot(dot(grad(x) - grad(self.x_tilde_n), tau_n) / (self.dt * xu_n) * Ome_n, phi_w) * dx
        
        EQ = eq_F + eq_M + eq_x + eq_Ome + eq_sig + eq_w 
        
        self.F_op, self.L = lhs(EQ), rhs(EQ)


    def _init_form(self):

        if self.solver.linearization_method == 'simple':
            self._init_form_simple()
        elif self.solver.linearization_method == 'picard_iteration':
            self._init_form_picard()
        elif self.solver.linearization_method == 'newton':
            pass                            
        return
                                                                    
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
        tol = self.solver.method_parameters['tol']
        maxiter = self.solver.method_parameters['maxiter']
        _ord = self.solver.method_parameters['ord']

        # Update functions from the previous time which are used in the discretized version 
        # of the time derivatives. These will not be updated during picard iteration!         
        _, _, x_tilde_n, Ome_tilde_n, sig_tilde_n, _ = split(self.u_n)
        
        x_tilde_n   = project(x_tilde_n, self.V3)
        Ome_tilde_n = project(Ome_tilde_n, self.V3)
        sig_tilde_n = project(sig_tilde_n, self.V3)
                
        self.x_tilde_n.assign(x_tilde_n)
        self.Ome_tilde_n.assign(Ome_tilde_n)
        self.sig_tilde_n.assign(sig_tilde_n)
        
        _iter = 0
                        
        print('Start picard iteration')

        while eps > tol and _iter < maxiter:

            solve(self.F_op == self.L, u, solver_parameters = self.solver.fenics_solver, bcs=self.bc)
            diff = u.vector().get_local() - self.u_n.vector().get_local()
            eps = np.linalg.norm(diff, ord = _ord)
            
            _iter += 1
            
            self.u_n.assign(u)
                                            
        if _iter < maxiter:
            print(f'Picard iteration converged after {_iter} iterations: norm={eps}')             
        else:
            print(f'Picard iteration did not converge: norm={eps}')             
            
        return u    
                                
    def solve(self, T, CS, F0 = None):

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

            _, _, x, Ome, sig, w = u.split()

            self.u_n.assign(u) 
            self.check_for_nans()

            # Update frame 
            self.F.update(x, self.e1, self.e2, self.e3, Ome, sig, w, t = self.t)
            FS.append(self.F.clone())
            
            print(f't={f_s.format(self.t)}')

        FS = FrameSequenceFenics(FS)            
        
        return FS

    def check_for_nans(self):
        
        f_names = ['F', 'M', 'x', 'Ome', 'sig', 'w']
        
        for i, f in enumerate(split(self.u_n)):
            
            f = project(f, self.V3)
            
            if np.isnan(f.vector().sum()):
                    raise RuntimeError(f'{f_names[i]} contains NaNs')
                                    
        return


    
    
    
        
    



