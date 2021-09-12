# Third party imports
from fenics import *
import numpy as np

# Local imports
from cosserat_rod.model_parameters import ModelParameters
from cosserat_rod.frame import FrameFenics, FrameSequenceFenics

def grad(function): return Dx(function, 0)
    
class Rod():
    
    def __init__(self, N, dt, model_parameters = None):
        
        self.N = N
        self.dt = dt
        
        if model_parameters is None:
            model_parameters = ModelParameters()

        self.model_parameters = model_parameters
        self._init_model_parameters()
        self._init_global_frame()
                    
        self._init_function_space()
        self._init_frame()
        self._init_x()
        self._init_curvuture_and_strain()
        self._assign_initial_values()
        self._init_controls()
                
        self._init_form()
        self._init_boundary_conditions()
        
    def _init_model_parameters(self):
        
        if type(self.model_parameters.B) is np.ndarray:
            self.B = Constant(self.model_parameters.B)
        
        if type(self.model_parameters.S) is np.ndarray:        
            self.S = Constant(self.model_parameters.S)
            
        if type(self.model_parameters.B_ast) is np.ndarray:
            self.B_ast = Constant(self.model_parameters.B_ast)
        
        if type(self.model_parameters.B_ast) is np.ndarray:
            self.S_ast = Constant(self.model_parameters.S_ast)
        
        self.K_rot = Constant(self.model_parameters.K_rot)
        self.K = self.model_parameters.K

        return

    def _init_function_space(self):
        
        if self.model_parameters.linearize:
        
            self.mesh = UnitIntervalMesh(self.N-1)
        
            P1 = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
            P1_3 = MixedElement([P1] * 3) 
        
            self.V = FunctionSpace(self.mesh, P1)
        
            self.V3 = FunctionSpace(self.mesh, P1_3)  
            #F, M, x, Ome, sig, w         
            self.VV = [self.V3, self.V3, self.V3, self.V3, self.V3, self.V3]         
            
            #x, Ome, sig, w
            #self.VV = [self.V3, self.V3, self.V3, self.V3]         
        
            #F, M, x, Ome, sig, w
            self.W = FunctionSpace(
                self.mesh,
                MixedElement([P1_3, P1_3, P1_3, P1_3, P1_3, P1_3]) 
            )
            #x, Ome, sig, w    
            # self.W = FunctionSpace(
            #     self.mesh,
            #     MixedElement([P1_3, P1_3, P1_3, P1_3]) 
            # )

        
            self.u_n = Function(self.W)
    
        else:
            # TODO
            pass
            
        return 
    
    def _init_global_frame(self):
        
        self.E1 = Constant((1, 0, 0)) 
        self.E2 = Constant((0, 1, 0))
        self.E3 = Constant((0, 0, 1))

        return
        
    def _init_frame(self):

        e1_0 = Expression(('1', '0', '0'), degree=1)
        e2_0 = Expression(('0', '1', '0'), degree=1)
        e3_0 = Expression(('0', '0', '1'), degree=1)

        return e1_0, e2_0, e3_0
                                            
    def _init_x(self):
    
        x0 = Expression(('0', '0', 'x[0]'), degree=1)                        
        x0  = project(x0, self.V3)

        # initial natural length element
        self.xu_0 = sqrt(dot(grad(x0), grad(x0))) 
                                                                                                                
        return x0

    def _init_curvuture_and_strain(self):
        
        sig0 = Expression(('0', '0', '0'), degree = 1)        
        sig0 = project(sig0, self.V3)
        
        Ome0 = Expression(('0', '0', '0'), degree = 0)
        Ome0 = project(Ome0, self.V3) 
        
        return Ome0, sig0
        
    def _init_angular_velocity(self):
        
        w0 = Expression(('0', '0', '0'), degree = 1)        
        w0 = project(w0, self.V3)
        
        return w0
         
    def _init_controls(self):
        
        Ome_pref = Expression(('0', '0', '0'), degree = 1)
        sig_pref = Expression(('0', '0', '0'), degree = 1)
        
        self.Ome_pref = project(Ome_pref, self.V3)
        self.sig_pref = project(sig_pref, self.V3)
        
        return
        
    def _assign_initial_values(self):
        
        e1_0, e2_0, e3_0 = self._init_frame()
        
        self.e1 = project(e1_0, self.V3)
        self.e2 = project(e2_0, self.V3)
        self.e3 = project(e3_0, self.V3)
                
        x0 = self._init_x()    
        Ome0, sig0 = self._init_curvuture_and_strain()
        w0 = self._init_angular_velocity()

        self.F0 = FrameFenics(
            x = x0,
            e1 = self.e1,
            e2 = self.e2,
            e3 = self.e3,
            Omega = Ome0,
            sigma = sig0,
            w = w0,
            t = 0.0)

        fa = FunctionAssigner(self.W, self.VV)
                
        #F, M, x, Ome, sig, w
        fa.assign(self.u_n, [Function(self.V3),
                             Function(self.V3),
                             x0, 
                             Ome0, 
                             sig0, 
                             w0]) 

        # x, Ome, sig, w
        # fa.assign(self.u_n, [x0, 
        #                      Ome0, 
        #                      sig0, 
        #                      w0]) 

                                
        return

    def _init_form(self):

        if ModelParameters.linearize:
            self._init_linearized_form()
            return

        F_n, M_n, x_n, Ome_n, sig_n, w_n = split(self.u_n)

        xu_n = sqrt(inner(grad(x_n), grad(x_n)))
                    
        # Trial and test function
        u = TrialFunction(self.W)
        v = TestFunction(self.W)

        F, M, x, Ome, sig, w = split(u)
        phi_F, phi_M, phi_x, phi_Ome, phi_sig, phi_w = split(v)
        
        tau = grad(x)/sqrt(dot(x, x))

        if self.model_parameters.external_force == 'linear_drag':            
            KK = self.K*Constant(np.identity(3))         
        elif self.model_parameters.external_force == 'resistive_force':             
            tautau = outer(tau, tau)
            P = Identity(3) - tautau            
            KK = self.K*P + tautau

        self.Q = outer(self.e1, self.E1) + outer(self.e2, self.E2) + outer(self.e3, self.E3)
       
        dx = Measure('dx', self.mesh)

        eq_F = dot(F, phi_F) * dx - dot(self.S * (sig -  self.sig_pref), phi_F) * dx - dot(self.S_ast * (sig - sig_n) / self.dt, phi_F) * dx
        eq_M = dot(M, phi_M) * dx - dot(self.B * (Ome  - self.Ome_pref) / xu_n, phi_M)  * dx + dot(self.B_ast * (Ome - Ome_n) / self.dt, phi_M) * dx
                        
        eq_x   = dot(KK * (x - x_n), phi_x) / self.dt * xu_n * dx - dot(self.Q * F, grad(phi_x)) * dx        
        
        #eq_Ome = dot(-self.Q * self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(tau, self.Q * F), phi_Ome) * xu_n * dx - dot(self.Q * M, grad(phi_Ome)) * dx  
        eq_Ome = dot(-self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(self.Q.T * tau, F), phi_Ome) * xu_n * dx + dot(cross(Ome_n, M), phi_Ome) * dx - dot(M, grad(phi_Ome)) * dx          
        
        eq_sig = dot(sig, phi_sig) * dx - dot(self.E3, phi_sig) * dx + dot(self.Q.T * grad(x), phi_sig) / self.xu_0 * dx 
        
        eq_w = dot(grad(w) - (Ome - Ome_n) / self.dt + cross(Ome, w), phi_w) * dx
        
        EQ = eq_F + eq_M + eq_x + eq_Ome + eq_sig + eq_w 
        
        self.F_op, self.L = lhs(EQ), rhs(EQ)
        
                                    
    def _init_linearized_form(self):
            
        F_n, M_n, x_n, Ome_n, sig_n, w_n = split(self.u_n)
        #x_n, Ome_n, sig_n, w_n = split(self.u_n)
    
    
        xu_n = sqrt(inner(grad(x_n), grad(x_n)))
        tau_n = grad(x_n) / xu_n
        
        if self.model_parameters.external_force == 'linear_drag':            
            KK = self.K*Constant(np.identity(3)) 
        
        elif self.model_parameters.external_force == 'resistive_force': 
            
            tautau_n = outer(tau_n, tau_n)
            P_n = Identity(3) - tautau_n            
            KK = self.K*P_n + tautau_n

            # e3_o_e3 = outer(self.e3, self.e3)
            # P_n = Identity(3) - e3_o_e3
            #
            # KK_n = self.K*P_n + e3_o_e3
                        
        self.Q = outer(self.e1, self.E1) + outer(self.e2, self.E2) + outer(self.e3, self.E3)
                    
        dx = Measure('dx', self.mesh)
    
        # Define variational problem
        u = TrialFunction(self.W)
        v = TestFunction(self.W)
       
        F, M, x, Ome, sig, w = split(u)
        phi_F, phi_M, phi_x, phi_Ome, phi_sig, phi_w = split(v)

        # x, Ome, sig, w = split(u)
        # phi_x, phi_Ome, phi_sig, phi_w = split(v)

        eq_F = dot(F, phi_F) * dx - dot(self.S * (sig -  self.sig_pref), phi_F) * dx - dot(self.S_ast * (sig - sig_n) / self.dt, phi_F) * dx
        eq_M = dot(M, phi_M) * dx - dot(self.B * (Ome  - self.Ome_pref) / xu_n, phi_M)  * dx + dot(self.B_ast * (Ome - Ome_n) / self.dt, phi_M) * dx
            
        # F = self.S * (sig -  self.sig_pref) + self.S_ast * (sig - sig_n) / self.dt
        # M = self.B * (Ome  - self.Ome_pref) / xu_n + self.B_ast * (Ome - Ome_n) / self.dt
            
        eq_x   = dot(KK * (x - x_n), phi_x) / self.dt * xu_n * dx - dot(self.Q * F, grad(phi_x)) * dx        
        eq_Ome = dot(-self.Q * self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(tau_n, self.Q * F), phi_Ome) * xu_n * dx - dot(self.Q * M, grad(phi_Ome)) * dx  
        #eq_Ome = dot(-self.K_rot * w, phi_Ome) * xu_n * dx + dot(cross(self.Q.T * tau_n, F), phi_Ome) * xu_n * dx + dot(cross(Ome_n, M), phi_Ome) * dx - dot(M, grad(phi_Ome)) * dx          
        
        #eq_sig = dot(self.Q * sig, phi_sig) * dx - dot(self.e3, phi_sig) * dx + dot(grad(x), phi_sig) / self.xu_0 * dx 
        eq_sig = dot(sig, phi_sig) * dx - dot(self.E3, phi_sig) * dx + dot(self.Q.T * grad(x), phi_sig) / self.xu_0 * dx 
        
        eq_w = dot(grad(w) - (Ome - Ome_n) / self.dt + cross(Ome_n, w), phi_w) * dx
        
        EQ = eq_F + eq_M + eq_x + eq_Ome + eq_sig + eq_w 
        #EQ = eq_x + eq_Ome + eq_sig + eq_w 
        
        self.F_op, self.L = lhs(EQ), rhs(EQ)
        
        return

    def _init_boundary_conditions(self):
        
        Ome_space = self.W.split()[3]
        self.Ome_b = project(self.Ome_pref, self.V3)    
        self.bc = [DirichletBC(Ome_space, self.Ome_b, lambda x, o: o)]
                
        return
        
    def update_frame(self, w):
        
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

    def solve(self, T, CS, F0 = None):

        self.t = 0.0

        n_timesteps = int(T / self.dt) 

        if F0 is None:
            F0 = self.F0
       
        w = F0.w
        
        self.F = F0.clone()
            
        assert len(CS['Omega']) == n_timesteps, 'Controls not available for every simulation step.'
        
        # dynamic format string
        f_s = ("{:.%if}" % len(str(self.dt).split('.')[-1]))                        
        print(f'Solve forward (t={f_s.format(self.t)}..{f_s.format(self.t + T)} / n_steps={n_timesteps})')
        print(f't={f_s.format(self.t)}')

        FS = []
                
        for (Ome_pref, sig_pref) in zip(CS['Omega'], CS['sigma']):
                        
            # Update preferred curvuture
            self.Ome_pref.assign(Ome_pref)
            self.sig_pref.assign(sig_pref)
            
            # Update frame
            #self.Q.assign(self.e1, self.E1) + outer(self.e2, self.E2) + outer(self.e3, self.E3)
            self.update_frame(w)            

            # Update boundary condition
            project(self.Ome_pref, self.V3, function=self.Ome_b)

            # Compute and update solution
            u = Function(self.W)
            
            
            
            solve(self.F_op == self.L, u, bcs=self.bc)

            F, M, x, Ome, sig, w = u.split()
            #x, Ome, sig, w = u.split()

            self.u_n.assign(u)
 
            self.t += self.dt
            self.check_for_nans()

            # Update frame 
            self.F.update(x, self.e1, self.e2, self.e3, Ome, sig, w, t = self.t)
            FS.append(self.F.clone())
            
            print(f't={f_s.format(self.t)}')

        FS = FrameSequenceFenics(FS)            
        
        return FS

    def check_for_nans(self):
        
        f_names = ['x', 'Ome', 'sig', 'w']
        
        for i, f in enumerate(split(self.u_n)):
            
            f = project(f, self.V3)
            
            if np.isnan(f.vector().sum()):
                    raise RuntimeError(f'{f_names[i]} contains NaNs')
                                    
        return


    
    
    
        
    



