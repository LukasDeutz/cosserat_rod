from typing import Union

import numpy as np

class ModelParameters:
    """
    Material parameters.
    """
    def __init__(
            self,
            external_force: str = 'linear_drag',
            K: Union[np.ndarray, float] = np.identity(3),
            K_rot: np.ndarray = np.identity(3),
            B: np.ndarray = np.identity(3),
            B_ast: np.ndarray = np.zeros((3,3)),
            S: np.ndarray = np.identity(3),
            S_ast: np.ndarray = np.zeros((3,3)),
            bc: bool = False,
            inertia: bool = False,
            rho: float = 1.0,
            J: np.ndarray = np.identity(3)
    ):
        """
        external_force: force exerted on the worm by the fluid, can be set to linear_drag or resistive_force
        K: External force amplitude 
        K_rot: The external moment.
        B: Bending/twist stiffness matrix 
        B_ast: Bending/twist viscosity matrix
        S: Shear/stretch stiffness matrix
        S_ast: Shear/stretch viscosity matrix 
        inertia: if True, then the inertia terms are included into the model
        rho: Mass-Density
        """
        
        self.external_force = external_force

        assert external_force in ['linear_drag', 'resistive_force']

        if self.external_force == 'resistive_force':
            if not type(K) == float:
                K = 40.              
            assert K > 0
                            
        self.K = K
        self.K_rot = K_rot
        self.B = B
        self.B_ast = B_ast
        self.S = S
        self.S_ast = S_ast

        self.bc = bc
        
        # Inertia
        self.inertia = inertia                        
        self.rho = rho
        self.J = J
                    
        assert np.all(np.diag(self.K_rot) >= 0)
        assert np.all(np.diag(self.B) > 0)
        assert np.all(np.diag(self.B_ast) >= 0)
        assert np.all(np.diag(self.S) > 0)
        assert np.all(np.diag(self.S_ast) >= 0)
        assert np.all(np.diag(self.J) >= 0)        
        assert rho >= 0
        



