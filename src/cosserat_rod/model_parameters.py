import numpy as np

class ModelParameters:
    """
    Material parameters.
    """
    def __init__(
            self,
            linearize: bool = True,
            external_force: str = 'linear_drag',
            K: float = 40.,
            K_rot: 'matrix' = np.identity(3),
            B: 'matrix' = np.identity(3),
            B_ast: 'matrix' = np.zeros((3,3)),
            S: 'matrix' = np.identity(3),
            S_ast: 'matrix' = np.zeros((3,3))
            
    ):
        """
        K: The external force exerted on the worm by the fluid.
        K_rot: The external moment.
        A: The bending modulus.
        B: The bending viscosity.
        C: The twisting modulus.
        D: The twisting viscosity.
        """

        self.linearize = linearize
        self.external_force = external_force
        self.K = K
        self.K_rot = K_rot
        self.B = B
        self.B_ast = B_ast
        self.S = S
        self.S_ast = S_ast


        assert external_force in ['linear_drag', 'resistive_force']
        assert self.K > 0
        assert np.all(np.diag(self.K_rot) >= 0)
        assert np.all(np.diag(self.B) > 0)
        assert np.all(np.diag(self.B_ast) >= 0)
        assert np.all(np.diag(self.S) > 0)
        assert np.all(np.diag(self.S_ast) >= 0)


