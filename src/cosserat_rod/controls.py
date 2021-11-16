# Build-in imports
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

# Third party imports
import numpy as np
from fenics import *

# Local imports
from cosserat_rod.util import f2n, v2f, expand_numpy

CONTROL_KEYS = ['Omega', 'sigma']

class Controls(ABC):
    """
    The rod is controlled with 3 forces (controls) acting along the body; alpha, beta and gamma.
    """

    def __init__(
            self,
            Omega = None,
            sigma = None,
            rod: 'Worm' = None,
        ):
        
        
        if rod is None:
            assert all(x is not None for x in [Omega, sigma])
        else:                    
            Omega, sigma = self._init_parameters(rod)
        
        self.Omega = Omega
        self.sigma = sigma
        
        #self._check_shapes()
        
        return
                
    @abstractmethod
    def _init_parameters(self, rod: 'Worm') -> Tuple:        
        pass

    @abstractmethod
    def clone(self) -> 'Controls':
        pass

    @abstractmethod
    def __eq__(self, other: 'Controls') -> bool:
        pass


class ControlsFenics(Controls):
    def __init__(
            self,
            Omega=None,
            sigma=None,
            rod=None
    ):
        super().__init__(Omega, sigma, rod)
        
                
    def _init_parameters(self, rod: 'Worm') -> Tuple[Function, Function, Function, Function]:
        """
        Use default parameters as set in the base Worm instance.
        """     
        Omega = Function(self.rod.func_space_dict['Omega'])
        sigma = Function(self.rod.func_space_dict['sigma'])
        
        Omega.assign(Expression(('0','0', '0'), degree = 1))
        sigma.assign(Expression(('0','0', '0'), degree = 1))
                                                                                                 
        return Omega, sigma

    def clone(self) -> 'ControlsFenics':
                                
        return ControlsFenics(
            Omega=self.Omega.copy(deepcopy=True),
            sigma=self.sigma.copy(deepcopy=True)
        )

    def __eq__(self, other: 'ControlsFenics') -> bool:
        # Convert to numpy for equality check
        c1 = self.to_numpy()
        c2 = other.to_numpy()
        return c1 == c2

    def to_numpy(self) -> 'ControlsNumpy':
                                                    
        return ControlsNumpy(
            Omega=f2n(self.Omega),
            sigma=f2n(self.sigma)
            )

class ControlsNumpy(Controls):
    
    def __init__(self,
            rod: 'Worm' = None,
            Omega=None,
            sigma = None
            
    ):
        super().__init__(Omega, sigma, rod)

        # require all controls to be defined


    def _init_parameters(self, rod: 'Worm') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Default forces to empty arrays.
        """
        Omega = np.zeros((3, rod.N))
        sigma = np.zeros((3, rod.N))
        
        return Omega, sigma

    def _check_shapes(self):
                
        assert self.Omega.shape == self.sigma.shape
        
    def clone(self) -> 'ControlsNumpy':
        return ControlsNumpy(
            Omega=self.Omega.copy(),
            sigma=self.sigma.copy()
        )

    def to_fenics(self, rod: 'Rod') -> ControlsFenics:
        """
        Convert to Fenics
        """
        return ControlsFenics(
            Omega=v2f(self.Omega, fs=rod.function_spaces['Omega']),
            sigma=v2f(self.sigma, fs=rod.function_spaces['sigma'])
        )

    def __eq__(self, other: 'ControlsNumpy') -> bool:
        return all(
            np.allclose(getattr(self, k), getattr(other, k))
            for k in CONTROL_KEYS
        )


class ControlSequence(ABC):
    def __init__(
            self,
            controls: Union[Controls, List[Controls]] = None,
            Omega=None,
            sigma=None,
            rod: 'Worm' = None,
            n_timesteps: int = 1
    ):
        if rod is None:
            # If no rod object passed then require all controls to be defined
            if controls is not None:
                # ..either with controls and no abgm
                assert all(x is None for x in [Omega, sigma])
                if type(controls) == list:
                    # Controls are given as a time-indexed list
                    controls = self._generate_sequence_from_list(controls)
                else:
                    # Controls are just a single example which will need expanding
                    controls = self._generate_sequence_from_control(controls, n_timesteps)
            else:
                # ..or in component form, with no control list
                assert all(x is not None for x in [Omega, sigma])
                assert len(Omega) == len(sigma)
                controls = self._generate_sequence_from_components(Omega, sigma)
        else:
            assert controls is None and all(x is None for x in [Omega, sigma])
            controls = self._generate_default_controls(rod, n_timesteps)

        self.controls = controls

    @abstractmethod
    def _generate_sequence_from_list(self, C: List[Controls]):
        pass

    @abstractmethod
    def _generate_sequence_from_control(self, C: Controls, n_timesteps: int):
        pass

    @abstractmethod
    def _generate_sequence_from_components(self, alpha, beta, gamma):
        pass

    @abstractmethod
    def _generate_default_controls(self, rod: 'Worm', n_timesteps: int):
        pass

    @abstractmethod
    def clone(self) -> 'ControlSequence':
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, i) -> Controls:
        pass

    @abstractmethod
    def __eq__(self, other: 'ControlSequence') -> bool:
        pass

    @property
    def n_timesteps(self) -> int:
        return len(self)


class ControlSequenceFenics(ControlSequence):
    def __init__(
            self,
            controls: Union[ControlsFenics, List[ControlsFenics]] = None,
            Omega: List[Function] = None,
            sigma: List[Function] = None,            
            rod: 'Worm' = None,
            n_timesteps: int = 1
    ):
        super().__init__(controls, Omega, sigma, rod, n_timesteps)

    def _generate_sequence_from_list(self, C: List[ControlsFenics]) -> List[ControlsFenics]:
        # A ControlSequence in fenics is just a list of Controls objects
        return C

    def _generate_sequence_from_control(self, C: ControlsFenics, n_timesteps: int):
        '''Controls are just a single example which will need expanding'''
        args = {k: getattr(C, k) for k in CONTROL_KEYS}
        Cs = [
            ControlsFenics(**args)
            for _ in range(n_timesteps)
        ]
        return Cs

    def _generate_sequence_from_components(
            self,
            Omega: List[Function],
            sigma: List[Function],
) -> List[ControlsFenics]:
        '''Controls are in component form, with no control list'''
        Cs = [
            ControlsFenics(
                Omega=Omega[t],
                sigma=sigma[t],
            )
            for t in range(len(Omega))
        ]
        return Cs

    def _generate_default_controls(self, rod: 'Worm', n_timesteps: int):
        '''Generate default controls if rod is None'''
        Cs = [
            ControlsFenics(rod=rod)
            for _ in range(n_timesteps)
        ]
        return Cs

    def clone(self) -> 'ControlSequenceFenics':
        controls = [C.clone() for C in self.controls]
        return ControlSequenceFenics(controls=controls)

    def __len__(self) -> int:
        return len(self.controls)

    def __getitem__(self, i) -> ControlsFenics:
        return self.controls[i]

    def __eq__(self, other: 'ControlSequenceFenics') -> bool:
        cs1 = self.to_numpy()
        cs2 = other.to_numpy()
        return cs1 == cs2

    def to_numpy(self) -> ControlsNumpy:
            return ControlSequenceNumpy(controls=[
                self[t].to_numpy()
                for t in range(len(self.controls))
            ])

class ControlSequenceNumpy(ControlSequence):
    def __init__(
            self,
            controls: Union[ControlsNumpy, List[ControlsNumpy]] = None,
            Omega: np.ndarray = None,
            sigma: np.ndarray = None,
            rod: 'Worm' = None,
            n_timesteps: int = 1
    ):
        super().__init__(controls, Omega, sigma, rod, n_timesteps)

    def _generate_sequence_from_list(self, C: List[ControlsNumpy]) -> dict:
        n_timesteps = len(C)
        return {
            k: np.stack([getattr(C[t], k) for t in range(n_timesteps)])
            for k in CONTROL_KEYS
        }

    def _generate_sequence_from_control(self, C: ControlsNumpy, n_timesteps: int) -> dict:
        # Expand controls across all timesteps
        Cs = {
            k: expand_numpy(getattr(C, k), n_timesteps)
            for k in CONTROL_KEYS
        }
        return Cs

    def _generate_sequence_from_components(
            self,
            alpha: np.ndarray,
            beta: np.ndarray,
            gamma: np.ndarray
    ) -> dict:
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
        }

    def _generate_default_controls(self, rod: 'Worm', n_timesteps: int) -> dict:
        C = ControlsNumpy(rod=rod)
        Cs = self._generate_sequence_from_controls(C, n_timesteps)
        return Cs

    def to_fenics(self, rod: 'Worm') -> ControlSequenceFenics:
        CSF = [
            self[t].to_fenics(rod)
            for t in range(self.n_timesteps)
        ]
        return ControlSequenceFenics(CSF)

    def clone(self) -> 'ControlSequenceNumpy':
        return ControlSequenceNumpy(
            alpha=self.alpha.copy(),
            beta=self.beta.copy(),
            gamma=self.gamma.copy(),
        )

    def __len__(self) -> int:
        return len(self.controls['alpha'])

    def __getitem__(self, i) -> ControlsNumpy:
        args = {k: self.controls[k][i] for k in CONTROL_KEYS}
        return ControlsNumpy(**args)

    def __getattr__(self, k):
        if k in CONTROL_KEYS:
            return self.controls[k]
        else:
            raise AttributeError(f'Key: "{k}" not found.')

    def __eq__(self, other: 'ControlSequenceNumpy') -> bool:
        return all(
            np.allclose(getattr(self, k), getattr(other, k))
            for k in CONTROL_KEYS
        )
