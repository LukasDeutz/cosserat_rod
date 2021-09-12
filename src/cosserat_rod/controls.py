# Build-in imports
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

# Third party imports
import numpy as np
from fenics import *

# Local imports
from cosserat_rod.util import f2n, v2f, expand_numpy

CONTROL_KEYS = ['alpha', 'beta', 'gamma', 'mu']


class Controls(ABC):
    """
    The worm is controlled with 3 forces (controls) acting along the body; alpha, beta and gamma.
    """

    def __init__(
            self,
            alpha=None,
            beta=None,
            gamma=None,
            mu=None,
            worm: 'Worm' = None
    ):
        if worm is None:
            # If no worm object passed then require all controls to be defined
            assert all(abgm is not None for abgm in [alpha, beta, gamma, mu])
        else:
            # Otherwise, require no controls to be passed
            assert all(abgm is None for abgm in [alpha, beta, gamma, mu])
            alpha, beta, gamma, mu = self._init_parameters(worm)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self._check_shapes()
        

    @abstractmethod
    def _init_parameters(self, worm: 'Worm') -> Tuple:
        """
        Return alpha, beta, gamma in appropriate format.
        """
        pass

    @abstractmethod
    def _check_shapes(self):
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
            alpha: Function = None,
            beta: Function = None,
            gamma: Function = None,
            mu: Function = None,
            worm: 'Worm' = None,
    ):
        super().__init__(alpha, beta, gamma, mu, worm)

    def _init_parameters(self, worm: 'Worm') -> Tuple[Function, Function, Function, Function]:
        """
        Use default parameters as set in the base Worm instance.
        """
        alpha = v2f(val=worm.alpha_pref_default, fs=worm.V, name='alpha')
        beta  = v2f(val=worm.beta_pref_default, fs=worm.V, name='beta')
        gamma = v2f(val=worm.gamma_pref_default, fs=worm.Q, name='gamma')
        mu    = v2f(val=worm.mu_pref_default, fs=worm.V, name='mu')
        return alpha, beta, gamma, mu

    def _check_shapes(self):
        assert self.alpha.function_space() == self.beta.function_space(), 'Function spaces differ from alpha to beta'
        # todo: check gamma?
        # todo: check mu?

    def clone(self) -> 'ControlsFenics':
        V = self.alpha.function_space()
        Q = self.gamma.function_space()
        return ControlsFenics(
            alpha=project(self.alpha, V),
            beta=project(self.beta, V),
            gamma=project(self.gamma, Q),
            mu=project(self.mu, V)
        )

    def __eq__(self, other: 'ControlsFenics') -> bool:
        # Convert to numpy for equality check
        c1 = self.to_numpy()
        c2 = other.to_numpy()
        return c1 == c2

    def to_numpy(self) -> 'ControlsNumpy':
            return ControlsNumpy(
                alpha=f2n(self.alpha),
                beta=f2n(self.beta),
                gamma=f2n(self.gamma),
                mu=f2n(self.mu)
            )


class ControlsNumpy(Controls):
    def __init__(
            self,
            alpha: np.ndarray = None,
            beta: np.ndarray = None,
            gamma: np.ndarray = None,
            mu: np.ndarray = None,
            worm: 'Worm' = None,
    ):
        super().__init__(alpha, beta, gamma, mu, worm)

    def _init_parameters(self, worm: 'Worm') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Default forces to empty arrays.
        """
        alpha = np.zeros(worm.N)
        beta  = np.zeros(worm.N)
        gamma = np.zeros(worm.N - 1)
        mu    = np.zeros(worm.N - 1)
        
        return alpha, beta, gamma, mu

    def _check_shapes(self):
        assert self.alpha.shape == self.beta.shape
        assert self.alpha.shape[-1] == self.mu.shape[-1] + 1
        assert self.alpha.shape[-1] == self.gamma.shape[-1] + 1
        
    def clone(self) -> 'ControlsNumpy':
        return ControlsNumpy(
            alpha=self.alpha.copy(),
            beta=self.beta.copy(),
            gamma=self.gamma.copy(),
            mu=self.mu.copy()
        )

    def to_fenics(self, worm: 'Worm') -> ControlsFenics:
        """
        Convert to Fenics
        """
        return ControlsFenics(
            alpha=v2f(self.alpha, fs=worm.V, name='alpha'),
            beta=v2f(self.beta, fs=worm.V, name='beta'),
            gamma=v2f(self.gamma, fs=worm.Q, name='gamma'),
            mu=v2f(self.mu, fs=worm.Q, name='mu')
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
            alpha=None,
            beta=None,
            gamma=None,
            mu = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1
    ):
        if worm is None:
            # If no worm object passed then require all controls to be defined
            if controls is not None:
                # ..either with controls and no abgm
                assert all(abgm is None for abgm in [alpha, beta, gamma, mu])
                if type(controls) == list:
                    # Controls are given as a time-indexed list
                    controls = self._generate_sequence_from_list(controls)
                else:
                    # Controls are just a single example which will need expanding
                    controls = self._generate_sequence_from_controls(controls, n_timesteps)
            else:
                # ..or in component form, with no control list
                assert all(abgm is not None for abgm in [alpha, beta, gamma, mu])
                assert len(alpha) == len(beta) == len(gamma)
                controls = self._generate_sequence_from_components(alpha, beta, gamma, mu)
        else:
            assert controls is None and all(abgm is None for abgm in [alpha, beta, gamma, mu])
            controls = self._generate_default_controls(worm, n_timesteps)

        self.controls = controls

    @abstractmethod
    def _generate_sequence_from_list(self, C: List[Controls]):
        pass

    @abstractmethod
    def _generate_sequence_from_controls(self, C: Controls, n_timesteps: int):
        pass

    @abstractmethod
    def _generate_sequence_from_components(self, alpha, beta, gamma):
        pass

    @abstractmethod
    def _generate_default_controls(self, worm: 'Worm', n_timesteps: int):
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
            alpha: List[Function] = None,
            beta: List[Function] = None,
            gamma: List[Function] = None,
            mu: List[Function] = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1
    ):
        super().__init__(controls, alpha, beta, gamma, mu, worm, n_timesteps)

    def _generate_sequence_from_list(self, C: List[ControlsFenics]) -> List[ControlsFenics]:
        # A ControlSequence in fenics is just a list of Controls objects
        return C

    def _generate_sequence_from_controls(self, C: ControlsFenics, n_timesteps: int):
        '''Controls are just a single example which will need expanding'''
        args = {k: getattr(C, k) for k in CONTROL_KEYS}
        Cs = [
            ControlsFenics(**args)
            for _ in range(n_timesteps)
        ]
        return Cs

    def _generate_sequence_from_components(
            self,
            alpha: List[Function],
            beta: List[Function],
            gamma: List[Function]
) -> List[ControlsFenics]:
        '''Controls are in component form, with no control list'''
        Cs = [
            ControlsFenics(
                alpha=alpha[t],
                beta=beta[t],
                gamma=gamma[t]
            )
            for t in range(len(alpha))
        ]
        return Cs

    def _generate_default_controls(self, worm: 'Worm', n_timesteps: int):
        '''Generate default controls if worm is None'''
        Cs = [
            ControlsFenics(worm=worm)
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
            alpha: np.ndarray = None,
            beta: np.ndarray = None,
            gamma: np.ndarray = None,
            mu: np.ndarray = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1
    ):
        super().__init__(controls, alpha, beta, gamma, mu, worm, n_timesteps)

    def _generate_sequence_from_list(self, C: List[ControlsNumpy]) -> dict:
        n_timesteps = len(C)
        return {
            k: np.stack([getattr(C[t], k) for t in range(n_timesteps)])
            for k in CONTROL_KEYS
        }

    def _generate_sequence_from_controls(self, C: ControlsNumpy, n_timesteps: int) -> dict:
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

    def _generate_default_controls(self, worm: 'Worm', n_timesteps: int) -> dict:
        C = ControlsNumpy(worm=worm)
        Cs = self._generate_sequence_from_controls(C, n_timesteps)
        return Cs

    def to_fenics(self, worm: 'Worm') -> ControlSequenceFenics:
        CSF = [
            self[t].to_fenics(worm)
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
