# Third party imports
from abc import ABC, abstractmethod
from typing import Tuple, List 
import numpy as np
from fenics import *

# Local imports
from .util import f2n

FRAME_KEYS = ['x', 'e1', 'e2', 'e3', 'Omega', 'sigma', 'w', 'F', 'M']
FRAME_COMPONENT_KEYS = ['e1', 'e2', 'e3']

class Frame(ABC):

    def __init__(
            self,
            x=None,
            e1=None,
            e2=None,
            e3=None,
            Omega=None,
            sigma=None,
            w=None,
            F=None,
            M=None,
            t=None):

        # TODO checks
        self.x = x
            
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        
        self.Omega = Omega
        self.sigma = sigma
                
        self.w = w
        self.F = F 
        self.M = M

        self.t = t
            
        return
    
    @abstractmethod
    def clone(self) -> 'Frame':
        pass


class FrameFenics(Frame):
    
    def __init__(
            self,
            #spaces: list[FunctionSpace],
            x: Function = None,
            e1: Function = None,
            e2: Function = None,
            e3: Function = None,
            Omega: Function = None,
            sigma: Function = None,
            w=None,
            F=None,
            M=None,
            t: float = None):
        
        super().__init__(x, e1, e2, e3, Omega, sigma, w, F, M, t)
                    
    def clone(self) -> 'FrameFenics':
        
        kwargs = {}

        for key in FRAME_KEYS:                        
            if getattr(self, key) is None:
                kwargs[key] = None
            else:
                kwargs[key] = getattr(self, key).copy(deepcopy=True)

        kwargs['t'] = getattr(self, 't')
                                        
        return FrameFenics(**kwargs)
            
    def update(self, x, e1, e2, e3, Omega, sigma, w, F = None, M = None, t = None):
                
        self.x = x
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.Omega = Omega
        self.sigma = sigma
        self.w = w
                
        # self.x.assign(x)
        # self.e1.assign(e1)
        # self.e2.assign(e2)
        # self.e3.assign(e3)
        # self.Omega.assign(Omega)
        # self.sigma.assign(sigma)
        # self.w.assign(w)
        
        # if F is None:
        #     self.F = F
        # else:
        #     self.F.assign(F)
        #
        # if M is None:
        #     self.M = M
        # else:
        #     self.M.assign(M)
            
        self.t = t
                
        return 
             
    def to_numpy(self):
                
        kwargs = {}
        
        for key in FRAME_KEYS:                        
            if getattr(self, key) is None:
                kwargs[key] = None
            else:                
                kwargs[key] = f2n(getattr(self, key)) #, key)

        kwargs['t'] = getattr(self, 't')
                            
        return FrameNumpy(**kwargs)
                                
class FrameNumpy(Frame):
    
    def __init__(
        self,
        x: np.array = None,
        e1: np.array = None,
        e2: np.array = None,
        e3: np.array = None,
        Omega: np.array = None,
        sigma: np.array = None,
        w: np.array = None,
        F: np.array = None,
        M: np.array = None,
        t = None):
    
        super().__init__(x, e1, e2, e3, Omega, sigma, w, F, M, t)
        
    def clone(self) -> 'FrameNumpy':
        
        kwargs = {}
        
        for key in FRAME_KEYS:                        
            if getattr(self, key) is None:
                kwargs[key] = None
            else:
                kwargs[key] = getattr(self, key).copy()

        kwargs['t'] = getattr(self, 't')
                    
        return FrameNumpy(**kwargs)
                
    def get_range(self):
        mins = self.x.min(axis=1)
        maxs = self.x.max(axis=1)
        return mins, maxs

    def get_bounding_box(self, zoom=1) -> Tuple[np.ndarray, np.ndarray]:
        mins, maxs = self.get_range()
        max_range = max(maxs - mins)
        means = mins + (maxs - mins) / 2
        mins = means - max_range / 2 / zoom
        maxs = means + max_range / 2 / zoom
        return mins, maxs

    def get_worm_length(self) -> float:
        return np.linalg.norm(self.x[:, :-1] - self.x[:, 1:], axis=0).sum()

    def __eq__(self, other: 'FrameNumpy') -> bool:
        return all(
            np.allclose(getattr(self, k), getattr(other, k))
            for k in FRAME_KEYS
        )


class FrameSequence(ABC):
    def __init__(
            self,
            frames: List[Frame] = None,
            x=None,
            e1=None,
            e2=None,
            e3=None,
            Omega=None,
            sigma=None,
            w=None,
            M=None,
            F=None,
            model_parameters=None
    ):
        # Can't instantiate with nothing!
        assert not all(v is None for v in [frames, x])
                
        if frames is not None:
            # Build sequence from a list of frames
            frames = self._generate_sequence_from_list(frames)
        else:
            # Build sequence from components - at a minimum this must include x
            assert x is not None
            frames = self._generate_sequence_from_components(x, e1, e2, e3, Omega, sigma, w, M, F)

        self.frames = frames
        self.model_parameters = model_parameters

    @abstractmethod
    def _generate_sequence_from_list(self, frames: List[Frame]):
        pass

    @abstractmethod
    def _generate_sequence_from_components(self, x, e0, e1, e2, Omega, sigma, w, M, F):
        pass

    @abstractmethod
    def clone(self) -> 'FrameSequence':
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, i) -> Frame:
        pass

    @abstractmethod
    def __eq__(self, other: 'FrameSequence') -> bool:
        pass

    @property
    def n_timesteps(self) -> int:
        return len(self)

class FrameSequenceFenics(FrameSequence):    

    def __init__(
            self,
            frames: List[FrameFenics] = None,
            x: List[Function] = None,
            e1: List[Function] = None,
            e2: List[Function] = None,
            e3: List[Function] = None,
            Omega: List[Function] = None,
            sigma: List[Function] = None,
            model_parameters: 'MaterialParameters' = None):
                                          
        super().__init__(frames, x, e1, e2, e3, Omega, sigma, model_parameters)
        
        return
    
    
    def _generate_sequence_from_list(self, frames: List[FrameFenics]) -> List[FrameFenics]:
        return frames

    def _generate_sequence_from_components(
            self, 
            x: List[Function],
            e1: List[Function],
            e2: List[Function],
            e3: List[Function],
            Omega: List[Function],
            sigma: List[Function],
            w: List[Function],
            F: List[Function],
            M: List[Function]) -> List[FrameFenics]:
        
        n_timesteps = len(x)
                
        frames = [
            FrameFenics(
                x=x[t], 
                e1=e1[t], 
                e2=e2[t], 
                e3=e3[t], 
                Omega=Omega[t],
                sigma=sigma[t],
                w=w[t],
                F=F[t],
                M=M[t])
            for t in range(n_timesteps)
        ]
        
    def clone(self) -> 'FrameSequence':
        
        return FrameSequenceFenics(
            frames=[
                self[t].clone()
                for t in range(self.n_timesteps)
            ]
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i) -> Frame:
        return self.frames[i]

    def __eq__(self, other: 'FrameSequence') -> bool:
        
        fs1 = self.to_numpy()
        fs2 = other.to_numpy()
        
        return fs1 == fs2

    def to_numpy(self) -> 'FrameSequenceNumpy':
        
        t_arr = [f.t for f in self.frames]
        
        return FrameSequenceNumpy(
            frames=[
                self[t].to_numpy()
                for t in range(self.n_timesteps)                
            ],
            model_parameters = self.model_parameters,
            t_arr = t_arr
        )


        
class FrameSequenceNumpy(FrameSequence):        
        
    def __init__(
            self,
            frames: List[FrameNumpy] = None,
            x: np.ndarray = None,
            e1: np.ndarray = None,
            e2: np.ndarray = None,
            e3: np.ndarray = None,
            Omega: np.ndarray = None,
            sigma: np.ndarray = None,
            w: np.ndarray = None,
            M: np.ndarray = None,
            F: np.ndarray = None,
            model_parameters: 'ModelParameters'= None,
            t_arr: np.array = None):
                
        super().__init__(frames, x, e1, e2, e3, Omega, sigma, model_parameters)
        
        self.t_arr = t_arr
        
    def _generate_sequence_from_list(self, frames: List[FrameNumpy]) -> dict:
        n_timesteps = len(frames)
        return {
            k: np.stack([getattr(frames[t], k) for t in range(n_timesteps)])
            for k in FRAME_KEYS
        }
        
    def _generate_sequence_from_components(
            self,
            x: np.ndarray = None,
            e1: np.ndarray = None,
            e2: np.ndarray = None,
            e3: np.ndarray = None,
            Omega: np.ndarray = None,
            sigma: np.ndarray = None,
            w: np.ndarray = None,
            M: np.ndarray = None,
            F: np.ndarray = None,
            
    ) -> dict:
        
        return {
            'x': x,
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'Omega': Omega,
            'sigma': sigma,
            'w': w,
            'M': M,
            'F': F
        }
        
    def clone(self) -> 'FrameSequenceNumpy':
        args = {
            k: self.frames[k].copy() if self.frames[k] is not None else None
            for k in FRAME_KEYS
        }
        return FrameSequenceNumpy(**args, 
                                  model_parameters=self.model_parameters,
                                  t_arr = self.t_arr)
        
    def __len__(self) -> int:
        return len(self.frames['x'])
        
    def __getitem__(self, i) -> FrameNumpy:
        args = {
            k: self.frames[k][i]
            for k in FRAME_KEYS if self.frames[k] is not None
        }
        return FrameNumpy(**args)

    def __getattr__(self, k):
        if k in FRAME_KEYS:
            return self.frames[k]
        else:
            raise AttributeError(f'Key: "{k}" not found.')

    def __eq__(self, other: 'FrameSequenceNumpy') -> bool:
        return all(
            np.allclose(getattr(self, k), getattr(other, k))
            for k in FRAME_KEYS
        )

    def get_range(self) -> Tuple[np.ndarray, np.ndarray]:
        # Get common scale
        mins = np.array([np.inf, np.inf, np.inf])
        maxs = np.array([-np.inf, -np.inf, -np.inf])
        for i in range(len(self)):
            f_min, f_max = self[i].get_range()
            mins = np.minimum(mins, f_min)
            maxs = np.maximum(maxs, f_max)
        return mins, maxs

    def get_bounding_box(self, zoom=1) -> Tuple[np.ndarray, np.ndarray]:
        
        mins, maxs = self.get_range()
        max_range = max(maxs - mins)
        means = mins + (maxs - mins) / 2
        mins = means - max_range / 2 / zoom
        maxs = means + max_range / 2 / zoom
        
        return mins, maxs
        
    
    
    
    
    
    