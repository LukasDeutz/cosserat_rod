# Build-in import
from itertools import product
import subprocess

N_arr  = [50, 100, 250, 500]
dt_arr = [1e-1, 1e-2, 1e-3, 1e-4]
T = 3.0

if __name__ == '__main__':

    for (N,dt) in product(N_arr, dt_arr):
    
        # qsub submit_constant_controls N dt
        subprocess.call(f'qsub submit_constant_controls.sh {N} {dt} {T}', shell = True)