# Build-in import
from itertools import product
import subprocess

N_arr  = [50]
dt_arr = [1e-1]
T = 3.0

if __name__ == '__main__':

    for (N,dt) in product(N_arr, dt_arr):
    
        # qsub submit_constant_controls N dt
        subprocess.call(f'qsub bash_submit_constant_controls.sh {N} {dt} {T}', shell = True)

    
 