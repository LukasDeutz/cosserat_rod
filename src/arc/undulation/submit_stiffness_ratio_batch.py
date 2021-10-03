# Build-in import
from itertools import product
import subprocess

N = 10
dt = 0.01
T = 0.1
c_arr = [1.0]

if __name__ == '__main__':

    for c in c_arr:
    
        # qsub submit_constant_controls N dt
        subprocess.call(f'qsub submit_stiffness_ratio.sh {N} {dt} {T} {c}', shell = True)