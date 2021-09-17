#!/bin/bash
# Set current working directory, use current env vars and modules
#$ -cwd -V
# Email at the beginning and end of the job
#$ -m be
# Request 1G RAM
#$ -l h_vmem=1G
# Request 10 min of runtime
#$ -l h_rt=00:10:00

source ~/.bashrc

#Activate conda environment
conda activate cosserat_rod

# Run scripy
constant_controls.py