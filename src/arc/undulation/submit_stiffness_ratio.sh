#!/bin/bash
echo "$N $dt"
# Set current working directory, use current env vars and modules
#$ -cwd -V
# Email at the beginning and end of the job
#$ -m be
# Request 10G RAM
#$ -l h_vmem=10G
# Request 10 min of runtime
#$ -l h_rt=02:00:00

source ~/.bashrc

#Activate conda environment
conda activate cosserat_rod

# Run scripy
python stifness_ratio_arc.py $1 $2 $3 $4