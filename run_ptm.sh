#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=thin
#SBATCH --time=24:00:00
 
#Execute program located in $HOME
. ~/mambaforge/etc/profile.d/conda.sh
conda activate ptm

cd /projects/0/einf4446/Pranav/PovertyTrapModel
python -m cProfile -o profile_ptm_nowhile_500N_125T.out FinalModel.py  

