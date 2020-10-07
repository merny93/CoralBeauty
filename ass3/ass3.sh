#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=64000
#SBATCH --time=06:00:00


module load gcc/9.3
module load fftw/3.3.8-gcc9
module load openblas/openblas_gcc9
module load libffi/3.3
module load python/3.8.2-gcc9



python3 ass3.py