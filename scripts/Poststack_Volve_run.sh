#!/bin/sh
# Runner for Poststack_Volve.py 

# export OMP_NUM_THREADS=16; export MKL_NUM_THREADS=16; export NUMBA_NUM_THREADS=16; mpiexec -n 2 python Poststack_Volve.py
# export OMP_NUM_THREADS=8; export MKL_NUM_THREADS=8; export NUMBA_NUM_THREADS=8; mpiexec -n 4 python Poststack_Volve.py
# export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 8 python Poststack_Volve.py
export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=2; export NUMBA_NUM_THREADS=2; mpiexec -n 16 python Poststack_Volve.py