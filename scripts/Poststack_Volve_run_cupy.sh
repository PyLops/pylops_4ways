#!/bin/sh
# Runner for Poststack_Volve_cupy.py (set hard-limit memory usage per GPU to 7GB)

export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=16; export NUMBA_NUM_THREADS=16; export CUPY_GPU_MEMORY_LIMIT="7000000000"; mpiexec -n 2 python Poststack_Volve_cupy.py