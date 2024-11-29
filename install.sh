#!/bin/bash
# 
# Installer for PyLops 4 ways environment with Cupy and CUDA 11.8 and MPI4Py
# 
# Run: ./install.sh
# 
# M. Ravasi, 26/08/2021

echo 'Creating PyLops GPU+MPI environment'

# create conda env
export CONDA_ALWAYS_YES="true"
conda env create -f environment.yml
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pylops_4ways
echo 'Created and activated environment:' $(which python)

# install latest versions of pylops and pyproximal
pip install git+https://github.com/PyLops/pyproximal.git@dev
pip install git+https://github.com/PyLops/pylops-mpi.git
pip install git+https://github.com/PyLops/pylops.git@dev

# check cupy and pylops work as expected
echo 'Checking cupy version and running a command...'
python -c 'import cupy as cp; print(cp.__version__); cp.ones(10000)*10'
python -c 'import numpy as np; import pylops; print(pylops.__version__); pylops.Identity(10) * np.ones(10)'

echo 'Done!'

