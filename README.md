# PU-DWR Combustion
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10641103.svg)](https://doi.org/10.5281/zenodo.10641103)

This software provides a space-time finite element solver to the low Mach number combustion equations
with goal oriented adaptive mesh refinement.

* Note: This software is a modification of [DTM++/dwr-diffusion](https://github.com/dtm-project/dwr-diffusion).

## Equations
The combustion equations describe the reaction between a dimensionless temperature $\theta$ and a combustable species concentration $Y$:
```math
\displaylines{
\partial_t\theta -\Delta\theta = \omega(\theta,Y)\quad\text{ in }\Omega\times(0,T),\\
\partial_t Y -\frac{1}{Le}\Delta Y = -\omega(\theta,Y)\text{ in }\Omega\times(0,T).
}
```
The reaction itself is described by Arrhenius law
```math
\omega(\theta,Y)\coloneqq\frac{\beta}{2Le}Y\exp(\frac{\beta(\theta-1)}{1+\alpha(\theta-1)}),
```
with parameters
- Lewis Number $Le>0$
- gas expansion rate $\alpha > 0$
- dimensionless activation energy $\beta >0$

## Adaptivity
Goal oriented adaptivity is achieved by the dual-weighted residual method (DWR),
with a space-time partition-of-unity as described in 
[![](https://img.shields.io/badge/DOI-Springer-blue.svg)]( https://doi.org/10.1007/s10915-024-02485-6)

Please cite this paper if you used the space-time PU-DWR method of this software.
```
@article{thiele2024numerical,
  title={Numerical modeling and open-source implementation of variational partition-of-unity localizations 
  of space-time dual-weighted residual estimators for parabolic problems},
  author={Thiele, Jan Philipp and Wick, Thomas},
  journal={Journal of Scientific Computing},
  volume={99},
  number={1},
  pages={25},
  year={2024},
  publisher={Springer}
}
```

## Setup

### Dependencies
This software has the following dependencies which can be installed together using [candi](https://github.com/dealii/candi):
  * MUMPS
  * Trilinos
  * p4est
  * HDF5  
  * deal.II v9.3.0 at least, linked to the previous packages

### Configure and Build
The software is configured and build using CMake by calling the following commands in the root folder of the repository.
```
   cmake -S. -Bbuild --DEAL_II_DIR=<path_to_your_deal_installation
   cmake --build build
```
This sets up a `build` directory in which the executable will be located.
It can be called as a single process or with MPI
```
    ./build/pu-dwr-combustion input/default.prm # single process
    mpirun -n <numprocs> build/pu-dwr-combustion input/default.prm # MPI parallel
```

