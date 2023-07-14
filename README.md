# README #

This README documents whatever steps are necessary to get the application
  pu-dwr-combustion
up and running.

### What is this repository for? ###

* Goal-oriented adaptivity for the low Mach number combustion equations

* This software is a modification of DTM++/dwr-diffusion which can be found 
  at https://github.com/dtm-project/dwr-diffusion

### How do I get set up? ###

* Dependencies
deal.II v9.3.0 at least, linked to Trilinos, p4est and HDF5 


* Configuration
```
   cmake -S. -Bbuild --DEAL_II_DIR=<path_to_your_deal_installation
   cmake --build build``
```
* Run (single process)
```
    ./build/pu-dwr-combustion input/default.prm
```
* Run (MPI parallel)
```
    mpirun -n <numprocs> build/pu-dwr-combustion input/default.prm
```

### Who do I talk to? ###

* Principial Author
    * Jan Philipp Thiele, M.Sc. (thiele@ifam.uni-hannover.de)
    
* Contributors (original dwr-diffusion)
    * Marius P. Bruchhaeuser (bruchhaeuser@hsu-hamburg.de)
    * Dr.-Ing. Dipl.-Ing. Uwe Koecher (koecher@hsu-hamburg.de, dtmproject@uwe.koecher.cc)


This software is adapted to the combustion equations and a partition-of-unity 
localization approach for dual-weighted residuals.

If you write scientific publication using results obtained by reusing parts
of pu-dwr-combustion, especially by reusing the partition-of-unity for your applications
please cite the following publication:

- J. P. Thiele, T. Wick: "Numerical modeling and open-source implementation
  of variational partition-of-unity localizations of space-time dual-weighted 
  residual estimators for parabolic problems, in preparation

Furthermore, if you reuse the
datastructures, algorithms and/or supporting parameter/data input/output
classes, please cite the following two publications:

- U. Koecher, M.P. Bruchhaeuser, M. Bause: "Efficient and scalable
  data structures and algorithms for goal-oriented adaptivity of space-time
  FEM codes", Software X, Vol. 10, 2019
  https://doi.org/10.1016/j.softx.2019.100239
  

- U. Koecher: "Variational space-time methods for the elastic wave equation
  and the diffusion equation", Ph.D. thesis,
  Department of Mechanical Engineering of the Helmut-Schmidt-University,
  University of the German Federal Armed Forces Hamburg, Germany, p. 1-188,
  urn:nbn:de:gbv:705-opus-31129, 2015. Open access via:
  http://edoc.sub.uni-hamburg.de/hsu/volltexte/2015/3112/

### Further references ###
 A description of the combustion equations as well as the test cases solved
 by the provided input files (default.prm and rod.prm) can be found in the
 following publication  
    
 - M. Schmich and B. Vexler. Adaptivity with Dynamic Meshes for Space-Time 
   Finite Element Discretizations of Parabolic Equations. 
   SIAM Journal on Scientific Compution, 30(1):369-393, Jan. 2008
   https://doi.org/10.1137/060670468
   

### License ###
Copyright (C) 2012-2023 by Jan Philipp Thiele and contributors


pu-dwr-combustion is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

pu-dwr-combustion is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public License
along with pu-dwr-combustion. If not, see <http://www.gnu.org/licenses/>.
Please see the file
        ./LICENSE
for details.
