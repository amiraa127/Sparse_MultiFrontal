
#ifndef SMF_DOCUMENTATION_HPP
#define SMF_DOCUMENTATION_HPP

// This file contains no source code but only documentation.

/** \mainpage Sparse Multifrontal solver

This software is a sparse multifrontal solver with fast factorization capabilty. 
It uses Hierarchically off-diagonal low-rank (HODLR) structures to approximate 
the frontal matrices and arrives at a fast solver that performs extremely well 
as a GMRES preconditioner. In addition, it provides both implicit and explicit 
conventional LU factorization capabilities.

\authors Amirhossein Aminfar, (Simon Praetorius)
\version 1.0.0
\date April 8th, 2015
\copyright 2015 by Amirhossein Aminfar 

# License #
This program is free software; you can redistribute it and/or modify it under 
the terms of MPL2 license. The Source Code Form is subject to the terms of the 
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with 
this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Citation #
If you use the implementation or any part of the implementation in your work, 
kindly cite as follows:

## Articles ##
~~~~~~~~~~~~~~~~~~~~~~{.txt}
@article{SMF2015,
  author  = {{A}minfar, {A}mirhossein and {A}mbikasaran, {S}ivaram and {D}arve, {E}ric},
  title   = {A Fast Sparse Solver with Applications to Finite-Element Matrices},
  journal = {arXiv:1410.2697},
  year    = {2015}
}
@article{SMF2014,
  author  = {{A}minfar, {A}mirhossein and {A}mbikasaran, {S}ivaram and {D}arve, {E}ric},
  title   = {A Fast Block Low-Rank Dense Solver with Applications to Finite-Element Matrices},
  journal = {arXiv:1403.5337},
  year    = {2014}
}
~~~~~~~~~~~~~~~~~~~~~~

## Code ##
~~~~~~~~~~~~~~~~~~~~~~{.txt}
@MISC{SMFCode2015,
  author       = {{A}minfar, {A}mirhossein},
  title        = {A fast sparse solver with HODLR compression},
  howpublished = {https://github.com/amiraa127/Sparse_MultiFrontal},
  year         = {2015}
}
~~~~~~~~~~~~~~~~~~~~~~

# Build #
This package has the following dependencies:

- [CMake](http://www.cmake.org) The build system.
- [Eigen](http://eigen.tuxfamily.org) C++ library for all matrix manipulations. 
- [SOTCH](https://www.labri.fr/perso/pelegrin/scotch) for nested dissection and matrix reordering.
- [HODLR Package](https://github.com/amiraa127/Dense_HODLR) for the fast multifrontal solver.
- [CPP Unit](http://sourceforge.net/projects/cppunit) for unit testing. 
  (only used when **ENABLE_TESTING** is enabled)
- [Pastix](http://pastix.gforge.inria.fr) for alternative matrix reordering
  (only used when **ENABLE_PASTIX** is enabled)

The easiest way to build the library is to use CMake. Go to the project directory 
and run:

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
mkdir build
cd build
cmake -DCMAKE_BUILD_PREFIX=/path/to/install/dir ..
make && make install
~~~~~~~~~~~~~~~~~~~~~~~~~

## Build options ##
Some options can be passed to cmake, eather by using commandline arguments, 
like `-DENABLE_XXX`, or any gui, like ccmake, or cmake-gui.
- **ENABLE_TESTING** Compiles the tests and adds the target *test*, that can be 
  run with `make test` after compiling. This option requires *CPP Unit* to be 
  installed and found by cmake.
- **ENABLE_BENCHMAKRS** This options is only available, with also **ENABLE_TESTING**
  is enabled and compiles additionally a benchmark executable.
- **ENABLE_PASTIX** Provides an alternative graph reordering library to be linked
  against SMF.
  
## Compiler support ##
Tested with the following compilers:
- **GNU g++:** 4.4.7, 4.7.3, 5.0.1
- **Intel icc:** 15.0.2
- **CLang++:** 3.0.1
- **MS Visual Studio:** 12.0 2013

The compiler needs support for the c++11 standard.

## Library versions ##
Tested with the follwoing versions of the used libraries:
- Eigen 3.2.4 needs the patch `patch_Eigen_3.2.4.diff` applied to `Eigen/src/SparseCore/SparseBlock.h`
- Scotch 6.0.4
- CPP Unit 1.12.1
*/


// - \subpage intro

//-----------------------------------------------------------

/** \page intro Introduction

*/

/** 
 * \namespace smf 
 * \brief The main namespace where all classes and functions of the 
 * Sparse Multifront solver are defined.
 * 
 * \namespace iml
 * \brief This namespace holds iterative methods that can be used together with
 * the \ref smf::sparseMF solver.
 **/

/** 
 * \defgroup output Output and error functions.
 * \brief Functions and classes for message output and throwing errors.
 *
 * \defgroup timer Time measurement and profiling.
 * \brief Functions and classes for profiling the solvers.
 * 
 * \defgroup solver Linear solvers.
 * \brief Linear solvers based on LU factorization and iterative refinement.
 */


#endif // SMF_DOCUMENTATION_HPP
