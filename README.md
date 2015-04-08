#Sparse MultiFrontal Package: A Fast Sparse Matrix Factorization Package 

This software is a sparse multifrontal solver with fast factorization capabilty. It uses Hierarchically off-diagonal low-rank (HODLR) structures to approximate the frontal matrices and arrives at a fast solver that performs extremely well as a GMRES preconditioner. In addition, it provides both implicit and explicit conventional LU factorization capabilities.

####Author :  

Amirhossein Aminfar: amirhossein.aminfar@gmail.com

####Citation:

If you use the implementation or any part of the implementation in your work, kindly cite as follows:

####Articles:

@article{aminfar2015fastMF,

author={{A}minfar, {A}mirhossein and {A}mbikasaran, {S}ivaram and {D}arve, {E}ric},

title={A Fast Sparse Solver with Applications to Finite-Element Matrices},

journal={arXiv:1410.2697},

year={2015}

}


@article{aminfar2014fast,

author={{A}minfar, {A}mirhossein and {A}mbikasaran, {S}ivaram and {D}arve, {E}ric},

title={A Fast Block Low-Rank Dense Solver with Applications to Finite-Element Matrices},

journal={arXiv:1403.5337},

year={2014}

}

####Code

@MISC{aminfar2015fastMFCode,

author = {{A}minfar, {A}mirhossein},

title = {A fast sparse solver with HODLR compression},

howpublished = {https://github.com/amiraa127/Sparse_MultiFrontal},

year = {2015}

}


####Version 1.00

Date: April 8th, 2015

Copyleft 2015: Amirhossein Aminfar 

Developed by Amirhossein Aminfar

####License


This program is free software; you can redistribute it and/or modify it under the terms of MPL2 license. The Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

####Build
This package is has the following dependencies:
1. [Eigen] (http://eigen.tuxfamily.org/index.php?title=Main_Page) C++ library for all matrix manipulations.

2. [SOTCH] (http://www.labri.fr/perso/pelegrin/scotch/) for nested dissection and matrix reordering.

3. [HODLR Package] (https://github.com/amiraa127/Dense_HODLR) for the fast multifrontal solver. After downloading the package, change the line `set(HODLR_Path /path/to/files/)` in `CMakeLists.txt` to the HODLR package directory.

4. [CPP Unit] (http://sourceforge.net/projects/cppunit/) for unit testing.

The easiest way to build the library is to use [CMake](http://www.cmake.org).
Go to the project directory and run:

```
mkdir build
cd build
cmake ../
make
```

####Documentation
Coming soon.....