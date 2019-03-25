#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program for Planar and Spherical Geometry.

Copyright Mary Coe m.k.coe@bristol.ac.uk

Created January 2019. Last Update January 2019.

This program uses the Fundamental Measure Theory (FMT) of Density Functional Theory (DFT) to
find the equilibrium density profile of a fluid subjected to an external potential due to a
wall. This wall can be planar or spherical. The program is largely based off the following 
papers:
    
    Roth R. 2010. J. Phys.:Condens. Matter 22 063102.
    Stewart M.C. 2006. Thesis (Ph.D) University of Bristol.
    Bryk P., Roth R., Mecke K.R., and Dietrich S. 2003. Phys. Rev. E. 68 031602
    
Full details of the theory and implementation of the program can be found in the supporting 
documentation (see github).


"""
"""
try:
    import cDFT.minimisation as DFT
except ImportError:
    raise ImportError('minimisation.py not found')

try:
    import cDFT.functionals as functionals
except ImportError:
    raise ImportError('functionals.py not found')

try:
    import cDFT.output as output
except ImportError:
    raise ImportError('output.py not found')


print("Successfully loaded cDFT package")


if __name__ == "__main__":
    
    import cDFT
    import functionals
    import output
    
"""