#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program.
Supports one-component hard-sphere and truncated Lennard-Jones
fluids incontact with homogenous planar surfaces, spherical
solutes and confined to a slit with homogeneous surfaces.

Created Februrary 2020. Last Update November 2021.
Author: Mary K. Coe
E-mail: m.k.coe@bristol.ac.uk

This program utilises FMT and supports the Rosenfeld,
White-Bear and White-Bear Mark II functionals.

For information on how to use this package please consult
the accompanying tutorials. Information on how the package
works can be found in Chapter 4 and Appendix A-C of the
following thesis:

M. K. Coe, Hydrophobicity Across Length Scales: The Role of
Surface Criticality, Ph.D. Thesis, University of Bristol (2021)
Available at: https://research-information.bris.ac.uk/ws/portalfiles/portal/304220732/Thesis_Mary_Coe.pdf

This module contains functions to implement the external
potentials available. See tutorials and minimisation module
for information on supported external potentials.

-------------------------------------------------------------------
Copyright 2021 Mary Coe

This file is part of cDFT.

cDFT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cDFT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with cDFT.  If not, see <https://www.gnu.org/licenses/>.
-------------------------------------------------------------------
"""

import numpy as np

############################## COMMON ####################################
def setup_HW(minimise):

    """
    Sets up a hard wall at the surface of the 'wall'
    """
    minimise.Vext = np.zeros(minimise.DFT.N)
    minimise.Vext[:minimise.NiW] = 500.0

    if minimise.slit:
        minimise.Vext[minimise.DFT.end:] = 500.0

############################# PLANAR #####################################

def setup_PLJ(minimise, right = False):

    """
    Adds an attractive LJ tail to the wall. This is derived by assuming the wall
    is uniform and made up of LJ particles. Integrating the potential felt by a
    single fluid particle due to the wall then yields this formula.
    """
    Vext = np.zeros(minimise.DFT.N)

    if right:
        sigma_wall = minimise.right_sigma_wall
    else:
        sigma_wall = minimise.sigma_wall

    Vext[minimise.rmask] = ((2.0/15.0)*\
                 np.power(sigma_wall/(minimise.r[minimise.rmask]-minimise.r[minimise.NiW-1]),9.0)-\
                 np.power(sigma_wall/(minimise.r[minimise.rmask]-minimise.r[minimise.NiW-1]),3.0))
    points = Vext>500; Vext[points] = 500;

    if right:
        Vext[minimise.rmask] = np.flip(Vext[minimise.rmask])
        Vext[:] *= minimise.right_epsilon_wall
    else:
        Vext[:] *= minimise.epsilon_wall

    minimise.Vext[minimise.rmask] += Vext[minimise.rmask]

def setup_PSLJ(minimise, right = False):

    """
    Adds an attractive LJ tail, shifted such that the minimum falls at the wall,
    to the wall. This is derived in the same way at PLJ. Shifted potentials are
    used as, close to the wall, the maximum potential must be capped (due to the
    asymptotic nature of a LJ potential), which can lead to steps in the density
    profile which would not occur in a real fluid.
    """

    Vext = np.zeros(minimise.DFT.N)

    if right:
        sigma_wall = minimise.right_sigma_wall
    else:
        sigma_wall = minimise.sigma_wall

    shifted_r = minimise.r[minimise.rmask] + np.power(0.4,1./6.)

    Vext[minimise.rmask] = ((2.0/15.0)*\
                 np.power(sigma_wall/(shifted_r[:]),9.0)-\
                 np.power(sigma_wall/(shifted_r[:]),3.0))

    if right:
        Vext[minimise.rmask] = np.flip(Vext[minimise.rmask])
        Vext[:] *= minimise.right_epsilon_wall
    else:
        Vext[:] *= minimise.epsilon_wall

    minimise.Vext[minimise.rmask] += Vext[minimise.rmask]


def setup_PWCALJ(minimise, right = False):

    """
    Adds an attractive LJ tail to the hard wall, using the WCA splitting.
    """
    Vext = np.zeros(minimise.DFT.N)

    if right:
        sigma_wall = minimise.right_sigma_wall
    else:
        sigma_wall = minimise.sigma_wall

    minimise.rminw = np.power(2.,1./6.)*sigma_wall
    minimise.zmask = minimise.r[:]  > minimise.rminw
    minimise.zmask[minimise.DFT.end:] = False
    minimise.izmask = np.invert(minimise.zmask)
    minimise.izmask[minimise.DFT.end:] = False; minimise.izmask[:minimise.NiW] = False


    Vext[minimise.zmask] = (2./15.)*np.power(sigma_wall,9.0)*np.power(minimise.r[minimise.zmask],-9.)\
                                    - np.power(sigma_wall,3.)*np.power(minimise.r[minimise.zmask],-3.)

    Vext[minimise.izmask] = (4./3.)*(np.power(sigma_wall/minimise.rminw,9.))
    Vext[minimise.izmask] -= (6./5.)*np.power(sigma_wall,9.)*minimise.r[minimise.izmask]\
                            /np.power(minimise.rminw,10.)
    Vext[minimise.izmask] += 3.*np.power(sigma_wall,3.)*minimise.r[minimise.izmask]\
                            /np.power(minimise.rminw,4.)
    Vext[minimise.izmask] -= 4.*np.power(sigma_wall/minimise.rminw,3.)
    Vext[minimise.izmask] += (3./2.)*minimise.rminw*minimise.rminw*minimise.r[minimise.izmask]\
                            /np.power(sigma_wall,3.)
    Vext[minimise.izmask] -= np.power(minimise.rminw/sigma_wall,3.)
    Vext[minimise.izmask] -= (1./2.)*np.power(minimise.r[minimise.izmask]/sigma_wall,3.)

    if right:
        Vext[minimise.rmask] = np.flip(Vext[minimise.rmask])
        Vext[:] *= minimise.right_epsilon_wall
    else:
        Vext[:] *= minimise.epsilon_wall

    minimise.Vext[minimise.rmask] += Vext[minimise.rmask]

############################# SPHERICAL ##################################

def setup_SLJ(minimise):

    """
    Adds an attractive LJ tail to the spherical solute. This is derived by
    assuming the wall is uniform and made up of LJ particles. Integrating the
    potential felt by a single fluid particle due to the wall then yields this
    formula.
    """

    rR_plus = (minimise.r[minimise.rmask]+minimise.Rs-minimise.DFT.dr)
    rR_minus = (minimise.r[minimise.rmask]-minimise.Rs+minimise.DFT.dr)

    rR_plus = 1.0/rR_plus; rR_minus = 1.0/rR_minus;

    minimise.Vext[minimise.rmask] = (2.0/15.0)*np.power(minimise.sigma_wall,9.0)*\
                (np.power(rR_minus,9.0)-np.power(rR_plus,9.0))
    minimise.Vext[minimise.rmask] += (3.0/20.0)*np.power(minimise.sigma_wall,9.0)*\
                (1.0/minimise.r[minimise.rmask])*(np.power(rR_plus,8.0)-np.power(rR_minus,8.0))
    minimise.Vext[minimise.rmask] += (np.power(minimise.sigma_wall,3.0)*(np.power(rR_plus,3.0)-np.power(rR_minus,3.0)))
    minimise.Vext[minimise.rmask] += (3.0/2.0)*np.power(minimise.sigma_wall,3.0)*(1.0/minimise.r[minimise.rmask])*\
                (np.power(rR_minus,2.0)-np.power(rR_plus,2.0))

    minimise.Vext *= minimise.epsilon_wall;
    points = minimise.Vext>500; minimise.Vext[points] = 500;

def setup_SSLJ(minimise):

    """
    Adds an attractive LJ tail to the spherical solute. This is derived by
    assuming the wall is uniform and made up of LJ particles. Integrating the
    potential felt by a single fluid particle due to the wall then yields this
    formula. To prevent ambiguity in the definition of the potential near the
    surface of the solute (due to the need for an upper cap), the potential is
    shifted to the wall.
    """

    shifted_r = minimise.r[minimise.NiW+1:minimise.DFT.N]
    rR_plus =  shifted_r + minimise.Rs
    rR_minus = shifted_r - minimise.Rs

    shifted_r = 1./shifted_r; rR_plus = 1./rR_plus; rR_minus = 1./rR_minus

    minimise.Vext[minimise.NiW+1:minimise.DFT.N] = (2.0/15.0)*np.power(minimise.sigma_wall,9.0)*\
                (np.power(rR_minus,9.0)-np.power(rR_plus,9.0))
    minimise.Vext[minimise.NiW+1:minimise.DFT.N] += (3.0/20.0)*np.power(minimise.sigma_wall,9.0)*\
                (shifted_r[:])*(np.power(rR_plus,8.0)-np.power(rR_minus,8.0))
    minimise.Vext[minimise.NiW+1:minimise.DFT.N] += (np.power(minimise.sigma_wall,3.0)*(np.power(rR_plus,3.0)-np.power(rR_minus,3.0)))
    minimise.Vext[minimise.NiW+1:minimise.DFT.N] += (3.0/2.0)*np.power(minimise.sigma_wall,3.0)*(shifted_r[:])*\
                (np.power(rR_minus,2.0)-np.power(rR_plus,2.0))

    minimise.shift = np.argmin(minimise.Vext[:])

    minimise.Vext[minimise.NiW:minimise.DFT.padding] = minimise.Vext[minimise.shift:minimise.DFT.padding+minimise.shift-minimise.NiW]
    minimise.Vext[minimise.NiW:] *= minimise.epsilon_wall;

