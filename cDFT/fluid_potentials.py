#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program.
Supports one-component hard-sphere and truncated Lennard-Jones
fluids incontact with homogenous planar surfaces, spherical
solutes and confined to a slit with homogeneous surfaces.

Created April 2019. Last Update November 2021.

This program utilises FMT and supports the Rosenfeld,
White-Bear and White-Bear Mark II functionals.

For information on how to use this package please consult
the accompanying tutorials. Information on how the package
works can be found in Chapter 4 and Appendix A-C of the
following thesis (link available December 2021)

This module contains the functions required to implement
the supported fluid potentials. See tutorials and minimisation
module for information on supported fluid potentials.
"""

import numpy as np


def LJ_104(X):

    return (0.8*np.power(X,-10.0) - 2.0*np.power(X,-4.0))

########################### Truncated Lennard-Jones ##############################
def TLJ_mu(DFT):

    mu = np.power(DFT.rmin,3) + (4.0/3.0)*((1.0/np.power(DFT.cut_off,9.0))-(1.0/np.power(DFT.rmin,9.0))) -\
                  4.0*((1.0/np.power(DFT.cut_off,3.0))-(1.0/np.power(DFT.rmin,3.0)))
    mu *= -(4.0/3.0)*np.pi*DFT.bulk_density

    return mu

def TLJ_pressure(DFT):

    p = (DFT.rmin**3 + (4.0/3.0)*((1.0/DFT.cut_off**9.0)-(1.0/DFT.rmin**9.0)) -\
                  4.0*((1.0/DFT.cut_off**3.0)-(1.0/DFT.rmin**3.0)))
    p *= -(2.0/3.0)*np.pi*DFT.bulk_density*DFT.bulk_density

    return p

def rs_TLJ_const(minimise):

    potential = np.zeros(minimise.DFT.Nrc + 1)
    rmin2 = minimise.DFT.rmin*minimise.DFT.rmin
    rmin104 = LJ_104(minimise.DFT.rmin)
    rc104 = LJ_104(minimise.DFT.cut_off)

    r = np.fromiter(((i*minimise.DFT.dr) for i in range(minimise.DFT.Nrc+1)), float)

    armin = r>minimise.DFT.rmin; brmin = np.invert(armin)
    potential[armin] = LJ_104(r[armin])
    potential[brmin] = r[brmin]*r[brmin] - rmin2 + rmin104


    minimise.DFT.potential = np.zeros(2*minimise.DFT.Nrc+1)
    minimise.DFT.potential[:minimise.DFT.Nrc+1] = np.flip(potential[:]);
    minimise.DFT.potential[-minimise.DFT.Nrc:] = potential[1:]

    minimise.potential = np.zeros((minimise.DFT.end-minimise.NiW,2*minimise.DFT.Nrc+1))
    for i in range(0,minimise.DFT.end-minimise.NiW):

        lower = i + minimise.NiW-minimise.DFT.Nrc
        upper = i + minimise.NiW+minimise.DFT.Nrc +1

        rdash = minimise.r[lower:upper]
        npmask = minimise.r[i+minimise.NiW] + rdash > minimise.DFT.cut_off
        pmask = np.invert(npmask)
        zmask = minimise.r[i+minimise.NiW] + rdash > minimise.r[minimise.NiW]
        zmask = np.invert(zmask)
        npmask[zmask] = False; pmask[zmask] = False

        minimise.potential[i,pmask] = minimise.DFT.potential[pmask] - LJ_104(minimise.r[i] + rdash[pmask])
        minimise.potential[i,npmask] = minimise.DFT.potential[npmask] - rc104


    minimise.potential *= np.pi*minimise.DFT.dr

    minimise.potential[:,-1]*=3./8.
    minimise.potential[:,-2]*=7./6.
    minimise.potential[:,-3]*=23./24;
    minimise.potential[:,0]*=3./8.
    minimise.potential[:,1]*=7./6.
    minimise.potential[:,2]*=23./24;

    del minimise.DFT.potential; del potential; del armin; del brmin; del zmask;
    del rdash; del npmask; del pmask;

def fourier_TLJ_const(minimise):

    rmin2 = minimise.DFT.rmin*minimise.DFT.rmin
    rmin104 = LJ_104(minimise.DFT.rmin)
    rc104 = LJ_104(minimise.DFT.cut_off)

    r = np.fromiter(((i*minimise.DFT.dr) for i in range(minimise.DFT.Nrc+1)), float)

    armin = r>minimise.DFT.rmin; brmin = np.invert(armin)
    potential = np.zeros(minimise.DFT.Nrc+1)

    potential[armin] = LJ_104(r[armin]) - rc104
    potential[brmin] = r[brmin]*r[brmin] - rmin2 + rmin104 - rc104
    potential[:] *= np.pi*minimise.DFT.dr

    potential[minimise.DFT.Nrc]*=3./8.
    potential[minimise.DFT.Nrc-1]*=7./6.
    potential[minimise.DFT.Nrc-2]*=23./24;

    minimise.DFT.potential = np.zeros(minimise.DFT.N)
    minimise.DFT.potential[:minimise.DFT.Nrc+1] = potential[:];
    minimise.DFT.potential[-minimise.DFT.Nrc:] = np.flip(potential[1:])

    del armin; del brmin; del potential;


def setup_TLJ_fluid(DFT):


    DFT.mu += TLJ_mu(DFT)
    DFT.pressure += TLJ_pressure(DFT)

################################ PLANAR #########################################

def fourier_PTLJ(minimise):

     minimise.fcp[:] = minimise.frho[:]*minimise.fatt[:]
     minimise.ifft_cp()

def rs_PTLJ(minimise):

    minimise.DFT.cp[:] = 0.0
    potential = np.zeros(2*minimise.DFT.Nrc)
    potential[:minimise.DFT.Nrc] = minimise.DFT.potential[-minimise.DFT.Nrc:]
    potential[minimise.DFT.Nrc:] = minimise.DFT.potential[:minimise.DFT.Nrc]

    for ri in range(minimise.NiW,minimise.DFT.end):

        if ri-minimise.DFT.Nrc<0:
            lower = 0
        else:
            lower = ri-minimise.DFT.Nrc

        upper = ri + minimise.DFT.Nrc
        minimise.DFT.cp[ri] = np.sum(minimise.DFT.rho[lower:upper]*potential[:])

################################ SPHERICAL ######################################

def fourier_STLJ(minimise):


    minimise.fcp[:] = minimise.frho[:] * minimise.fatt[:]
    minimise.ifft_cp()
    minimise.DFT.cp[minimise.cmask] /= minimise.r[minimise.cmask]
    minimise.DFT.cp[np.invert(minimise.cmask)] = 0.0


def rs_STLJ(minimise):

    minimise.DFT.cp[:] = 0.0
    rho = minimise.r[:]*minimise.DFT.rho[:]
    rho[minimise.NiW] *= 0.5

    for i in range(minimise.NiW,minimise.DFT.end):


        upper = minimise.DFT.Nrc + i +1
        lower = i-minimise.DFT.Nrc
        minimise.DFT.cp[i] = np.sum(rho[lower:upper] * minimise.potential[i-minimise.NiW,:])

    minimise.DFT.cp[minimise.cmask] /= minimise.r[minimise.cmask]
    minimise.DFT.cp[np.invert(minimise.cmask)] = 0.0

############################# Full Lennard-Jones ##################################

def LJ_mu(DFT):

    """
    LJ chemical potential
    """

    mu = np.power(DFT.rmin,3) - (4.0/3.0)*(1.0/np.power(DFT.rmin,9.0)) +\
                  4.0*(1.0/np.power(DFT.rmin,3.0))
    mu *= -(4.0/3.0)*np.pi*DFT.bulk_density


    return mu

def LJ_pressure(DFT):

    """
    LJ pressure
    """

    p = -1.*np.power(DFT.rmin,3.0) + (4.0/3.0)*np.power(DFT.rmin,-9.0) -\
                4.*np.power(DFT.rmin,-3.0)
    p *= (2.0/3.0)*np.pi*DFT.bulk_density*DFT.bulk_density

    return p

def LJ_const(DFT):

    """
    This is the contribution to the LJ fluid potential which remains constant.
    """
    potential = np.zeros(DFT.N)

    rmin104 = LJ_104(DFT.rmin)

    for i in range(DFT.N):

        if i*DFT.dr <= DFT.rmin:
            potential[i] = (i*DFT.dr)*(i*DFT.dr) - DFT.rmin*DFT.rmin + rmin104

        else:
            potential[i] = LJ_104(i*DFT.dr)

    DFT.potential = potential[:]

def setup_LJ_fluid(DFT):

    """
    Sets up the system for a LJ potential.
    """

    LJ_const(DFT)
    DFT.catt_temp = np.zeros(DFT.N)
    DFT.mu += LJ_mu(DFT)
    DFT.pressure += LJ_pressure(DFT)

################################### PLANAR ########################################

def PLJ_bulk(z,L):

    """
    LJ potential felt at z due to a semi-infinite slab of LJ fluid extending
    from L.
    """

    return (4./45.)*np.power(L-z,-9.) -(2./3.)*np.power(L-z,-3.)


def PLJ_bulk_rmin(L,z,rmin):

    """
    LJ potential felt at z due to a semi-infinite slab of LJ fluid extending
    from L. This is used when r+rmin extends past the end of the system.
    The current system set up prevents this from being used.
    """

    LJ1 = -(1./3.)*(2.*np.power(rmin,3.)+np.power(L-z,3.))
    LJ2 = -0.8*np.power(rmin,-10.) + 2.*np.power(rmin,-4.)+rmin*rmin
    LJ2 *= (L-z)
    LJ3 = (8./9.)*np.power(rmin,-9.) -(8./3.)*np.power(rmin,-3.)


    LJ = LJ1 + LJ2 + LJ3

    return LJ

def Pbulk(minimise):

    """
    Calculates the contribution to the potential from a semi-infinite slab of
    LJ fluid located at L.
    """

    minimise.bulk = np.zeros(minimise.DFT.N); L = minimise.r[minimise.DFT.end]

    for j in range(minimise.DFT.end):

        if L-minimise.r[j] > minimise.DFT.rmin:
            minimise.bulk[j] = PLJ_bulk(minimise.r[j],L)
        else:
            minimise.bulk[j] = PLJ_bulk_rmin(L,minimise.r[j],minimise.DFT.rmin)

    minimise.bulk[:] *= minimise.DFT.bulk_density

def rs_PLJ(minimise):

    """
    Performs the planar convolution for a LJ fluid.
    """

    minimise.DFT.cp[:] = 0.0
    minimise.rho[:] = minimise.DFT.rho[:]
    minimise.rho[minimise.NiW]*=0.5

    pot = np.zeros(minimise.DFT.N);

    for i in range(minimise.NiW,minimise.DFT.end):

        pot[i:] = minimise.DFT.potential[:minimise.DFT.N-i]
        pot[:i] = np.flip(minimise.DFT.potential[:i])

        minimise.DFT.cp[i] = np.sum(minimise.rho[minimise.NiW:minimise.DFT.end] \
                              * pot[minimise.NiW:minimise.DFT.end]) * minimise.DFT.dr

    minimise.DFT.cp[minimise.NiW:minimise.DFT.end] += minimise.bulk[minimise.NiW:minimise.DFT.end]
    minimise.DFT.cp[minimise.NiW:minimise.DFT.end] *= np.pi


################################## SPHERICAL ######################################

def SLJ_bulk(r,L):

    """
    Calculates the potential felt at r due to a semi-infinite spherical shell
    of LJ fluid extending from L.
    """

    plus = r+L; minus = r-L;
    LJ1 = L*(np.power(minus,-9.)+np.power(plus,-9.)) +\
                (1./8.)*(np.power(plus,-8.)-np.power(minus,-8.))
    LJ1*=(-4./45.)

    LJ2 = L*(np.power(minus,-3.)+np.power(plus,-3.)) +\
            0.5*(np.power(plus,-2.)-np.power(minus,-2.))

    LJ2 *= (2./3.)

    LJ = LJ2+LJ1;

    return LJ

def SLJ_bulk_rmin(L,r,rmin):

    """
    Calculates the potential felt at r due to a semi-infinite spherical shell
    of LJ fluid extending from L. This is the case where r+rmin extends past L.
    The current system set up prevents this.
    """

    plus = r+rmin; rr = 2.*r + rmin

    rLminus = r-L; rLplus = r+L

    LJ1 = np.power(rLminus,3.)*L + np.power(rmin,3.)*plus
    LJ1 += 0.25*(np.power(rLminus,4.) - np.power(rmin,4.))
    LJ1 *= (1./3.)

    LJ2 = 0.8*np.power(rmin,-10.) - 2.*np.power(rmin,-4.) - rmin*rmin
    LJ2 *= 0.5*(plus*plus-L*L)

    LJ3 = (plus*np.power(rr,-9.)-L*np.power(rLplus,-9.)+0.125*(np.power(rr,-8.)-np.power(rLplus,-8.)))
    LJ3 *= (4./45.)

    LJ4 = L*np.power(rLplus,-3.) -plus*np.power(rr,-3.) + 0.5*(np.power(rLplus,-2.)-np.power(rr,-2.))
    LJ4 *= (2./3.)

    LJ5 = (4./45.)*(plus*(np.power(rmin,-9.)-np.power(rr,-9.)) +0.125*np.power(rmin,-8.) -\
           np.power(rr,-8.))

    LJ6 = (-2./3.)*(plus*(np.power(rmin,-3.)-np.power(rr,-3.)) + 0.5*(np.power(rmin,-2.) -\
                   np.power(rr,-2.)))

    LJ = LJ1+LJ2+LJ3+LJ4+LJ5+LJ6

    return LJ

def Sbulk(minimise):

    """
    Sets up the potential felt at each r due to a semi-infinite spherical shell
    of LJ fluid extending from L.
    """

    minimise.bulk = np.zeros(minimise.DFT.N); L = minimise.r[minimise.DFT.padding]

    for j in range(minimise.DFT.N):

        if minimise.r[j]+minimise.DFT.rmin <= L:
            minimise.bulk[j] = SLJ_bulk(minimise.r[j],L)
        else:
            minimise.bulk[j] = SLJ_bulk_rmin(L,minimise.r[j],minimise.DFT.rmin)

    minimise.bulk[:] *= minimise.DFT.bulk_density


def rs_SLJ(minimise):

    """
    Calculates the real space convolution for a LJ fluid.
    """

    minimise.DFT.cp[:] = 0.0
    minimise.rho[:] = minimise.DFT.rho[:]*minimise.r[:]
    minimise.rho[minimise.NiW]*=0.5

    pot = np.zeros(minimise.DFT.N);

    for i in range(minimise.NiW,minimise.DFT.padding):

        plus = LJ_104(minimise.r[i]+minimise.r[:])

        pot[i:] = minimise.DFT.potential[:minimise.DFT.N-i]
        pot[:i] = np.flip(minimise.DFT.potential[:i])

        pot[:] -= plus[:]

        minimise.DFT.catt_temp[i] = np.sum(minimise.rho[minimise.NiW:minimise.DFT.padding] \
                              * pot[minimise.NiW:minimise.DFT.padding]) * minimise.DFT.dr

    minimise.DFT.catt_temp[minimise.NiW:minimise.DFT.padding] += minimise.bulk[minimise.NiW:minimise.DFT.padding]
    minimise.DFT.catt_temp[minimise.NiW:minimise.DFT.padding] *= np.pi
    minimise.DFT.cp[minimise.NiW:minimise.DFT.padding] = \
        (minimise.DFT.catt_temp[minimise.NiW:minimise.DFT.padding])/minimise.r[minimise.NiW:minimise.DFT.padding]

    minimise.DFT.cp[np.invert(minimise.cmask)] = 0.0
