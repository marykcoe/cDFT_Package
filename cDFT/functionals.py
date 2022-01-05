#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program.
Supports one-component hard-sphere and truncated Lennard-Jones
fluids incontact with homogenous planar surfaces, spherical
solutes and confined to a slit with homogeneous surfaces.

Created April 2019. Last Update November 2021.
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

This module contains the functions required to implement
the supported fluid potentials. See tutorials and minimisation
module for information on supported functionals.

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

# Common functions and constants
pi = np.pi
pi4 = pi*4.0; pi8 = pi*8.0; pi12 = pi*12.0; pi36 = pi*36.0;
pi6 = pi*6.0; pi24 = pi*24.0;

def calculate_dn0(DFT, nzp):

    """
    Calculates the n0 derivative. This is the same for the Rosenfeld, Whitebear
    and WhitebearII functionals.
    """

    return -np.log(DFT.n3neg[nzp])


def calculate_dn1(DFT,nzp):

    """
    Calculates n1 derivative. This is equivalent for the Rosenfeld and Whitebear
    functionals. WhitebearII uses this form, however contains an extra factor.
    """
    return DFT.n2[nzp]/DFT.n3neg[nzp]

def calculate_dn1v(DFT,nzp):

    """
    Calculates the n1 vector derivative.
    """
    return -1.0*DFT.n2v[nzp]/DFT.n3neg[nzp]

# Rosenfeld Functional specific functions
def Rosenfeld_free_energy(DFT):

    """
    Uses Rosenfeld's original functional to calculate the excess free energy density
    of the system.
    """

    zero_points = DFT.n3 == 0. ; nzp = DFT.n3>0;
    zero_points[DFT.padding:] = True; nzp[DFT.padding:] = False;

    DFT.n3neg[:] = 1.0-DFT.n3[:]
    free_energy = np.zeros(DFT.N)
    free_energy[nzp] = -1.0*DFT.n0[nzp]*np.log(DFT.n3neg[nzp]) + \
                (DFT.n1[nzp]*DFT.n2[nzp]-DFT.n1v[nzp]*DFT.n2v[nzp])/(DFT.n3neg[nzp]) + \
                        ((DFT.n2[nzp]**3) - 3.0*DFT.n2[nzp]*DFT.n2v[nzp]**2)/(24.0*np.pi*DFT.n3neg[nzp]**2)
    return free_energy[:]

def calculate_Rosenfeld_pressure(eta, Temp, R):
    """
    The Rosenfeld equation is derived from the Percus-Yevick equation of state
    hence the pressure is the Percus-Yevick pressure as given in Byrk 2003 eq 28
    """

    pressure = Temp*3.0*eta*(1+eta+eta**2)
    pressure /= (pi4*(R**3)*(1.0-eta)**3)
    return pressure

def calculate_Rosenfeld_chemical_potential(eta, Temp):
    """
    Calculates the excess chemical potential for the Rosenfeld functional.
    """

    mu = (14.0*eta - 13.0*eta*eta + 5.0*eta**3)/(2.0*(1-eta)**3)
    mu -= np.log(1.0-eta)

    return Temp*mu

def calculate_Rosenfeld_derivatives(DFT):

    """
    Calculates all the required combined derivatives for Rosenfeld functional cDFT.
    """
    # If n3 is 0, there is no point taking the derivative as it will lead to
    # nan. Instead, we first search through the n3 array to find its non-zero
    # indicies and perform operations only on these
    zero_points = DFT.n3 == 0. ; nzp = DFT.n3>0;
    zero_points[DFT.padding:] = True; nzp[DFT.padding:] = False;

    DFT.d2[zero_points] = 0.0; DFT.d3[zero_points] = 0.0; DFT.d2v[zero_points] = 0.0

    # Start by calculating constants used more than once
    DFT.n3neg[:] = 1.0 - DFT.n3[:]
    n3neg2 = np.zeros(DFT.N); n2v2 = np.zeros(DFT.N);
    n3neg2[nzp] = DFT.n3neg[nzp]**2; n2v2[nzp] = DFT.n2v[nzp]**2;

    # We combine the derivatives in order to reduce the number of
    # operations
    DFT.d2[nzp] = calculate_dn0(DFT,nzp) / (pi4*(DFT.R)**2)
    DFT.d2[nzp] += calculate_dn1(DFT,nzp) / (pi4*DFT.R)
    DFT.d2[nzp] += DFT.n1[nzp]/DFT.n3neg[nzp] + (DFT.n2[nzp]**2 - n2v2[nzp])/(pi8*n3neg2[nzp])

    DFT.d3[nzp] = DFT.n0[nzp]/DFT.n3neg[nzp] + (DFT.n1[nzp]*DFT.n2[nzp] - DFT.n1v[nzp]*DFT.n2v[nzp])/(n3neg2[nzp])
    DFT.d3[nzp] += (DFT.n2[nzp]**3 - 3*DFT.n2[nzp]*n2v2[nzp])/(pi12*DFT.n3neg[nzp]**3)

    DFT.d2v[nzp] = calculate_dn1v(DFT,nzp)/(pi4*DFT.R)
    DFT.d2v[nzp] -= (DFT.n1v[nzp]/DFT.n3neg[nzp] + DFT.n2[nzp]*DFT.n2v[nzp]/(pi4*n3neg2[nzp]))


# Whitebear functional specific functions

def Whitebear_free_energy(DFT):

    nzp = DFT.n3!=0; nzp[DFT.padding:] = False; nzp[:2*DFT.NiR] = False

    DFT.n3neg[:] = 1.0-DFT.n3[:]
    n3neg2 = DFT.n3neg[:]**2

    free_energy = np.zeros(DFT.N)
    free_energy[nzp] = -DFT.n0[nzp]*np.log(DFT.n3neg[nzp]) + \
                (DFT.n1[nzp]*DFT.n2[nzp] - DFT.n1v[nzp]*DFT.n2v[nzp])/(DFT.n3neg[nzp])
    free_energy[nzp] += (DFT.n2[nzp]**3 - 3.0*DFT.n2[nzp]*DFT.n2v[nzp]*DFT.n2v[nzp])*\
            (DFT.n3[nzp]+(n3neg2[nzp])*np.log(DFT.n3neg[nzp]))/(pi36*DFT.n3[nzp]*DFT.n3[nzp]*(n3neg2[nzp]))

    return free_energy[:]

def calculate_Whitebear_pressure(eta,Temp,R):

    """
    Returns the WB pressure.
    """

    pressure = Temp * (3.0 * eta) * (1 + eta + eta**2 - eta**3)
    pressure /=(pi4*(R**3)*(1-eta)**3)
    return pressure

def calculate_Whitebear_chemical_potential(eta, Temp):

     """
     Returns WB chemical potential as given in Roth's program. Need to
     confirm where this equation comes from.
     """

     mu = (8.0*eta - 9.0*eta**2 + 3.0*eta**3) /((1.0-eta)**3)

     return Temp*mu

def calculate_Whitebear_derivatives(DFT):

    # If n3 is 0, there is no point taking the derivative as it will lead to
    # nan. Instead, we first search through the n3 array to find its non-zero
    # indicies and perform operations only on these
    zero_points = DFT.n3 == 0. ; nzp = DFT.n3!=0;
    zero_points[DFT.padding:] = True; nzp[DFT.padding:] = False;

    DFT.d2[zero_points] = 0.0; DFT.d3[zero_points] = 0.0; DFT.d2v[zero_points] = 0.0

    # Start by calculating constants used more than once
    DFT.n3neg[:] = 1.0 - DFT.n3[:]
    n3neg2 = np.zeros(DFT.N); n2v2 = np.zeros(DFT.N); n32 = np.zeros(DFT.N);
    n3neg2[nzp] = DFT.n3neg[nzp]**2; n2v2[nzp] = DFT.n2v[nzp]**2; n32[nzp] = DFT.n3[nzp]**2;

    # We combine the derivatives in order to reduce the number of operations
    #
    DFT.d2[nzp] = calculate_dn0(DFT,nzp) / (pi4*(DFT.R)**2)
    DFT.d2[nzp] += calculate_dn1(DFT,nzp) / (pi4*DFT.R)
    DFT.d2[nzp] += DFT.n1[nzp]/DFT.n3neg[nzp]
    DFT.d2[nzp] += (DFT.n2[nzp]**2 - n2v2[nzp])* \
        (DFT.n3[nzp] + (n3neg2[nzp])*np.log(DFT.n3neg[nzp]))/(pi12*(DFT.n3[nzp]**2)*n3neg2[nzp])

    DFT.d3[nzp] = DFT.n0[nzp]/DFT.n3neg[nzp] + (DFT.n1[nzp]*DFT.n2[nzp] - DFT.n1v[nzp]*DFT.n2v[nzp])/(n3neg2[nzp])
    DFT.d3[nzp] += ((DFT.n2[nzp]**3 - 3.0*DFT.n2[nzp]*n2v2[nzp])/pi36) * \
        (((DFT.n3[nzp]*(5.0-DFT.n3[nzp])-2.0)/((n32[nzp])*DFT.n3neg[nzp]**3)) - \
           (2.0*np.log(DFT.n3neg[nzp]))/(DFT.n3[nzp]**3))

    DFT.d2v[nzp] = calculate_dn1v(DFT,nzp)/ (pi4*DFT.R)
    DFT.d2v[nzp] -= (DFT.n1v[nzp]/DFT.n3neg[nzp] + DFT.n2[nzp]*DFT.n2v[nzp] * \
        ((DFT.n3[nzp]+ (n3neg2[nzp])*np.log(DFT.n3neg[nzp]))/(pi6*(n32[nzp])*(n3neg2[nzp]))))

# WhitebearII functional specific functions

def calculate_WhitebearII_phi2(DFT,nzp):

    """
    Calculates the phi2 function in the WhitebearII functional
    """

    DFT.phi2[nzp] = 2.*DFT.n3[nzp] - DFT.n3[nzp]*DFT.n3[nzp] +\
                        2.*DFT.n3neg[nzp]*np.log(DFT.n3neg[nzp])
    DFT.phi2[nzp] /= DFT.n3[nzp]


def calculate_WhitebearII_phi3(DFT,nzp):

    """
    Calculates the phi2 function in the WhitebearII functional
    """

    DFT.phi3[nzp] = 2.*DFT.n3[nzp] - 3.*DFT.n3[nzp]*DFT.n3[nzp]
    DFT.phi3[nzp] += 2.*DFT.n3[nzp]*DFT.n3[nzp]*DFT.n3[nzp]
    DFT.phi3[nzp] += 2.*DFT.n3neg[nzp]*DFT.n3neg[nzp]*np.log(DFT.n3neg[nzp])
    DFT.phi3[nzp] /= (DFT.n3[nzp]*DFT.n3[nzp])

def WhitebearII_free_energy(DFT):

    nzp = DFT.n3!=0; nzp[DFT.padding:] = False; nzp[:2*DFT.NiR] = False

    free_energy = np.zeros(DFT.N)
    DFT.n3neg[nzp] = 1.0-DFT.n3[nzp]

    calculate_WhitebearII_phi2(DFT,nzp)
    calculate_WhitebearII_phi3(DFT,nzp)

    free_energy[nzp] = -1.0*DFT.n0[nzp]*np.log(DFT.n3neg[nzp])
    free_energy[nzp] += (DFT.n1[nzp]*DFT.n2[nzp] - DFT.n1v[nzp]*DFT.n2v[nzp]) *\
		(1+(1./3.)*DFT.phi2[nzp])/DFT.n3neg[nzp]
    free_energy[nzp] += (np.power(DFT.n2[nzp],3.) - 3.*DFT.n2[nzp]*np.power(DFT.n2v[nzp],2.)) *\
		(1-(1./3.)*DFT.phi3[nzp])/(pi24*DFT.n3neg[nzp]*DFT.n3neg[nzp])

    return free_energy


def calculate_WhitebearII_dphi2(DFT, nzp):

    """
    Calculates the phi2 function in the WhitebearII functional
    """

    DFT.dphi2[nzp] = -1. + 2.*(-1./DFT.n3[nzp] - \
						np.log(DFT.n3neg[nzp])/(DFT.n3[nzp]*DFT.n3[nzp]))


def calculate_WhitebearII_dphi3(DFT,nzp):

    """
    Calculates the phi2 function in the WhitebearII functional
    """

    DFT.dphi3[nzp] = 1. - (1./(DFT.n3[nzp]*DFT.n3[nzp]))
    DFT.dphi3[nzp] += 2.*(1./(DFT.n3[nzp]*DFT.n3[nzp]) - \
					1./(DFT.n3[nzp]*DFT.n3[nzp]*DFT.n3[nzp]))*np.log(DFT.n3neg[nzp])
    DFT.dphi3[nzp] -= (1.-2./DFT.n3[nzp] + 1./(DFT.n3[nzp]*DFT.n3[nzp]))/(DFT.n3neg[nzp])
    DFT.dphi3[nzp] *= 2.

def calculate_WhitebearII_derivatives(DFT):

    # Start by calculating the additional factors required for WBII
    zero_points = DFT.n3 == 0. ; nzp = DFT.n3>0;
    zero_points[DFT.padding:] = True; nzp[DFT.padding:] = False;

    DFT.d2[zero_points] = 0.0; DFT.d3[zero_points] = 0.0; DFT.d2v[zero_points] = 0.0

    DFT.n3neg[nzp] = 1.0-DFT.n3[nzp]

    calculate_WhitebearII_phi2(DFT,nzp)
    calculate_WhitebearII_phi3(DFT,nzp)
    calculate_WhitebearII_dphi2(DFT,nzp)
    calculate_WhitebearII_dphi3(DFT,nzp)

    DFT.d2[nzp] = (DFT.n1[nzp] + DFT.n2[nzp]/(pi4*DFT.R))*\
		(1 + (1./3.)*DFT.phi2[nzp])/DFT.n3neg[nzp]
    DFT.d2[nzp] += (1-(1./3.)*DFT.phi3[nzp])*(DFT.n2[nzp]*DFT.n2[nzp] - \
				  DFT.n2v[nzp]*DFT.n2v[nzp])/(pi8*DFT.n3neg[nzp]*DFT.n3neg[nzp])
    DFT.d2[nzp] -= np.log(DFT.n3neg[nzp])/(pi4*DFT.R*DFT.R)

    DFT.d2v[nzp] = -1.*(DFT.n1v[nzp] + DFT.n2v[nzp]/(pi4*DFT.R))*\
				(1.+(1./3.)*DFT.phi2[nzp])/DFT.n3neg[nzp]
    DFT.d2v[nzp] -= DFT.n2[nzp]*DFT.n2v[nzp]*(1-(1./3.)*\
					DFT.phi3[nzp])/(pi4*DFT.n3neg[nzp]*DFT.n3neg[nzp])

    DFT.d3[nzp] = DFT.n0[nzp]/DFT.n3neg[nzp]
    DFT.d3[nzp] += (DFT.n1[nzp]*DFT.n2[nzp] - DFT.n1v[nzp]*DFT.n2v[nzp])*\
		(((1.+(1./3.)*DFT.phi2[nzp])/(DFT.n3neg[nzp]*DFT.n3neg[nzp])) + \
				(1./3.)*DFT.dphi2[nzp]/DFT.n3neg[nzp])
    DFT.d3[nzp] += (1./pi24)*(np.power(DFT.n2[nzp],3.) -\
				3.*DFT.n2[nzp]*DFT.n2v[nzp]*DFT.n2v[nzp]) * \
		((2.*(1.-(1./3.)*DFT.phi3[nzp])/(np.power(DFT.n3neg[nzp],3.))) -\
		   (DFT.dphi3[nzp]/(3.*DFT.n3neg[nzp]*DFT.n3neg[nzp])))

if __name__ == "__main__":

    pass



