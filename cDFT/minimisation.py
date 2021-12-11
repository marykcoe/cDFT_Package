#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program.
Supports one-component hard-sphere and truncated Lennard-Jones
fluids incontact with homogenous planar surfaces, spherical
solutes and confined to a slit with homogeneous surfaces.

Created January 2019. Last Update November 2021.
Author: Mary K. Coe
E-mail: m.k.coe@bristol.ac.uk

This program utilises FMT and supports the Rosenfeld,
White-Bear and White-Bear Mark II functionals.

For information on how to use this package please consult
the accompanying tutorials. Information on how the package
works can be found in Chapter 4 and Appendix A-C of the
following thesis (link available December 2021)

This module contains the objects and methods required to
perform cDFT to find the equilibrium density profile.

Supported fluid types are:
    Hard-Sphere Fluid (HS)
    Truncated Lennard-Jones Fluid (arbitrary truncation) (TLJ)

Supported Hard-Sphere functionals are:
    Rosenfeld (RF)
    White-Bear (WB)
    White-Bear Mark II (WBII)

Supported geometries are:
    Planar (planar)
    Spherical (spherical)
    Slit (slit)

Supported external potentials for planar geometry are:
    Hard Wall (HW)
    Lennard-Jones (LJ)
    Shifted Lennard-Jones (SLJ)
    WCA Lennard-Jones (WCALJ)

Supported external potentials for spherical geometry are:
    Hard Wall (HW)
    Lennard-Jones (LJ)
    Shifted Lennard-Jones (SLJ)

Supported external potentials for slit geometry:
    Hard Wall (HW)
    Shifted Lennard-Jones (SLJ)

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

import os
import sys
import datetime
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pyfftw as fft

import cDFT.functionals as functionals
import cDFT.measures as measure
import cDFT.fluid_potentials as fluid
import cDFT.external_potentials as ext
import cDFT.exceptions as exceptions


#Constants
pi = np.pi
pi2 = np.pi*2.0
pi4 = np.pi*4.0

def valid_alpha(alpha):

    """
    Checks if a valid value of the Picard mixing parameter (alpha) has
    been supplied.

    Args:
        alpha(float): Picard mixing parameter. Should be between 0.0
                      and 1.0.
    Returns:
        alpha(float): If the alpha supplied is valid, it is returned. If not,
                      0.1 is returned as default.
    """

    if (alpha<1.0) and (alpha>0.0):
        return alpha
    else:
        print(f'Invalid value of alpha (Picard mixing parameter) supplied.')
        print(f'alpha must be between 0.0 and 1.0.')
        print(f'alpha will be set to 0.1.')
        return 0.1

def valid_ng(ng):

    """
    Checks if a valid value of moves between Ng updates has been supplied.

    Args:
        ng(int): Number of moves to leave between Ng updates.
                 Must be greater than 3.
    Returns:
        ng(int): If the ng supplied is valid, it is returned. If not,
                 10 is returned as default.
    """

    if int(ng)>3:
        return int(ng)
    else:
        print(f'Invalid value of ng supplied.')
        print(f'The number of moves between Ng move updates must be at least 3.')
        print(f'Setting ng to 10.')
        return 10

class DFT:

    """
    The DFT class details parameters which are not geometry specific.
    Examples of these include fluid parameters and grid size.
    Also stores weighted densities and correlation functions
    as well as wiehgt functions themselves.
    """

    def __init__(self, bulk_density, temperature,  fluid_type,
                 functional='RF', L=50.0, dr=0.001, cut_off = 2.5):

        """
        Sets up grid for minimisation.
        Defines fluid parameers and corresponding functions.
        Defines necessary arrays for minimisation process and calculates
        fourier transformed weight functions/fluid potentials.

        Args:
            Required:
                bulk_density(float): Density of fluid in bulk
                T(int): Temperature
                fluid_type(string): Fluid-fluid interaction potential
                                    Options:
                                        HS = hard-sphere
                                        TLJ = truncated 12-6 Lennard-Jones

        Optional:
            functional(string): FMT functional to be used for hard-sphere
                                calculations
                                Options:
                                    RF = Rosenfeld
                                    WB = White-Bear
                                    WBII = White-Bear Mark II
                                Default is RF
            L(float): Distance over which to calculate density profile
                      Default is 50.0
            dr(float): Distance between grid points. Should be chosen
                       such that 1/dr = integer. Default is 0.001
            cut_off(float): Cut-off radius of interaction for TLJ fluid.
                            Default is 2.5

        Returns:
            None
        """

        # Copy of input parameters
        self.bulk_density = bulk_density
        self.fluid_type = fluid_type
        self.T = temperature
        self.L = L
        self.dr = dr
        self.cut_off = cut_off
        self.functional = functional

        # Calculate other required parameters
        self.R = 0.5            # Radius of a fluid particle
        self.sigma = 1.0        # Diameter of a fluid particle
        self.beta = 1.0/self.T
        self.eta = (np.pi/6.0)*bulk_density   # Packing fraction
        self.N = int(np.ceil(self.L/self.dr)) # Number of grid points
        self.NiR = int(self.R/dr) # Number of grid points within radius of particle

        # Sets up hard-sphere part of fluid dependent on functional
        if functional == 'RF':
           self.calculate_pressure = functionals.calculate_Rosenfeld_pressure
           self.calculate_chemical_potential = functionals.calculate_Rosenfeld_chemical_potential
           self.free_energy = functionals.Rosenfeld_free_energy
           self.calculate_derivatives = functionals.calculate_Rosenfeld_derivatives

        elif functional == 'WB':
            self.calculate_pressure = functionals.calculate_Whitebear_pressure
            self.calculate_chemical_potential = functionals.calculate_Whitebear_chemical_potential
            self.free_energy = functionals.Whitebear_free_energy
            self.calculate_derivatives = functionals.calculate_Whitebear_derivatives

        elif functional == 'WBII':
            # Note because only one-component systems are supported, WBII pressure
            # & mu == WB pressure & mu
            self.calculate_pressure = functionals.calculate_Whitebear_pressure
            self.calculate_chemical_potential = functionals.calculate_Whitebear_chemical_potential
            self.free_energy = functionals.WhitebearII_free_energy
            self.calculate_derivatives = functionals.calculate_WhitebearII_derivatives

        else:
            raise exceptions.UnsupportedFunctionalError(functional)
            sys.exit(1)

        # Calculate pressure and chemical potential
        self.pressure = self.calculate_pressure(self.eta, self.T, self.R)
        self.mu =self.calculate_chemical_potential(self.eta,self.T)

        # Sets up fluid potential
        # Sets up padding of grid dependent on fluid potential
        if fluid_type == 'HS':
            self.N = self.N + 8*self.NiR
            self.end = self.N - 4*self.NiR
            self.padding = self.N- 2*self.NiR

        elif fluid_type == 'TLJ':
            self.rmin = np.power(2.0,1.0/6.0)
            self.Nrc = int(np.floor(self.cut_off/self.dr))
            self.Nrmin = int(np.floor(self.rmin/self.dr))

            if self.Nrc>2*self.NiR:
                self.N = self.N + 4*self.Nrc
                self.end = self.N-2*self.Nrc
                self.padding = self.N-self.Nrc

            else:
                self.N = self.N + 8*self.NiR
                self.end = self.N - 4*self.NiR
                self.padding = self.N- 2*self.NiR

            fluid.setup_TLJ_fluid(self)

        else:
            raise exceptions.UnsupportedFluidError(fluid_type)
            sys.exit(1)

        # Sets up weight functions, weighted density and correlation arrays
        self.n2 = fft.empty_aligned(self.N, dtype='float64')
        self.n3 = fft.empty_aligned(self.N, dtype='float64')
        self.n2v = fft.empty_aligned(self.N, dtype='float64')
        self.n2[:] = 0.0; self.n3[:] = 0.0; self.n2v[:] = 0.0

        self.n3neg = np.zeros(self.N)
        self.n0 = np.zeros(self.N); self.n1 = np.zeros(self.N); self.n1v = np.zeros(self.N)

        self.d2 = fft.empty_aligned(self.N, dtype='float64')
        self.d3 = fft.empty_aligned(self.N, dtype='float64')
        self.d2v = fft.empty_aligned(self.N, dtype='float64')
        self.c2 = fft.empty_aligned(self.N, dtype='float64')
        self.c3 = fft.empty_aligned(self.N, dtype='float64')
        self.c2v = fft.empty_aligned(self.N, dtype='float64')
        self.c2v_dummy = fft.empty_aligned(self.N, dtype='float64')
        self.density = fft.empty_aligned(self.N, dtype='float64')
        self.old_density = fft.empty_aligned(self.N, dtype='float64')

        w2 = fft.empty_aligned(self.N, dtype='float64')
        w3 = fft.empty_aligned(self.N, dtype='float64')
        w2v = fft.empty_aligned(self.N, dtype='float64')

        self.fw2 = fft.empty_aligned(int(self.N//2)+1, dtype='complex128')
        self.fw3 = fft.empty_aligned(int(self.N//2)+1, dtype='complex128')
        self.fw2v = fft.empty_aligned(int(self.N//2)+1, dtype='complex128')

        if functional == 'WBII':
            self.phi2 = np.zeros(self.N); self.phi3 = np.zeros(self.N)
            self.dphi2 = fft.empty_aligned(self.N, dtype='float64')
            self.dphi3 = fft.empty_aligned(self.N, dtype='float64')

        # Calculates weight functions in real space and fourier transforms
        # for later use
        fft_weights = fft.FFTW(w2,self.fw2, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))

        w2[:] = 0.0; w3[:] = 0.0; w2v[:] = 0.0;

        for i in range(0,self.NiR+1):
            w2[i] = pi2*self.R*dr
            w3[i] = pi*(self.R*self.R-i*dr*i*dr)*dr
            w2v[i] = pi2*i*dr*dr

            if i>0:
                w2[-i] = w2[i]
                w3[-i] = w3[i]
                w2v[-i] = -1.0*w2v[i]

        # Modifying the weight functions slightly helps the numerics (Roth 2010)
        w2[self.NiR]*=3.0/8.0; w2[-self.NiR] *=3.0/8.0; w3[self.NiR]*=3.0/8.0;
        w3[-self.NiR]*=3.0/8.0; w2v[self.NiR]*=3.0/8.0; w2v[-self.NiR]*=3.0/8.0;

        w2[self.NiR-1]*=7.0/6.0; w2[-self.NiR+1]*=7.0/6.0; w3[self.NiR-1]*=7.0/6.0;
        w3[-self.NiR+1]*=7.0/6.0; w2v[self.NiR-1]*=7.0/6.0; w2v[-self.NiR+1]*=7.0/6.0;

        w2[self.NiR-2]*=23.0/24.0; w2[-self.NiR+2]*=23.0/24.0; w3[self.NiR-2]*=23.0/24.0;
        w3[-self.NiR+2]*=23.0/24.0; w2v[self.NiR-2]*=23.0/24.0; w2v[-self.NiR+2]*=23.0/24.0;

        fft_weights.execute()
        fft_weights.update_arrays(w3,self.fw3); fft_weights.execute();
        fft_weights.update_arrays(w2v,self.fw2v); fft_weights.execute();

        del fft_weights; del w2; del w2v; del w3;

    def update_state_point(self,bulk_density, T=None):

        """
        Updates the state point of the fluid, given by the bulk density and
        temperature.
        All other paramaters stay the same.

        Args:
            Required:
                bulk_density(float): Updated density of bulk fluid

            Optional:
                T(float): Updated temperature

        Returns:
            None
        """

        # Define new density and calculate new packing fraction
        self.bulk_density = bulk_density
        self.eta = (np.pi/6.0)*bulk_density

        # Define new temperature and calculate new inverse temperature
        if T is not None:
            self.T = T
            self.beta = 1./T

        # Update pressure and excess chemical potential
        self.pressure = self.calculate_pressure(self.eta,self.T,self.R)
        self.mu = self.calculate_chemical_potential(self.eta,self.T)
        if self.fluid_type == 'TLJ':
            fluid.setup_TLJ_fluid(self)

        # Reset arrays
        self.old_density[:] = 0.0
        self.density[:] = 0.0

    def copy_parameters(self,):
        """Returns DFT parameters as a dict."""

        params = {"bulk_density": self.bulk_density, "temperature": self.T,
                  "fluid_type": self.fluid_type, "functional": self.functional,
                  "L": self.L, "dr": self.dr, "cut_off": self.cut_off}
        return params

    def information(self,):
        """
        Prints information about the DFT object.

        Args:
            None
        Returns:
            None
        """

        if self.fluid_type == 'HS':
            print(f'Fluid type: hard-sphere')
        else:
            print(f'Fluid type: truncated Lennard-Jones fluid with '\
                  f'cut-off radius of interaction {self.cut_off} * '\
                  f'diameter of a fluid particle')

        print(f'Temperature: {self.T}')
        print(f'Density of bulk fluid: {self.bulk_density}')

        if self.functional == 'RF':
            print(f'Functional: Rosenfeld')
        elif self.functional == 'WB':
            print(f'Functional : White-Bear')
        else:
            print(f'Functional: White-Bear Mark II')


        print(f'Length of grid: {self.L} * diameter of a fluid particle')
        print(f'Distance between grid points: {self.dr} * '\
              f'diameter of a fluid particle')


class minimisation:

    """
    Base class for minimisation objects. Contains attributes and methods which
	are not geometry specific.
    """

    def __init__(self, cDFT, inplace=False):

        """
		Initialises arrays and fourier objects required for minimisation procedure.

		Args:
			Required:
				cDFT (DFT_obj):    DFT object to be minimised

			Optional:
				inplace (bool):   Defines whether to overwrite DFT object
				                  Default is false
		Returns:
			None

        """

        # Create deep or shallow copy of DFT object
        if not inplace:
            self.DFT = copy.deepcopy(cDFT)
        else:
            self.DFT = copy.copy(cDFT)

        # Printing flags
        self.called_contact_sum_rule = False

        # Density arrays
        self.density = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.frho = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')

        # Mask for updating density profile
        self.NiW = cDFT.N - cDFT.end
        self.rmask = np.empty(self.DFT.N,dtype=bool);
        self.rmask[:] = False; self.rmask[self.NiW:self.DFT.end] = True;

        # Ng update arrays
        self.gn = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.g01 = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.g02 = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.Ngd1 = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.Ngd2 = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.Ngdn = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.Ngd01 = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.Ngd02 = fft.empty_aligned(self.DFT.N, dtype = 'float64')

        self.gn[:] = 0.0; self.g01[:] = 0.0; self.g02[:] = 0.0; self.Ngdn[:] = 0.0;
        self.Ngd01[:] = 0.0; self.Ngd02[:] = 0.0; self.Ngd1[:] = 0.0;
        self.Ngd2[:] = 0.0;

        # Fourier weighted density arrays
        self.fn2 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fn3 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fn2v = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')

        # Fourier weighted density derivative arrays
        self.fd2 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fd3 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fd2v = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')

        # Fourier correlation arrays
        self.fc2 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fc3 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fc2v = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fc2v_dummy = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')

        if self.DFT.fluid_type != 'HS':
            self.DFT.cp = fft.empty_aligned(self.DFT.N, dtype = 'float64')


        # FFTW objects to perform the fourier transforms
        self.fft_rho = fft.FFTW(self.density,self.frho, direction = 'FFTW_FORWARD',
                                flags = ('FFTW_ESTIMATE',))

        self.ifft_n2 = fft.FFTW(self.fn2,self.DFT.n2, direction = 'FFTW_BACKWARD',
                                flags = ('FFTW_ESTIMATE',))
        self.ifft_n3 = fft.FFTW(self.fn3,self.DFT.n3, direction = 'FFTW_BACKWARD',
                                flags = ('FFTW_ESTIMATE',))
        self.ifft_n2v = fft.FFTW(self.fn2v,self.DFT.n2v, direction = 'FFTW_BACKWARD',
                                 flags = ('FFTW_ESTIMATE',))

        self.fft_d2 = fft.FFTW(self.DFT.d2,self.fd2, direction = 'FFTW_FORWARD',
                               flags = ('FFTW_ESTIMATE',))
        self.fft_d3 = fft.FFTW(self.DFT.d3,self.fd3, direction = 'FFTW_FORWARD',
                               flags = ('FFTW_ESTIMATE',))
        self.fft_d2v = fft.FFTW(self.DFT.d2v,self.fd2v, direction = 'FFTW_FORWARD',
                                flags = ('FFTW_ESTIMATE',))

        self.ifft_c2 = fft.FFTW(self.fc2,self.DFT.c2, direction = 'FFTW_BACKWARD',
                                flags = ('FFTW_ESTIMATE',))
        self.ifft_c3 = fft.FFTW(self.fc3,self.DFT.c3, direction = 'FFTW_BACKWARD',
                                flags = ('FFTW_ESTIMATE',))
        self.ifft_c2v = fft.FFTW(self.fc2v,self.DFT.c2v, direction = 'FFTW_BACKWARD',
                                 flags = ('FFTW_ESTIMATE',))
        self.ifft_c2v_dummy = fft.FFTW(self.fc2v_dummy, self.DFT.c2v_dummy,
                                       direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))

        # Set the initial devation to an arbritrary number greater than the tolerance
        self.dev = 1.0

        # Set up any output directories
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        # Set fail
        self.fail = False

        # Set whether equilibrium profile found
        self.equilibrium = False

    def update(self):

        """
        Updates density profile using Picard and Ng update procedures.

        Args:
            None
        Returns:
            None
        """

        # Calculate one-body direct correlation function
        self.corr = np.zeros(self.DFT.N); rho_new = np.zeros(self.DFT.N);
        self.corr[1:] = -1.0*(self.DFT.c2[1:] + self.DFT.c3[1:] + self.DFT.c2v[1:])

        if self.DFT.fluid_type != 'HS':
            self.corr[1:] -=  self.DFT.beta*self.DFT.cp[1:]

        # Calculate trial density profile
        rho_new[self.rmask] = self.DFT.bulk_density*np.exp(self.corr[self.rmask] + \
                                           self.DFT.beta*(self.DFT.mu-self.Vext[self.rmask]))

        # Save current density profile
        self.DFT.old_density[self.rmask] = self.DFT.density[self.rmask]

        # Check the new density profile is valid. If it is not, end the simulation
        # with a failure notice.
        resultsnan=np.isnan(rho_new); resultsinf = np.isinf(rho_new)
        if np.any(resultsnan) or np.any(resultsinf):
            self.simulation_failed(rho_new)
            self.fail = True

        # Update Ng update arrays
        if self.attempts > 1:
            self.g02[self.rmask] = self.g01[self.rmask]
            self.g01[self.rmask] = self.gn[self.rmask]
            self.gn[self.rmask] = rho_new[self.rmask]

            self.Ngd2[self.rmask] = self.Ngd1[self.rmask]
            self.Ngd1[self.rmask] = self.Ngdn[self.rmask]

            self.Ngdn[self.rmask] = self.gn[self.rmask] - self.DFT.density[self.rmask]
            self.Ngd01[self.rmask] = self.Ngdn[self.rmask] - self.Ngd1[self.rmask]
            self.Ngd02[self.rmask] = self.Ngdn[self.rmask] - self.Ngd2[self.rmask]

        elif self.attempts == 1:
            self.g01[self.rmask] = rho_new[self.rmask]
            self.Ngd1[self.rmask] = self.g01[self.rmask] - self.DFT.density[self.rmask]

        else:
            self.g02[self.rmask] = rho_new[self.rmask]
            self.Ngd2[self.rmask] = self.g02[self.rmask] - self.DFT.density[self.rmask]

        # Perform Ng update
        if self.dev<1e-3 and self.attempts%self.ng==0:

            ip0101 = np.sum(self.Ngd01[self.rmask]*self.Ngd01[self.rmask])*self.DFT.dr
            ip0202 = np.sum(self.Ngd02[self.rmask]*self.Ngd02[self.rmask])*self.DFT.dr
            ip0102 = np.sum(self.Ngd01[self.rmask]*self.Ngd02[self.rmask])*self.DFT.dr
            ipn01 = np.sum(self.Ngdn[self.rmask]*self.Ngd01[self.rmask])*self.DFT.dr
            ipn02 = np.sum(self.Ngdn[self.rmask]*self.Ngd02[self.rmask])*self.DFT.dr

            norm = ip0101*ip0202 - ip0102*ip0102
            norm = 1.0/norm
            a1 = norm*(ip0202*ipn01 - ip0102*ipn02); a2 = norm*(ip0101*ipn02-ip0102*ipn01);

            self.DFT.density[self.rmask] = (1.0-a1-a2)*self.gn[self.rmask] +a1*self.g01[self.rmask] +\
                                a2*self.g02[self.rmask]

        # Perform Picard update
        else:
            self.DFT.density[self.rmask] = (1.0 - self.alpha)*self.DFT.density[self.rmask] + \
                        self.alpha*rho_new[self.rmask]

        # Calculate deviation between new and old density profiles
        self.dev = max(abs(self.DFT.density[self.rmask]-self.DFT.old_density[self.rmask]))

    def minimise(self):

        """
        Iterative procedure to calculate the equilibrium density profile.
        The equilibrium density profile is found when the deviation between iteractions
        is less than 1e-12. The program exits early if the updating procedure fails,
        or if the maximum number of iteractions is reached.

        Args:
            None
        Returns:
            None
        """

        self.attempts = 0
        while  self.dev>1e-12 and self.attempts<10000000 and not self.fail:

            self.weighted_densities()
            self.correlation()
            self.update()
            self.attempts+=1

            if (self.attempts%1000 == 0) and (self.deriv is False):
                print(f'{self.attempts} complete. Deviation: {self.dev}\n')

        if self.deriv is False:
            if (self.attempts<10000000):
                print(f'Convergence achieved in {self.attempts} attempts.')

            else:
                print(f'Density profile failed to converge after {self.attempts} attempts.')

        if (self.attempts<10000000):
            self.equilibrium = True


    def output_simulation_data(self):

        """
        Writes system parameters and final profiles to file.

        Args:
            None
        Returns:
            None
        """

        # Works out the correct precision for the distance output.
        pres = 0; dr = self.DFT.dr;
        while dr<1:
            dr*=10; pres+=1;

        # Outputs parameters and appropriate density profiles
        with open(self.output_file_name, 'a') as out:

            out.write(f'Produced {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')

            # Fluid Details
            out.write(f'Fluid Type = {self.DFT.fluid_type}\n')
            if (self.DFT.fluid_type == 'TLJ'):
                out.write(f'cut off radius: {self.DFT.cut_off}sigma\n\n')

            # Wall Details
            if self.slit:
                out.write(f'Left Wall Type: {self.wall_type}\n')
                if self.wall_type != 'HW':
                    out.write(f'Left Wall Strength (epsilon_wall) = {self.epsilon_wall}\n')
                    out.write(f'Left Wall Sigma = {self.sigma_wall}\n')

                out.write(f'Right Wall Type: {self.right_wall_type}\n')
                if self.right_wall_type != 'HW':
                    out.write(f'Right Wall Strength (right_epsilon_wall) = {self.right_epsilon_wall}\n')
                    out.write(f'Right Wall Sigma = {self.right_sigma_wall}\n\n')
            else:
                out.write(f'Wall Type: {self.wall_type}\n')
                if self.wall_type != 'HW':
                    out.write(f'Wall Strength (epsilon_wall) = {self.epsilon_wall}\nWall Sigma = {self.sigma_wall}\n\n')

            out.write(f'Bulk Density = {self.DFT.bulk_density:.10f}\n')
            if self.compressibility:
                out.write(f'Bulk Compressibility = {self.bulk_compressibility:.12f}\n')
            if self.susceptibility:
                out.write(f'Bulk Susceptibility = {self.bulk_susceptibility:.12f}\n')
            out.write(f'R = {self.DFT.R}\nT = {self.DFT.T}\n')
            out.write(f'L = {self.DFT.L}sigma\nN = {self.DFT.N}\ndr = {self.DFT.dr}\n\nFunctional = {self.DFT.functional}\n')
            out.write(f'Pressure = {self.DFT.pressure:.12f}\nExcess Chemical Potential = {self.DFT.mu:.12f}\n')
            out.write(f'Chemical Potential = {self.DFT.mu + self.DFT.T*np.log(self.DFT.bulk_density):.12f}\n\n')
            out.write(f'Contact Density = {self.DFT.density[self.NiW]:.12f}\n')
            out.write(f'Convergence in {self.attempts} attempts.\n\n')

            if self.compressibility and self.susceptibility:
                out.write(f'i\tr\trho\t\trho/rho_b\tchi_mu\t\t')
                out.write(f'chi_mu\chi_mu_bulk\tchi_T\t\tchi_T/chi_T_bulk\n')
                for i in range(self.DFT.N):
                    out.write(f'{i}\t{self.r[i]:.{pres}f}\t{self.DFT.density[i]:.12f}\t')
                    out.write(f'{self.DFT.density[i]/self.DFT.bulk_density:.12f}\t')
                    out.write(f'{self.compressibility_profile[i]:.12f}\t')
                    out.write(f'{self.compressibility_profile[i]/self.bulk_compressibility:.12f}\t')
                    out.write(f'{self.susceptibility_profile[i]:.12f}\t')
                    out.write(f'{self.susceptibility_profile[i]/self.bulk_susceptibility:.12f}\n')

            elif self.compressibility:
                out.write(f'i\tr\trho\t\trho/rho_b\tchi_mu\t\tchi_mu\chi_mu_bulk\n')
                for i in range(self.DFT.N):
                    out.write(f'{i}\t{self.r[i]:.{pres}f}\t{self.DFT.density[i]:.12f}\t')
                    out.write(f'{self.DFT.density[i]/self.DFT.bulk_density:.12f}\t')
                    out.write(f'{self.compressibility_profile[i]:.12f}\t')
                    out.write(f'{self.compressibility_profile[i]/self.bulk_compressibility:.12f}\n')

            elif self.susceptibility:
                out.write(f'i\tr\trho\t\trho/rho_b\tchi_T\t\tchi_T/chi_T_bulk\n')
                for i in range(self.DFT.N):
                    out.write(f'{i}\t{self.r[i]:.{pres}f}\t{self.DFT.density[i]:.12f}\t')
                    out.write(f'{self.DFT.density[i]/self.DFT.bulk_density:.12f}\t')
                    out.write(f'{self.susceptibility_profile[i]:.12f}\t')
                    out.write(f'{self.susceptibility_profile[i]/self.bulk_susceptibility:.12f}\t')

            else:
                out.write(f'i\tr\trho\t\trho/rho_b\n')
                for i in range(self.DFT.N):
                    out.write(f'{i}\t{self.r[i]:.{pres}f}\t{self.DFT.density[i]:.12f}\t')
                    out.write(f'{self.DFT.density[i]/self.DFT.bulk_density:.12f}\n')

            out.write(f'\nSum Rules:\nAdsorption (Gamma) = {self.adsorp:.12f}\t-dgamma/dmu = {self.gamma_deriv:.12f}\n')
            out.write(f'Relative error = {self.error:.12f}\n')

    def simulation_failed(self,rho_new):

        """
        Outputs the current state of the system upon update failure.

        Args:
            rho_new(np.array(float)): The failed proposed density profile.
        Returns:
            None
        """

        with open(self.file_path + 'Failure_Data', 'w') as out:
            out.write(f'Produced {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
            out.write(f'Simulation failed at {self.attempts} attempts.\n')
            if self.Rs is None:
                out.write(f'\nPlanar Geometry.\n')
            else:
                out.write(f'Spherical Geometry.\nRs = {self.Rs}\n\n')

            out.write(f'Fluid Type = {self.DFT.fluid_type}\n')
            if (self.DFT.fluid_type == 'TLJ'):
                out.write(f'cut off radius: {self.DFT.cut_off}sigma\n\n')
            else:
                self.DFT.cp = np.zeros(self.DFT.N)

            # Wall Details
            if self.slit:
                out.write(f'Left Wall Type: {self.wall_type}\n')
                if self.wall_type != 'HW':
                    out.write(f'Left Wall Strength (epsilon_wall) = {self.epsilon_wall}\n')
                    out.write(f'Left Wall Sigma = {self.sigma_wall}\n')

                out.write(f'Right Wall Type: {self.right_wall_type}\n')
                if self.right_wall_type != 'HW':
                    out.write(f'Right Wall Strength = {self.right_epsilon_wall}\n')
                    out.write(f'Right Wall Sigma = {self.right_sigma_wall}\n\n')
            else:
                out.write(f'Wall Type: {self.wall_type}\n')
                if self.wall_type != 'HW':
                    out.write(f'Wall Strength (epsilon_wall) = {self.epsilon_wall}\nWall Sigma = {self.sigma_wall}\n\n')

            out.write(f'Bulk Density = {self.DFT.bulk_density:.6f}\nR = {self.DFT.R}\nT = {self.DFT.T}\n')
            out.write(f'L = {self.DFT.N}sigma\nN = {self.DFT.N}\ndr = {self.DFT.dr}\n\nFunctional = {self.DFT.functional}\n')
            out.write(f'Pressure = {self.DFT.pressure:.12f}\nExcess Chemical Potential = {self.DFT.mu:.12f}\n')
            out.write(f'Chemical Potential = {self.DFT.mu + self.DFT.T*np.log(self.DFT.bulk_density):.12f}\n\n')
            out.write(f'i\tr\trho_old\t\trho_new\tc2\tc3\tc2v\tcp\tn2\tn3\tn2v\n')
            for i in range(self.DFT.N):
                out.write(f'{i}\t{self.r[i]:.3f}\t{self.DFT.density[i]:.6f}\t{rho_new[i]:.6f}')
                out.write(f'\t{self.DFT.c2[i]:.6f}\t{self.DFT.c3[i]:.6f}\t{self.DFT.c2v[i]:.6f}\t')
                out.write(f'{self.DFT.cp[i]:.6f}\t{self.DFT.n2[i]:.6f}\t{self.DFT.n3[i]:.6f}\t{self.DFT.n2v[i]:.6f}\n')

    def equilibrium_profile(self,):
        """
        Returns grid points and equilibrium profiles in order
        grid points -> density profile -> local compressibility profile
        (if applicable) -> local thermal susceptibility profile (if applicable)
        """

        if self.equilibrium:
            if self.compressibility and self.susceptibility:
                return copy.deepcopy(self.r[self.NiW:self.DFT.end]), \
                        copy.deepcopy(self.DFT.density[self.NiW:self.DFT.end]),\
                        copy.deepcopy(self.compressibility_profile[self.NiW:self.DFT.end]),\
                        copy.deepcopy(self.susceptibility_profile[self.NiW:self.DFT.end])
            elif self.compressibility:
                return copy.deepcopy(self.r[self.NiW:self.DFT.end]), \
                            copy.deepcopy(self.DFT.density[self.NiW:self.DFT.end]),\
                            copy.deepcopy(self.compressibility_profile[self.NiW:self.DFT.end])
            elif self.susceptibility:
                return copy.deepcopy(self.r[self.NiW:self.DFT.end]), \
                            copy.deepcopy(self.DFT.density[self.NiW:self.DFT.end]),\
                            copy.deepcopy(self.susceptibility_profile[self.NiW:self.DFT.end])
            else:
                return copy.deepcopy(self.r[self.NiW:self.DFT.end]), \
                            copy.deepcopy(self.DFT.density[self.NiW:self.DFT.end])

        else:
            print(f'Equilibrium profile has not been found.')
            return None, None

    def adsorption_sum_rule(self,):
        """Prints results of adsorption sum rule"""

        if self.equilibrium:
            print(f'\n--------------------------------------------')
            print(f'Adsorption Sum Rule Results:')
            print(f'Gamma/A = {self.adsorp:.10f}')
            print(f'-dgamma/dmu = {self.gamma_deriv:.10f}')
            print(f'Relative error = {self.error:.10f}')
            print(f'--------------------------------------------')
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating sum rule.')

    def information(self,):
        """
        Prints information about the object instance to screen.

        Args:
            None
        Returns:
            None
        """

        if self.Rs is not None:
            print(f'Geometry: Spherical Wall')
            print(f'Radius: {self.Rs} * diameter of a fluid particle')
        elif self.slit:
            print(f'Geometry: Slit')
            print(f'Width: {self.DFT.L} * diameter of a fluid particle')
        else:
            print(f'Geometry: Planar Wall')

        if self.slit:
            if self.wall_type == 'HW':
                print(f'Left wall type: Hard Wall')
            else:
                if self.wall_type == 'LJ':
                    print(f'Left wall type: Hard wall with 9-3 Lennard-Jones tail')
                elif self.wall_type == 'SLJ':
                    print(f'Left wall type: Hard wall with 9-3 Lennard-Jones tail shifted '\
                          f'such that minimum occurs at wall')
                else:
                    print(f'Left wall type: Hard Wall with WCA splitting of 9-3 '
                          f'Lennard-Jones tail')

                print(f'Diameter of left wall particles: {self.sigma_wall} * '\
                      f'fluid particle diameter')
                print(f'Left wall attraction strength: {self.epsilon_wall} * '\
                      f'fluid-fluid attraction strength')

            if self.right_wall_type == 'HW':
                print(f'Right wall type: Hard Wall')
            else:
                if self.wall_type == 'LJ':
                    print(f'Right wall type: Hard wall with 9-3 Lennard-Jones tail')
                elif self.wall_type == 'SLJ':
                    print(f'Right wall type: Hard wall with 9-3 Lennard-Jones tail shifted '\
                          f'such that minimum occurs at wall')
                else:
                    print(f'Right wall type: Hard Wall with WCA splitting of 9-3 '
                          f'Lennard-Jones tail')

                print(f'Diameter of right wall particles: {self.right_wall_sigma} * '\
                      f'fluid particle diameter')
                print(f'Right wall attraction strength: {self.right_epsilon_wall} * '\
                      f'fluid-fluid attraction strength')

        else:
            if self.wall_type == 'HW':
                print(f'Wall type: Hard Wall')
            else:
                if self.wall_type == 'LJ':
                    print(f'Wall type: Hard wall with 9-3 Lennard-Jones tail')
                elif self.wall_type == 'SLJ':
                    print(f'Wall type: Hard wall with 9-3 Lennard-Jones tail shifted '\
                          f'such that minimum occurs at wall')
                else:
                    print(f'Wall type: Hard Wall with WCA splitting of 9-3 '
                          f'Lennard-Jones tail')

                print(f'Diameter of wall particles: {self.sigma_wall} * '\
                      f'fluid particle diameter')
                print(f'Attraction strength: {self.epsilon_wall} * '\
                      f'fluid-fluid attraction strength')

        print(f'Picard mixing parameter: {self.alpha}')
        print(f'Ng move frequency: {self.ng} picard moves')
        print(f'Output local compressibility: {self.compressibility}')
        print(f'Output local thermal susceptibility: {self.susceptibility}')

    def plot(self, save = False):
        """
        Plots the equilibrium density profile, if found. If save is True,
        this plot is saved to the output folder.

        Args:
            Optional:
                save(bool): If True, the plot is written to pdf.

        Returns:
            None.
        """

        # We will plot the profile close to the wall
        # in the main plot and the full profile in
        # an inset.
        fig = plt.figure()
        ax_main = fig.add_subplot(111)
        if not self.slit:
            ax_inset = ax_main.inset_axes([0.6,0.6,0.38,0.38],
                                           transform = ax_main.transAxes)

        # Plot the density profile close to the wall in the main
        # figure and the full profile in the inset
        L = int(10./self.DFT.dr) + self.NiW


        if self.Rs is None:
            if self.slit is False:

                ax_main.plot(self.r[self.NiW:L],
                             self.DFT.density[self.NiW:L]/self.DFT.bulk_density,
                             color='cornflowerblue')
                ax_inset.plot(self.r[self.NiW:self.DFT.end],
                          self.DFT.density[self.NiW:self.DFT.end]/self.DFT.bulk_density,
                          color='cornflowerblue')

                ax_inset.set_xlabel(r'$z/\sigma$')
                ax_inset.set_ylabel(r'$\rho(z)/\rho_b$')
                ax_main.set_xlim(0.0,self.r[L])
            else:
                ax_main.plot(self.r[self.NiW:self.DFT.end],
                             self.DFT.density[self.NiW:self.DFT.end]/self.DFT.bulk_density,
                             color='cornflowerblue')
                ax_main.set_xlim(0.0,self.DFT.L)

            ax_main.set_xlabel(r'$z/\sigma$')
            ax_main.set_ylabel(r'$\rho(z)/\rho_b$')


        else:
            ax_main.plot((self.r[self.NiW:L]-self.Rs),
                         self.DFT.density[self.NiW:L]/self.DFT.bulk_density,
                         color='mediumvioletred')
            ax_inset.plot((self.r[self.NiW:self.DFT.end]-self.Rs),
                      self.DFT.density[self.NiW:self.DFT.end]/self.DFT.bulk_density,
                      color='mediumvioletred')

            ax_main.set_xlim(0.0,self.r[L]-self.Rs)
            ax_main.set_xlabel(r'$(r-R_s)/\sigma$')
            ax_main.set_ylabel(r'$\rho(r-R_s)/\rho_b$')

            ax_inset.set_xlabel(r'$(r-R_s)/\sigma$')
            ax_inset.set_ylabel(r'$\rho(r-R_s)/\rho_b$')

        # Axis formatting
        if self.DFT.fluid_type == 'HS':
            maxy = 0.5*np.ceil(self.DFT.density[self.NiW]/0.5)/self.DFT.bulk_density
        else:
            maxy = 0.5*np.ceil(np.amax(self.DFT.density)/0.5)/self.DFT.bulk_density

        ax_main.set_ylim(0.0,maxy)
        ax_main.tick_params(right=True, top=True, direction='in', pad = 5)
        if not self.slit:
            ax_inset.set_xlim(0.0,self.DFT.L)
            ax_inset.set_ylim(0.0,maxy)
            ax_inset.tick_params(right=True, top=True, direction='in')

        if save:
            plt.savefig(self.output_file_name.replace('.','') + '.pdf')

    def pressure(self,):
        """Returns the pressure of the system."""
        return self.DFT.pressure

    def chemical_potential(self, excess = False, ideal = False):
        """
        Returns the chemical potential of the system.
        If excess is True, only the excess component is returned.
        If ideal is True, only the ideal component is returned.
        If both are true, both are returned with excess as argument 1.

        Args:
            Optional:
                excess (bool): If True, only excess component of
                    chemical potential is returned.
                    Default is False.
                ideal (bool): If True, only ideal component of
                    chemical potential is returned.
                    Default is False.

        Returns:
            chemical potential (float)
        """

        if excess and ideal:
            return self.DFT.mu, self.DFT.T*np.log(self.DFT.bulk_density)
        elif excess:
            return self.DFT.mu
        elif ideal:
            return self.DFT.T*np.log(self.DFT.bulk_density)
        else:
            return (self.DFT.mu + self.DFT.T*np.log(self.DFT.bulk_density))

    def surface_tension(self, write_to_file = False):
        """
        Returns the surface tension of the system.

        Args:
            Optional:
                write_to_file(bool): If True, result is written to
                    the output file. Default is False.

        Returns:
            surface_tension (float)
        """
        if self.equilibrium:
            surface_tension = measure.surface_tension(self,fout = write_to_file)
            return surface_tension
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating surface measures.')
            return None

    def grand_potential(self, write_to_file = False):
        """
        Returns the grand potential of the system.

        Args:
            Optional:
                write_to_file(bool): If True, result is written to
                    the output file. Default is False.

        Returns:
            grand_potential (float)
        """
        if self.equilibrium:
            grand_potential = measure.grand_potential(self, fout = write_to_file)
            return grand_potential
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating surface measures.')
            return None



    def adsorption(self, ):
        """
        Returns the adsorption of the system.

        Args:
            None

        Returns:
            adsorption (float)
        """
        if self.equilibrium:
            return self.adsorp
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating surface measures.')
            return None

    def contact_density(self,):
        """
        Returns the contact density of the system.

        Args:
            None

        Returns:
            contact_density (float)
        """
        if self.equilibrium:
            return self.DFT.density[self.NiW]
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating surface measures.')
            return None

class planar(minimisation):

    """
    Minimisation object for a single planar external potential which has
    the form of an impenetrable wall to the left of the grid with an
    optional long-ranged attractive tail.
    """

    def __init__(self, cDFT, wall_type, alpha=0.1, ng=10,
                 file_path='cDFT_Results', epsilon_wall = 1.0,
                 sigma_wall=1.0, compressibility = False, susceptibility=False,
                 deriv = False, inplace=False):

        """
        Sets up cDFT minimisation in planar geometry.

        Args:
            Required:
                cDFT(DFT object): Contains parameters of fluid
                wall_type(string): Wall-fluid interaction
                                   Options: HW = Hard Wall
                                            LJ = Lennard-Jones 9-3
                                            SLJ = Lennard-Jones 9-3 with minimum
                                                shifted to occur at wall
                                            WCALJ = Lennard-Jones 9-3 with WCA
                                                splitting

            Optional:
                alpha(float): Mixing parameter for Picard update
                              Must be in interval (0,1)
                              Default is 0.1
                ng(int): Number of updates beween Ng update
                         Must be equal to or larger than 3
                         Default is 10
                file_path(string): Path to location to output results
                                   Default is cDFT_Results

                epsilon_wall(float): Strength of wall-fluid interaction
                                     The wall-fluid potential is multiplied by this
                                     hence 0 would correspond to a hard-wall
                                     Default is 1.0
                sigma_wall(float): Diameter of wall particles
                                   Default is the same as the diameter of fluid

                compressibility(bool): Defines whether to calculate local
                                       compressibility profile
                                       Default is False
                susceptibility(bool): Defines whether to calculate local thermal
                                      susceptibility profile
                                      Default is False

                deriv(bool): Defines whether this is a derivative calculation
                             Used when calculating chemical potential and
                             temperature derivatives, to prevent derivative
                             caculations outputting files
                             Default is false
                inplace(bool): Defined whether the minimisation procedure
                               should overwrite variables within the DFT
                               object
                               Default is false
       Returns:
            None
        """

        # Set parameters
        self.alpha = valid_alpha(alpha)
        self.ng = valid_ng(ng)
        self.file_path = file_path
        self.wall_type = wall_type
        self.Rs = None
        self.compressibility = compressibility
        self.susceptibility = susceptibility
        self.sigma_wall=sigma_wall
        self.epsilon_wall = epsilon_wall
        self.slit = False

        minimisation.__init__(self, cDFT, inplace)
        self.deriv = deriv

        # Sets up grid, placing the wall on the left and calculating the
        # distance of each grid point from the wall.
        self.r = np.fromiter(((i-self.NiW) for i in range(self.DFT.N)), float)
        self.r[:] *= self.DFT.dr

        # Set initial density
        self.DFT.density[self.NiW:self.DFT.padding] = self.DFT.bulk_density

        # Masks for updating weighted densities and correlation functions
        self.wmask = np.empty(self.DFT.N,bool)
        self.wmask[:] = False
        self.wmask[self.NiW] = True

        self.nmask = np.empty(self.DFT.N,bool)
        self.nmask[:] = False
        self.nmask[self.NiW-self.DFT.NiR+1:] = True

        self.cmask = np.empty(self.DFT.N,bool)
        self.cmask[1:] = True; self.cmask[0] = False

        # Initialises wall potential
        ext.setup_HW(self)

        # Sets up wall
        if self.wall_type == 'LJ':
            ext.setup_PLJ(self)

        elif self.wall_type == 'SLJ':
            ext.setup_PSLJ(self)

        elif self.wall_type == 'WCALJ':
            ext.setup_PWCALJ(self)

        elif self.wall_type == 'LV':

            # Only used by measure function to find liquid-vapour
            # surface tension
            self.DFT.density[:int(np.floor(self.DFT.N/2.))] = self.DFT.vapour_density
            self.DFT.density[:self.DFT.N-self.DFT.padding] = 0.0
            self.Vext[:] = 0.0

        # Sets up geometric aspects of fluid
        if self.DFT.fluid_type == 'TLJ':
            fluid.fourier_TLJ_const(self)
            self.calculate_attraction = fluid.fourier_PTLJ

            self.fatt = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            self.fcp = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            self.fft_att = fft.FFTW(self.DFT.potential,self.fatt, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
            self.ifft_cp = fft.FFTW(self.fcp,self.DFT.cp, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))

            self.fft_att()

        # Set up file name

        self.output_file_name = 'P' + self.wall_type + '_' + self.DFT.fluid_type + '_'

        if self.DFT.fluid_type == 'TLJ':
            self.output_file_name += str(self.DFT.cut_off) + '_'

        if self.wall_type != 'HW' and self.wall_type!='LV':
            self.output_file_name += str(self.epsilon_wall) + '_'

        self.output_file_name = self.output_file_name + str(self.DFT.bulk_density) + '_' + str(self.DFT.T)  \
                                   + '_' + str(self.DFT.functional)

        self.output_file_name = os.path.join(self.file_path, self.output_file_name)

    def weighted_densities(self):

        """
        Calculates the weighted densities in fourier space.
        Used internally within object.

        Args:
            None
        Returns:
            None
        """

        # Calculate fourier transform of r*density
        self.density[:] = self.DFT.density[:]
        self.density[self.wmask]*=0.5   # Numerical trick to impove accuracy
        self.fft_rho()

        # 2d weighted density
        self.fn2[:] = self.frho[:]*self.DFT.fw2[:]
        self.ifft_n2()
        self.DFT.n2[np.invert(self.nmask)] = 0.0

        # 3d weighted density
        self.fn3[:] = self.frho[:]*self.DFT.fw3[:]
        self.ifft_n3()
        self.DFT.n3[np.invert(self.nmask)] = 0.0

        # Vector 2d weighted density
        self.fn2v[:] = (self.frho[:]*self.DFT.fw2v[:])
        self.ifft_n2v()
        self.DFT.n2v[np.invert(self.nmask)] = 0.0

        # 1d,0d, vector 1d weighted densities can be defined using
        # results above
        self.DFT.n1[self.nmask] = self.DFT.n2[self.nmask]/(pi4*self.DFT.R)
        self.DFT.n0[self.nmask] = self.DFT.n2[self.nmask]/(pi4*self.DFT.R**2)
        self.DFT.n1v[self.nmask] = self.DFT.n2v[self.nmask]/(pi4*self.DFT.R)

    def correlation(self):

        """
        Calculates the correlation functions in fourier space.
        Used internally within object.

        Args:
            None
        Returns:
            None
        """

        # Calculate derivatives of functional
        self.DFT.calculate_derivatives(self.DFT)
        self.DFT.d2[0] = 0.0; self.DFT.d3[0] = 0.0; self.DFT.d2v[0] = 0.0;

        # Fourier transform derivatives
        self.fft_d2(); self.fft_d3(); self.fft_d2v();

        # Correlations proportional to scalar weight functions
        self.fc2[:] = self.fd2[:]*self.DFT.fw2[:]
        self.fc3[:] = self.fd3[:]*self.DFT.fw3[:]
        self.ifft_c2(); self.ifft_c3();

        # Correlations proportional to vector weight functions
        self.fc2v[:] = self.fd2v[:]*(-1.0*self.DFT.fw2v[:])
        self.ifft_c2v();

        self.DFT.c2[np.invert(self.cmask)] = 0.0
        self.DFT.c3[np.invert(self.cmask)] = 0.0
        self.DFT.c2v[np.invert(self.cmask)] = 0.0

        if self.DFT.fluid_type != 'HS':
            self.calculate_attraction(self)


    def minimise(self):

        """
        Performs numerical minimisation of planar object.
        Stores results within object, and outputs to file.

        Args:
            None
        Returns:
            None
        """

        # Perform minimisation of variational principle
        super().minimise()

        # If this is not a derivative calculation, output results
        if self.deriv is False:

            with open(self.output_file_name, 'w') as out:

                if self.slit:
                    out.write(f'Slit Geometry.\n')
                else:
                    out.write(f'Planar Geometry.\n')

            # Calculate additional profiles as required
            # The adsorption sum rule is always performed, to give an estimate of
            # the numerical accuracy of the results
            if self.compressibility:
                self.compressibility_profile, self.adsorp, self.gamma_deriv, self.error = measure.mu_derivative_measures(self)
                self.bulk_compressibility = measure.bulk_compressibility(self.DFT)
            else:
                self.adsorp, self.gamma_deriv, self.error = measure.mu_derivative_measures(self, chi=False)

            if self.susceptibility:
                self.susceptibility_profile, self.bulk_susceptibility = measure.temperature_derivative(self)

            self.output_simulation_data()

    def copy_parameters(self,):

        """Returns dict of parameters."""
        params = {'wall_type': self.wall_type, 'alpha': self.alpha, 'ng': self.ng,
                  'file_path': self.file_path, 'epsilon_wall':self.epsilon_wall,
                  'sigma_wall': self.sigma_wall}

        return params

    def contact_sum_rule(self,):
        """
        Calculates appropariate contact sum rule. Prints result to screen
        and to file.

        Args:
            None
        Returns:
            None

        """
        if self.called_contact_sum_rule:
            fout = True
        else:
            fout = False
            self.called_contact_sum_rule = True

        if self.equilibrium:
            if self.wall_type in ['LJ', 'SLJ', 'WCALJ']:
                measure.soft_substrate_sum_rule(self, pout = True, fout = fout)
            else:
                measure.planar_hw_contact_sum_rule(self, pout = True, fout = fout)
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating sum rule.')



class spherical(minimisation):

    """
    Minimisation object for a single spherically symmetric external potential
    which has the form of an impenetrable solute to the left of the grid with
    an optional long-ranged attractive tail.
    """

    def __init__(self, cDFT, Rs, wall_type, alpha = 0.1, ng=10,
                 file_path = 'cDFT_Results', deriv = False,
                 epsilon_wall = 1.0, sigma_wall=1.0,
                 compressibility = False, susceptibility=False,
                 inplace=False):

        """
        Sets up cDFT minimisation in spherical geometry.

        Args:
            Required:
                cDFT(DFT object): Contains parameters of fluid
                Rs (float): Radius of spherical particle on left
                            in units of fluid particle diameters
                wall_type (string): Wall-fluid interaction
                                    Options:
                                        'HW' = Hard Wall
                                        'LJ' = Lennard-Jones 9-3
                                        'SLJ' = Lennard-Jones 9-3 with minimum
                                                shifted to occur at wall

            Optional:
                alpha(float): Mixing parameter for Picard update
                              Must be in interval (0,1)
                              Default is 0.1
                ng(int): Number of updates beween Ng update
                         Must be equal to or larger than 3
                         Default is 10

                file_path(string): Path to location to output results
                                    Default is 'cDFT_Results'

                epsilon_wall(float): Strength of wall-fluid interaction
                                     The wall-fluid potential is multiplied by this
                                     hence 0 would correspond to a hard-wall
                                     Default is 1.0
                sigma_wall(float): Diameter of wall particles
                                   Default is the same as the diameter of fluid

                compressibility(bool): Defines whether to calculate local
                                       compressibility profile
                                       Default is False
               susceptibility(bool): Defines whether to calculate local thermal
                                      susceptibility profile
                                      Default is False

                deriv(bool): Defines whether this is a derivative calculation
                             Used when calculating chemical potential and
                             temperature derivatives, to prevent derivative
                             caculations outputting files
                             Default is False
                inplace(bool): Defined whether the minimisation procedure
                               should overwrite variables within the DFT
                               object
                               Default is False

        Returns:
            None
        """

        # Set parameters
        self.alpha = valid_alpha(alpha)
        self.ng = valid_ng(ng)
        self.file_path = file_path
        self.deriv = deriv
        self.wall_type = wall_type
        self.epsilon_wall = epsilon_wall
        self.sigma_wall = sigma_wall
        self.NiW = cDFT.N - cDFT.end
        self.slit = False
        self.compressibility = compressibility
        self.susceptibility = susceptibility

        # Initialise non-geometry dependent parts
        minimisation.__init__(self, cDFT, inplace)

         # Sets up grid, placing the spherical wall on the left and calculating the
        # distance of each grid point from the centre of the wall.
        self.Rs = Rs*self.DFT.sigma
        self.r = np.fromiter((((self.Rs-self.NiW*self.DFT.dr) + i*self.DFT.dr)  for i in range(self.DFT.N)), float)

        # Initialise density profile. This is set to the bulk density
        # for all grid points not within a wall
        self.zero = 0
        self.DFT.density[:] = 0.0
        self.DFT.density[self.NiW:self.DFT.padding] = self.DFT.bulk_density

        # Initialise external potential
        ext.setup_HW(self)

        if wall_type == 'LJ':
            ext.setup_SLJ(self)

        elif wall_type == 'SLJ':
            ext.setup_SSLJ(self)

        # Masks for updating weighted densities and correlation functions
        self.nmask = np.empty(self.DFT.N,dtype=bool);
        self.nmask[:] = False;
        self.nmask[1:self.DFT.padding] = True
        self.cmask = np.empty(self.DFT.N,dtype=bool);
        self.cmask[1:] = True; self.cmask[0] = False

        if self.DFT.fluid_type == 'HS':
            self.nmask[0:self.NiW-2*self.DFT.NiR+1] = False
            self.cmask[0:self.NiW-2*self.DFT.NiR+1] = False

        elif self.DFT.fluid_type == 'TLJ':

            zero = self.r == 0.0
            if np.any(zero):
                indx = np.where(zero);
                self.DFT.zero_indx = indx[0][0]
                self.cmask[:self.DFT.zero_indx+1] = False
                self.nmask[:self.DFT.zero_indx+1] = False

            if 2*self.Rs > self.DFT.cut_off:
                fluid.fourier_TLJ_const(self)

                self.calculate_attraction = fluid.fourier_STLJ
                self.fatt = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
                self.fcp = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
                self.fft_att = fft.FFTW(self.DFT.potential,self.fatt, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
                self.ifft_cp = fft.FFTW(self.fcp,self.DFT.cp, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
                self.fft_att()

            else:

                fluid.rs_TLJ_const(self)
                self.calculate_attraction = fluid.rs_STLJ

        # Set up file name

        self.output_file_name = 'S' + self.wall_type + '_' + self.DFT.fluid_type + '_' + str(self.Rs) + '_'

        if self.DFT.fluid_type == 'TLJ':
            self.output_file_name += str(self.DFT.cut_off) + '_'

        if self.wall_type != 'HW':
            self.output_file_name += str(self.epsilon_wall) + '_'

        self.output_file_name = self.output_file_name  + str(self.DFT.bulk_density) +\
                            '_' + str(self.DFT.T) + '_' + str(self.DFT.functional)
        self.output_file_name = os.path.join(self.file_path, self.output_file_name)

    def weighted_densities(self):

        """
		Calculates the weighted densities in fourier space.
		Used internally within object.

		Args:
			None

		Returns:
			None
        """

        # Fourier transform density profile
        self.density[:] = self.DFT.density[:]*self.r[:]
        self.density[self.NiW]*=0.5 # Numerical trick
        self.fft_rho()

        # 2d wieghted density
        self.fn2[:] = self.frho[:]*self.DFT.fw2[:]
        self.ifft_n2()
        self.DFT.n2[self.nmask]/=self.r[self.nmask]
        self.DFT.n2[:self.NiW-self.DFT.NiR+1] = 0.0

        # 3d weighted density
        self.fn3[:] = self.frho[:]*self.DFT.fw3[:]
        self.ifft_n3()
        self.DFT.n3[self.nmask]/=self.r[self.nmask]
        self.DFT.n3[:self.NiW-self.DFT.NiR+1] = 0.0

        # Vector 2d weighted density
        self.fn2v[:] = (self.frho[:]*self.DFT.fw2v[:])
        self.ifft_n2v()
        self.DFT.n2v[self.nmask]/=self.r[self.nmask]
        self.DFT.n2v[self.nmask] += self.DFT.n3[self.nmask]/self.r[self.nmask]
        self.DFT.n2v[:self.NiW-self.DFT.NiR+1] = 0.0

        # Numerical tricks
        if self.wall_type == 'SLJ':
            self.DFT.n3[self.NiW-self.DFT.NiR+1] = 0.0

        if self.wall_type == 'LJ':
            points = self.DFT.n3<1e-12; self.DFT.n3[points] = 1e-12

        # C1d, 0d, vector 1d weighted densities
        self.DFT.n1[:] = self.DFT.n2[:]/(pi4*self.DFT.R)
        self.DFT.n0[:] = self.DFT.n2[:]/(pi4*(self.DFT.R*self.DFT.R))
        self.DFT.n1v[:] = self.DFT.n2v[:]/(pi4*self.DFT.R)

    def correlation(self):

        """
		Calculates the correlation functions in fourier space.
		Used internally within object.

		Args:
			None

		Returns:
			None
        """

        # Calculate derivatives of functional
        self.DFT.calculate_derivatives(self.DFT)
        self.DFT.d2[np.invert(self.cmask)] = 0.0
        self.DFT.d2v[np.invert(self.cmask)] = 0.0
        self.DFT.d3[np.invert(self.cmask)] = 0.0

        # Multiply by r
        self.DFT.d2[:]*=self.r[:]; self.DFT.d3[:]*=self.r[:];

        # Fourier transform derivatives
        self.fft_d2(); self.fft_d3(); self.fft_d2v();

        # Correlations proportional to scalar weight functions
        self.fc2[:] = self.fd2[:]*self.DFT.fw2[:]
        self.fc3[:] = self.fd3[:]*self.DFT.fw3[:]
        self.ifft_c2(); self.ifft_c3();
        self.DFT.c2[self.cmask]/=self.r[self.cmask]
        self.DFT.c3[self.cmask]/=self.r[self.cmask]

        # Correlations proportional to vector weight functions
        self.fc2v[:] = self.fd2v[:]*self.DFT.fw3[:]
        self.DFT.d2v[:]*=self.r[:]
        self.fft_d2v()
        self.fc2v_dummy[:] = self.fd2v[:]*self.DFT.fw2v[:]
        self.ifft_c2v(); self.ifft_c2v_dummy();
        self.DFT.c2v[self.cmask] -= self.DFT.c2v_dummy[self.cmask]
        self.DFT.c2v[self.cmask]/=self.r[self.cmask]

        # Zero as appropriate
        self.DFT.c2[np.invert(self.cmask)] = 0.0
        self.DFT.c2v[np.invert(self.cmask)] = 0.0
        self.DFT.c3[np.invert(self.cmask)] = 0.0

        # Correlations due to fluid-fluid attraction
        if self.DFT.fluid_type != 'HS':
            self.calculate_attraction(self)


    def minimise(self):

        """
		Performs numerical minimisation of spherical object.
		Stores results within object, and outputs to file.

		Args:
			None

		Returns:
			None
        """

        # Perform minimisation of variational principle
        super().minimise()

        if not self.deriv and not self.fail:
            with open(self.output_file_name, 'w') as out:
                out.write(f'Spherical Geometry.\nRs = {self.Rs:.6f}\n')

            # Calculate additional profiles as required
            # The adsorption sum rule is always performed, to give an estimate of
            # the numerical accuracy of the results
            if self.compressibility:
                self.compressibility_profile, self.adsorp, self.gamma_deriv, self.error = measure.mu_derivative_measures(self)
                self.bulk_compressibility = measure.bulk_compressibility(self.DFT)
            else:
                self.adsorp, self.gamma_deriv, self.error = measure.mu_derivative_measures(self, chi=False)

            if self.susceptibility:
                self.susceptibility_profile, self.bulk_susceptibility = measure.temperature_derivative(self)

            self.output_simulation_data()

    def copy_parameters(self):

        params = {"wall_type": self.wall_type, "alpha": self.alpha, "ng": self.ng,
				  "file_path": self.file_path, "epsilon_wall": self.epsilon_wall,
				  "sigma_wall": self.sigma_wall}

        return params

    def contact_sum_rule(self,):
        """
        Calculates appropriate contact sum rule. Prints result to screen.
        If it is the first time the function has been called, the result
        is also printed to file.
        """

        if self.called_contact_sum_rule:
            fout = True
        else:
            fout = False
            self.called_contact_sum_rule = True

        if self.equilibrium:
            if self.wall_type == 'HW':
                measure.spherical_contact_sum_rules(self, pout = True,
                                                    fout = fout)
            else:
                measure.soft_substrate_sum_rule(self, pout = True,
                                                fout = fout)
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating sum rule.')



class slit(planar):

    """
    Minimisation object for an infinite slit geometry. Hard walls
    at either end of grid with optional attractive tail.
    """

    def __init__(self, cDFT, left_wall_type, right_wall_type,
                 alpha=0.1, ng=10, file_path='cDFT_Results',
                 left_epsilon_wall = 1.0, left_sigma_wall=1.0,
                 right_epsilon_wall=1.0, right_sigma_wall = 1.0,
                 compressibility = False, susceptibility=False,
                 deriv = False, inplace=False):
        """
        Sets up cDFT minimisation in a slit geometry.

        Args:
            Required:
                cDFT(DFT object): Contains parameters of fluid
                left_wall_type(string): Left wall-fluid interaction type
                                   Options: HW = Hard Wall
                                            SLJ = Lennard-Jones 9-3 with minimum
                                                shifted to occur at wall
               right_wall_type(string): Right wall-fluid interaction type
                                   Options: HW = Hard Wall
                                            SLJ = Lennard-Jones 9-3 with minimum
                                                shifted to occur at wall

            Optional:
                alpha(float): Mixing parameter for Picard update
                              Must be in interval (0,1)
                              Default is 0.1
                ng(int): Number of updates beween Ng update
                         Must be equal to or larger than 3
                         Default is 10
                file_path(string): Path to location to output results
                                   Default is cDFT_Results

                left_epsilon_wall(float): Strength of left wall-fluid interaction
                                     The wall-fluid potential is multiplied by this
                                     hence 0 would correspond to a hard-wall
                                     Default is 1.0
                left_sigma_wall(float): Diameter of left wall particles
                                   Default is the same as the diameter of fluid

                right_epsilon_wal(float): Strength of right wall-fluid interaction
                                     The wall-fluid potential is multiplied by this
                                     hence 0 would correspond to a hard-wall
                                     Default is 1.0
                right_sigma_wall(float): Diameter of right wall particles
                                   Default is the same as the diameter of fluid

                compressibility(bool): Defines whether to calculate local
                                       compressibility profile
                                       Default is False
                susceptibility(bool): Defines whether to calculate local thermal
                                      susceptibility profile
                                      Default is False

                deriv(bool): Defines whether this is a derivative calculation
                             Used when calculating chemical potential and
                             temperature derivatives, to prevent derivative
                             caculations outputting files
                             Default is false
                inplace(bool): Defined whether the minimisation procedure
                               should overwrite variables within the DFT
                               object
                               Default is false
       Returns:
            None
        """

        # Set parameters
        self.alpha = valid_alpha(alpha)
        self.ng = valid_ng(ng)
        self.file_path = file_path

        self.slit =True
        self.Rs = None
        self.compressibility = compressibility
        self.susceptibility = susceptibility

        self.wall_type = left_wall_type
        self.sigma_wall = left_sigma_wall
        self.epsilon_wall = left_epsilon_wall
        self.right_wall_type = right_wall_type
        self.right_epsilon_wall = right_epsilon_wall
        self.right_sigma_wall = right_sigma_wall

        minimisation.__init__(self, cDFT, inplace)
        self.deriv = deriv

        # Sets up grid, placing the wall on the left and calculating the
        # distance of each grid point from the wall.
        self.r = np.fromiter(((i-self.NiW) for i in range(self.DFT.N)), float)
        self.r[:] *= self.DFT.dr

        # Set initial density
        self.DFT.density[self.NiW:self.DFT.end] = self.DFT.bulk_density

        # Masks for updating weighted densities and correlation functions
        self.wmask = np.empty(self.DFT.N,bool)
        self.wmask[:] = False
        self.wmask[self.NiW] = True
        self.wmask[self.DFT.end-1] = True

        self.nmask = np.empty(self.DFT.N,bool)
        self.nmask[:] = False
        self.nmask[self.NiW-self.DFT.NiR+1:self.DFT.end+self.DFT.NiR-1] = True

        self.cmask = np.empty(self.DFT.N,bool)
        self.cmask[1:-1] = True
        self.cmask[0] = False
        self.cmask[-1] = False

        # Initialises wall potential
        ext.setup_HW(self)

        # Sets up wall
        if self.wall_type == 'LJ':
            ext.setup_PLJ(self)

        elif self.wall_type == 'SLJ':
            ext.setup_PSLJ(self)

        elif self.wall_type == 'WCALJ':
            ext.setup_PWCALJ(self)

        if self.right_wall_type == 'LJ':
            ext.setup_PLJ(self,True)

        elif self.right_wall_type == 'SLJ':
            ext.setup_PSLJ(self, True)

        elif self.right_wall_type == 'WCALJ':
            ext.setup_PWCALJ(self, True)

        # Sets up geometric aspects of fluid
        if self.DFT.fluid_type == 'TLJ':
            fluid.fourier_TLJ_const(self)
            self.calculate_attraction = fluid.fourier_PTLJ

            self.fatt = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            self.fcp = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            self.fft_att = fft.FFTW(self.DFT.potential,self.fatt, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
            self.ifft_cp = fft.FFTW(self.fcp,self.DFT.cp, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))

            self.fft_att()

        # Set up file name
        self.output_file_name = 'SL' + self.wall_type
        self.output_file_name +=  self.right_wall_type + '_' + self.DFT.fluid_type + '_'

        if self.DFT.fluid_type == 'TLJ':
            self.output_file_name += str(self.DFT.cut_off) + '_'

        self.output_file_name = self.output_file_name + str(self.DFT.bulk_density) + '_' + str(self.DFT.T)  \
                                   + '_' + str(self.DFT.functional)
        self.output_file_name = os.path.join(self.file_path, self.output_file_name)

    def copy_parameters(self,):

        """Returns dict of parameters."""
        params = {'left_wall_type': self.wall_type, 'alpha': self.alpha, 'ng': self.ng,
                  'file_path': self.file_path, 'left_epsilon_wall':self.epsilon_wall,
                  'left_sigma_wall': self.sigma_wall, 'right_sigma_wall': self.right_sigma_wall,
                  'right_epsilon_wall': self.right_epsilon_wall, 'right_wall_type': self.right_wall_type}

        return params

    def contact_sum_rule(self,):
        """
        Calculates appropariate contact sum rule. Prints result to screen
        and to file.

        Args:
            None
        Returns:
            None

        """
        if self.called_contact_sum_rule:
            fout = True
        else:
            fout = False
            self.called_contact_sum_rule = True

        if self.equilibrium:
            measure.slit_sum_rule(self, pout = True, fout = fout)
        else:
            print(f'Equilibrium profile has not been found.')
            print(f'Please perform minimisation before calculating sum rule.')

