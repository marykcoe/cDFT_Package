#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program.
Supports hard-sphere and truncated Lennard-Jones fluids in
contact with homogenous planar surfaces, spherical solutes
and confined to a slit with homogeneous surfaces.

Created April 2019. Last Update November 2021.
Author: Mary K. Coe
E-mail: m.k.coe@bristol.ac.uk

This program utilises FMT and supports the Rosenfeld,
White-Bear and White-Bear Mark II functionals.

For information on how to use this package please consult
the accompanying tutorials. Information on how the package
works can be found in Chapter 4 and Appendix A-C of the
following thesis (link available December 2021)

This module contains functions to performmeasures of the system.
Supported measures are:
    Adsorption
    Grand Potential
    Surface Tension
    Local Compressibility
    Local Thermal Susceptibility
    Liquid-Vapour Surface Tension
    Contact Sum Rules (form depends on geometry and potentials of system)
    Adsorption Sum Rule
    
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

import datetime
from copy import deepcopy
import numpy as np
from scipy.optimize import bisect

import cDFT.minimisation as minimisation
import cDFT.fluid_potentials as potentials

def grand_potential_arr(minimise):

    """
	Calculates the grand potential at every grid point in the system.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

	Returns:
			Array containing grand potential for each grid point
    """

    # Calculate chemical potential (ideal + excess)
    mu = minimise.DFT.mu + minimise.DFT.T*np.log(minimise.DFT.bulk_density)

    # FMT hard-sphere part
    minimise.weighted_densities()
    phi = minimise.DFT.T*minimise.DFT.free_energy(minimise.DFT)

    # Ideal part
    phi[minimise.rmask] += minimise.DFT.T*minimise.DFT.density[minimise.rmask]*\
                            (np.log(minimise.DFT.density[minimise.rmask])-1.0)

    # Extrinsic part
    phi[minimise.rmask] += minimise.DFT.density[minimise.rmask]\
                                    *(minimise.Vext[minimise.rmask] - mu)

    # Attractive fluid part
    if minimise.DFT.fluid_type != 'HS':
            phi[minimise.rmask] += 0.5*minimise.DFT.density[minimise.rmask]\
                                                *minimise.DFT.cp[minimise.rmask]

    # Spherical integration
    if minimise.Rs is not None:
        phi[:] *=4.*np.pi*minimise.r[:]*minimise.r[:]

    phi[:]*=minimise.DFT.dr

    return phi[:]

def grand_potential(minimise, pout = False, fout = False):

    """
	Calculates the grand potential of a system

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False

	Returns:
			Grand potential
    """

    # Calculate grand potential at each grid point
    phi = grand_potential_arr(minimise)

    # Sum
    phi = np.sum(phi[1:minimise.DFT.end])

    # Print out if applicable
    if pout:
        print(f'Grand Potential = {phi:.5f}')

    # Write out to file if applicable
    if fout:
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'Grand Potential = {phi:.12f}\n')

    return phi

def surface_tension(minimise, pout = False, fout = False):

    """
	Calculates the surface tension of the system.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			Surface tension of system
    """

    # Calculate grand potential at each grid point
    gamma = grand_potential_arr(minimise)

    # Calculate excess grand potential (minus ideal part)
    if minimise.Rs is not None:
        gamma[minimise.rmask] += minimise.DFT.pressure*minimise.DFT.dr*(4.0*\
                 np.pi*minimise.r[minimise.rmask]*minimise.r[minimise.rmask])
    else:
        gamma[minimise.NiW:minimise.DFT.end] += (minimise.DFT.pressure*minimise.DFT.dr)

    # Sum the appropriate parts of the array
    if minimise.wall_type != 'LV':
        if not minimise.slit:
            gamma = np.sum(gamma[1:minimise.DFT.end])
        else:
            gamma = np.sum(gamma[1:minimise.DFT.end+minimise.DFT.NiR])

    else:
        gamma = np.sum(gamma[minimise.rmask])

    # Divide by the surface area of the iterface
    if minimise.Rs is not None:
        gamma /= (4.0*np.pi*minimise.Rs*minimise.Rs)

     # Print out if applicable
    if pout:
        print(f'Surface Tension = {gamma:.5f}')

    # If applicable, write the result to the output file
    if fout:
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'Surface Tension = {gamma:.12f}\n')

    return gamma

def adsorption(minimise, pout = False, fout = False):

    """
	Calculates the adsorption of the system.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			Adsorption of system
    """

    # Calculate the difference between the density profile and its bulk value
    drho = minimise.DFT.density[minimise.rmask] - minimise.DFT.bulk_density

    # Integrate in the appropriate coordinate system
    if minimise.Rs is not None:
        drho[:] *= minimise.r[minimise.rmask]*minimise.r[minimise.rmask]
        adsorption = minimise.DFT.dr*np.sum(drho)/(minimise.Rs*minimise.Rs)
    else:
        adsorption = np.sum(drho)*minimise.DFT.dr

    # Print out if applicable
    if pout:
        print(f'Adsorption = {adsorption:.5f}')

     # If applicable, write the result to the output file
    if fout:
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'Adsorption = {adsorption:.12f}\n')

    return adsorption

def planar_hw_contact_sum_rule(planar, pout = False, fout = True):

    """
    Calculates the planar hard wall contact sum rule.

    Args:
        Required:
            planar(minimise.planar): Planar minimisation object

        Optional:
            pout(bool): Indicates whether to print result to screen
            fout(bool): Indicates whether to print result to file

    Returns:
        None
    """
    LHS = planar.DFT.pressure
    RHS = planar.DFT.T*planar.DFT.density[planar.NiW]

    # Print out if applicable
    if pout:
        print(f'\n--------------------------------------------')
        print(f'Contact Sum Rule Results:')
        print(f'pressure = {LHS:.10f}\nkbT*rho(0) = {RHS:.10f}')
        print(f'Relative Error = {abs(LHS-RHS)/LHS:.10f}')
        print(f'--------------------------------------------\n')

    # If applicable, write the result to the output file
    if fout:
        with open(planar.output_file_name, 'a') as out:
            out.write(f'p = {LHS:.12f}\tkbTrho(0) = {RHS:.12f}\n')
            out.write(f'Relative Error = {abs(LHS-RHS)/LHS:.12f}\n\n')

    return LHS, RHS

def spherical_derivative(spherical, mode, dR=4.0):

    """
	Performs a numerical derivative with respect to the radius (Rs) of the
	spherical substrate

	Args:
		Required:
			spherical(minimise.spherical): Spherical minimisation object
			mode (string):            Derivative to take
									  Options are:
										 'grand potential'
										 'surface tension'
										 'all' (performs both options above)

		Optional:
			dR (float): Number of grid points to extend radius of substrate by
					  for differential
	Returns:
			if mode == 'grand potential':
				returns derivative of grand potential with respect to Rs
			if mode == 'surface tension':
				returns derivative of surface tension with respect to Rs and
					surface tension of input spherical object
			if mode == 'all':
				returns derivatives of grand potential and surface tension
					with repect to Rs and surface tension of spherical input
					object
    """


    # Calculate grand potential of spherical object
    if mode == 'grand potential':
        old = grand_potential(spherical)

    # Calculate gsurface tension of spherical object
    elif mode == 'surface tension':
        old = surface_tension(spherical)

    # Calculate grand potential and surface tension of spherical object
    elif mode == 'all':
        old = grand_potential_arr(spherical)
        old_omega = np.sum(old[:spherical.DFT.end])
        old[spherical.rmask] += spherical.DFT.pressure*spherical.DFT.dr*(4.*\
           np.pi*spherical.r[spherical.rmask]*spherical.r[spherical.rmask])
        old[np.invert(spherical.nmask)] = 0.0
        old_gamma = np.sum(old[:spherical.DFT.end])/(4.0*np.pi*spherical.Rs*spherical.Rs)

    # Increase radius of substrate
    Rs_new = spherical.Rs + (dR*spherical.DFT.dr)
    params = spherical.copy_parameters()
    spherical_new = minimisation.spherical(spherical.DFT, Rs_new, deriv=True, **params)

    # Find new equilibrium density profile
    spherical_new.minimise()

    # Caculate difference in radii
    dRs = (spherical_new.Rs - spherical.Rs)

    # Calculate new measures as appropriate
    if mode == 'grand potential':
        new = grand_potential_arr(spherical_new)
        new = np.sum(new[:spherical_new.DFT.end-int(dR)])

    elif mode == 'surface tension':
        new = grand_potential_arr(spherical_new)
        new[spherical_new.rmask] += spherical_new.DFT.pressure*spherical_new.DFT.dr*(4.*\
               np.pi*spherical_new.r[spherical_new.rmask]*spherical_new.r[spherical_new.rmask])
        new = np.sum(new[:spherical_new.DFT.end-int(dR)])/(4.0*np.pi*spherical_new.Rs*spherical_new.Rs)

    elif mode == 'all':
        new = grand_potential_arr(spherical_new)
        new_omega = np.sum(new[:spherical_new.DFT.end-int(dR)])

        new[spherical_new.rmask] += spherical_new.DFT.pressure*spherical_new.DFT.dr*(4.*\
               np.pi*spherical_new.r[spherical_new.rmask]*spherical_new.r[spherical_new.rmask])
        new_gamma = np.sum(new[:spherical_new.DFT.end-int(dR)])/(4.0*np.pi*spherical_new.Rs*spherical_new.Rs)

    # Delete new spherical object
    del spherical_new

    # Find the derivatives
    if mode == 'all':
        deriv_gamma = (new_gamma-old_gamma)/dRs
        deriv_omega = (new_omega-old_omega)/dRs
    else:
         deriv =(new-old)/dRs

    # Return the appropriate derivatives
    if mode == 'grand potential':
        return deriv
    elif mode == 'surface tension':
        return deriv, old
    elif mode == 'all':
        return deriv_omega, deriv_gamma, old_gamma

def spherical_contact_sum_rule_a(minimise, pout = False, fout = False):

    """
	Calculates a sum rule and associated error.
	This sum rule relates the derivative of the grand potential with respect to
	the radius of the spherical substrate to the contact density at the substrate

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			LHS and RHS of sum rule.
    """

    # Calculate spherical derivative of grand potential (LHS of sum rule)
    LHS = minimise.DFT.beta*spherical_derivative(minimise, 'grand potential')

    # Calculates RHS of sum rule
    RHS = 4.0*np.pi*(minimise.Rs)*(minimise.Rs)*minimise.DFT.density[minimise.NiW]

    # Print out if applicable
    if pout:
        print(f'\n--------------------------------------------')
        print(f'Contact Sum Rule A Results:')
        print(f'beta*d(omega)/(dRb) = {LHS:.10f}')
        print(f'4piRs^2rho[Rs] = {RHS:.10f}')
        print(f'Relative Error = {abs(LHS-RHS)/RHS:.10f}')
        print(f'--------------------------------------------\n')

    # If applicable, write the result to the output file
    if fout:
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'beta*domega/dRs = {LHS:.12f}\t4piRs^2rho(Rs) = {RHS:.12f}\n')
            out.write(f'Relative Error = {abs(LHS-RHS)/RHS:.12f}\n')

    return LHS, RHS

def spherical_contact_sum_rule_b(minimise, pout = False, fout = False):

    """
	Calculates a sum rule and associated error.
	This sum rule relates the surface tension to the contact density at the
	substrate.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			LHS and RHS of sum rule.
    """

    # RHS of sum rule
    deriv, gamma_old = spherical_derivative(minimise,'surface tension')
    RHS = minimise.DFT.pressure + 2.0*gamma_old/(minimise.Rs) + deriv

    # Calculate LHS of sum rule
    LHS = minimise.DFT.T*minimise.DFT.density[minimise.NiW]

    # Print out if applicable
    if pout:
        print(f'\n--------------------------------------------')
        print(f'Contact Sum Rule B Results:')
        print(f'rho[Rs] = {LHS:.10f}\t sum_rule = {RHS:.10f}')
        print(f'Relative Error = {abs(LHS-RHS)/LHS:.10f}')
        print(f'--------------------------------------------\n')

    # If applicable, write the result to the output file
    if fout:
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'\np + 2gamma/Rs + dgamma/dRs = {RHS:.12f}\tkbTrho(Rs) = {LHS:.12f}\n')
            out.write(f'Relative Error = {abs(LHS-RHS)/LHS:.12f}\n\n')

    return LHS, RHS

def spherical_contact_sum_rules(spherical, dR=4.0, pout=False, fout = False,
								return_all = False):

    """
	Calculates both spherical contact sum rule and associated error.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			dR (float):        Number of grid points to increase substrate radius by
							   Default is 4.0
			pout (bool):       Print results of sum rule to terminal
						       Default is False
			fout (bool):       Write results of sum rule to file
						       Default is False
			return_all (bool): Determines whether to return each side of sum rules
							   or whether to return only relative errors
							   Default is False
	Returns:
			if return_all:
				returns LHS and RHS of each sum rule in order sum rule a, then
					sum rule b
			else:
				returns relative error of sum rule a and sum rule b (in that order)
    """

    # Find contact density
    rho_contact = spherical.DFT.density[spherical.NiW]

    # Calculate derivatives
    LHS_omega, deriv_gamma, old_gamma = spherical_derivative(spherical, 'all', dR=dR)

    # Calculate contact sum rule a
    RHS_omega = 4.0*np.pi*(spherical.Rs*spherical.Rs)*rho_contact
    LHS_omega *= spherical.DFT.beta

    # Calculate contact sum rule b
    LHS_gamma = spherical.DFT.T*rho_contact
    RHS_gamma =(spherical.DFT.pressure + 2.0*old_gamma/(spherical.Rs) \
                                    + deriv_gamma)

    # Calculate relative errors
    omega_error = abs(LHS_omega-RHS_omega)/RHS_omega
    gamma_error = abs(LHS_gamma-RHS_gamma)/LHS_gamma

    # Print out if applicable
    if pout:
        print(f'\n----------------------------------------------------')
        print(f'Contact Sum Rule A:')
        print(f'beta*d(omega)/(dRb) = {LHS_omega:.10f}')
        print(f'4piRs^2rho(Rs) = {RHS_omega:.10f}')
        print(f'Relative Error = {omega_error:.10f}\n')
        print(f'Contact Sum Rule B:')
        print(f'kbTrho(Rs) = {LHS_gamma:.10f}')
        print(f'p + 2gamma/Rs + dgamma/dRs = {RHS_gamma:.10f}')
        print(f'Relative Error = {gamma_error:.10f}')
        print(f'-----------------------------------------------------\n')

    # If applicable, write the result to the output file
    if fout:
        with open(spherical.output_file_name, 'a') as out:
            out.write(f'\np + 2gamma/Rs + dgamma/dRs = {RHS_gamma:.12f}\tkbTrho(Rs) = {LHS_gamma:.12f}\n')
            out.write(f'Relative Error = {gamma_error:.12f}\n\n')
            out.write(f'beta*domega/dRs = {LHS_omega:.12f}\t4piRs^2rho(Rs) = {RHS_omega:.12f}\n')
            out.write(f'Relative Error = {omega_error:.12f}\n')

    if return_all:
        return LHS_omega, RHS_omega, LHS_gamma, RHS_gamma
    else:
        return omega_error, gamma_error

def calculate_mu(rhob, DFT, dT):

    """
	Calculates the chemical potential difference between two DFT objects.
	This functions is used within a numerical root finder to find the densities
	at two different temperatures which have the same chemical potential.

	Args:
		Required:
			rhob (float):            Density to try
			minimise (minimise_obj): Planar or spherical minimisation object
			dT (float):              Difference between new temperature and
									 current temperature.
	Returns:
			difference between chemical potentials of DFT objects
    """

    # Calculate chemical potential
    temp = deepcopy(DFT)
    temp.update_state_point(rhob,temp.T+dT)

    # Add ideal component to excess chemical potential
    mu_new = temp.mu + temp.T*np.log(rhob)
    mu_old = DFT.mu + DFT.T*np.log(DFT.bulk_density)

    return mu_new-mu_old

def rho_bulk_const_mu(minimise,dT=1e-7):

    """
	Calculates the density at a temperature T+dT which has the same chemical
	potential as a fluid at temperature T.
	This is required when calculated the local thermal susceptibility, which
	involves a temperature derivative at constant chemical potential.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			dT (float):              Difference between new temperature and
									 current temperature.

	Returns:
			Density for temperature T+dT
	"""

    # Find new bulk density
    rho_bulk = bisect(calculate_mu,0.99*minimise.DFT.bulk_density,
					  1.01*minimise.DFT.bulk_density, args=(minimise.DFT,dT),
					   xtol=1e-15, maxiter=200)

    return rho_bulk

def temperature_derivative(minimise,  dT = 1e-7, return_profile = False):

    """
	Calculates the temperature derivative at constant chemical potential
	of the density profile.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			dT (float):              Difference between new temperature and
									 current temperature.
			return_profile (bool):   Return density profile at new chemical
									 potential
									 Default is false

	Returns:
			local thermal susceptibility profile
			local thermal susceptibility in bulk

	Example:
			>>> cDFT.measures.temperature_derivative(planar_obj)

    """

    # Find bulk density at constant chemical potential at new temperature
    rho_bulk = rho_bulk_const_mu(minimise,dT)

    # Define new DFT and minimisation objects
    temp = deepcopy(minimise.DFT)
    temp.update_state_point(rho_bulk, T = temp.T + dT)

    # Define new minimisation object. Only the DFT within the object is changed.
    # The object is identified as a derivative, so no output files will be written.
    params = minimise.copy_parameters()
    if minimise.Rs is not None:
        minimise_new = minimisation.spherical(temp,minimise.Rs,deriv=True,**params)
    else:
        minimise_new = minimisation.planar(temp,**params,deriv=True)

    # Find equilibrium density profile
    minimise_new.minimise()

    # Calculate local thermal susceptibility and bulk value
    result = (minimise_new.DFT.density[:]-minimise.DFT.density[:])/(dT)
    bulk = (rho_bulk-minimise.DFT.bulk_density)/dT

    # Return appropriate measures
    if return_profile:
        return result, bulk, minimise_new
    else:
        return result, bulk


def local_thermal_susceptibility(minimise):

    """
	Calculates the local thermal susceptibility and its value in bulk.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

	Returns:
			Local thermal susceptibility
			Bulk thermal susceptibility
    """

    chi_T, bulk = temperature_derivative(minimise)

    return chi_T, bulk


def mu_derivative(minimise, mode, return_profile=False):

    """
	Calculates the chemical potential derivative of the density profile.
	Returns the appropriate output for the mode.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object
			mode (string):           Derivatives to calculate.
									 Available modes are:
										compressibility: drho(r)/dmu
										surface_tension: dgamma/dmu (gamma = surface tension)
										all:             drho(r)/dmu and dgamma/dmu
		Optional:
			return_profile (bool):   Return density profile at new chemical
									 potential
									 Default is false

		Returns:

			if mode == 'compressibility':
				returns local compressibility profile

			elif mode == 'surface tension':
				returns derivative of surface tension with
						respect to chemical potential

			elif mode == 'all':
				returns local compressibility profile,
						derivative of surface tension with
						respect to chemical potential

		Example:
			>>> cDFT.measures.mu_derivative(planar_obj, 'all')
    """


    # Define new DFT and minimisation objects
    temp = deepcopy(minimise.DFT)
    temp.update_state_point(temp.bulk_density + 1e-8)

    # Define new minimisation object. Only the DFT within the object is changed.
    # The object is identified as a derivative, so no output files will be written.
    params = minimise.copy_parameters()

    if minimise.Rs is not None:
        minimise_new = minimisation.spherical(temp,minimise.Rs,deriv=True,**params)
    elif minimise.slit:
        minimise_new = minimisation.slit(temp, **params, deriv = True)
    else:
        minimise_new = minimisation.planar(temp,**params,deriv=True)

    # Find equilibrium density profile
    minimise_new.minimise()

    # Calculate chemical potentials to find difference
    mu_old = minimise.DFT.mu + minimise.DFT.T*np.log(minimise.DFT.bulk_density)
    mu_new = minimise_new.DFT.mu + minimise_new.DFT.T*np.log(minimise_new.DFT.bulk_density)
    dmu = mu_new-mu_old

    # Calculate and return the appropriate measures
    if mode == 'compressibility':

        result = (minimise_new.DFT.density[:]-minimise.DFT.density[:])/dmu

        if return_profile:
            return result, minimise_new
        else:
            return result

    elif mode =='surface tension':

        old_surface_tension = surface_tension(minimise)
        new_surface_tension = surface_tension(minimise_new)
        result = (new_surface_tension-old_surface_tension)/dmu
        if return_profile:
            return result, minimise_new
        else:
            return result

    elif mode == 'all':

        chi = (minimise_new.DFT.density[:]-minimise.DFT.density[:])/dmu
        old_surface_tension = surface_tension(minimise)
        new_surface_tension = surface_tension(minimise_new)
        Gamma = (new_surface_tension-old_surface_tension)/dmu

        if return_profile:
            return chi, Gamma, minimise_new
        else:
            return chi,Gamma


def mu_derivative_measures(minimise, chi = True, sum_rule = True,
						   pout = False, fout = False):

    """
	Calculates measures which require derivatives with respect to
	the chemical potential.
	These derivatives are:
		local compressibility: drho(r)/dmu
		adsorption sum rule: dgamma/dmu

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object
		Optional:
			chi (bool): Calculate local compressibility
						Default is True
			sum_rule (bool): Calculate adsorption sum rule
							 Default is True
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False

		Returns:
			if chi:
				returns local compressibility profile,
						adsorption
						surface tension derivative with chemical potential
						relative error in sum rule
			else:
				returns adsorption
						surface tension derivative with chemical potential
						relative error in sum rule
    """

    if chi:
        compressibility, gamma_deriv = mu_derivative(minimise,'all')
    else:
        gamma_deriv = mu_derivative(minimise,'surface tension')

    gamma_deriv*=-1.0
    adsorp = adsorption(minimise)
    error = abs((gamma_deriv - adsorp)/adsorp)

     # Print out if applicable
    if pout:
        print(f'\n----------------------------------------------------')
        print(f'Adsorption = {adsorp:.10f} -dgamma/dmu = {gamma_deriv:.10f}')
        print(f'Relative Error = {error:.10f}')
        print(f'----------------------------------------------------\n')

    # If applicable, write the result to the output file
    if fout:
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'\nAdsorption (Gamma) = {adsorp:.12f}\t-dgamma/dmu = {gamma_deriv:.12f}\n')
            out.write(f'Relative error = {error:.12f}\n')
    if chi:
        return compressibility, adsorp, gamma_deriv, error
    else:
        return adsorp, gamma_deriv, error


def local_compressibility(minimise):

    """
	Calculates the local compressibility.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object
	Returns:
			Local compressibility
    """

    compressibility = mu_derivative(minimise, 'compressibility')

    return compressibility

def bulk_compressibility(DFT):

    """
	Calculates the bulk compressibility.

	Args:
		Required:
			minimise (minimise_obj): Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			Bulk compressibility
    """

    # Hard-sphere part from FMT
    if DFT.functional == 'RF':
        chi = 1 + 4*DFT.eta + 4*DFT.eta*DFT.eta
        chi *= DFT.T
        chi /= (np.power(1-DFT.eta,4.0))

    # For a single component system, WB and WBII compressibility will
    # be identical
    elif DFT.functional =='WB' or DFT.functional == 'WBII':
        chi = 1 + 4*DFT.eta + 4*np.power(DFT.eta,2.0) -\
                4*np.power(DFT.eta,3.0) + np.power(DFT.eta,4.0)
        chi *= DFT.T
        chi /= (np.power(1-DFT.eta,4.0))

    # Attractive contribution
    if DFT.fluid_type == 'TLJ':
        chi += 2*potentials.TLJ_pressure(DFT)/DFT.bulk_density
    elif DFT.fluid_type == 'LJ':
        chi += 2*potentials.LJ_pressure(DFT)/DFT.bulk_density

    chi = 1.0/chi; chi*=DFT.bulk_density;

    return chi

def soft_substrate_sum_rule(minimise, pout=False, fout = False):

    """
	Calculates the sum rule for a substrate which interacts with the fluid
	through an attractive potential.

	Args:
		Required:
			minimise (minimise_obj):  Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			LHS, RHS and relative error of sum rule
    """

    # Identify grid points with non-infinite external potential

    # Planar Geometry
    if minimise.Rs is None:

        # Calculate appropriate attraction for external potential type
        if minimise.wall_type == 'LJ':
            points = np.ones(minimise.DFT.N, dtype = bool)
            points[:minimise.NiW] = False
            points[minimise.DFT.end:] = False
            RHS = -(6.0/5.0)*(np.power(minimise.sigma_wall,9.0)*np.power((1.0/(minimise.r[points]-minimise.r[minimise.NiW-1])),10.0))
            RHS[:]+=3.0*(np.power(minimise.sigma_wall,3.0)*np.power(1.0/(minimise.r[points]-minimise.r[minimise.NiW-1]),4.0))
            RHS[:]*=minimise.DFT.density[points]*minimise.DFT.dr
            RHS = minimise.DFT.T*minimise.DFT.density[minimise.NiW]-minimise.epsilon_wall*np.sum(RHS[:])

        elif minimise.wall_type == 'SLJ':
            points = minimise.Vext<500
            points[minimise.DFT.end:] = False
            shifted_r = minimise.r[points] + np.power(0.4,1./6.)
            RHS = -(6.0/5.0)*(np.power(minimise.sigma_wall,9.0)*np.power(shifted_r[:],-10.))
            RHS[:]+=3.0*(np.power(minimise.sigma_wall,3.0)*np.power(shifted_r[:],-4.0))
            RHS[:]*=minimise.DFT.density[points]*minimise.DFT.dr
            RHS = minimise.DFT.T*minimise.DFT.density[minimise.NiW] - minimise.epsilon_wall*np.sum(RHS[:])

        elif minimise.wall_type == 'WCALJ':

            pa = 3.*np.power(minimise.sigma_wall,3.)*np.power(minimise.r[minimise.zmask]-minimise.r[minimise.NiW-1],-4.)
            pa -=1.2*np.power(minimise.sigma_wall,9.)*np.power(minimise.r[minimise.zmask]-minimise.r[minimise.NiW-1],-10.)
            pb = 1.5*(minimise.rminw*minimise.rminw-np.power(minimise.r[minimise.izmask]-minimise.r[minimise.NiW-1],2.))\
                /np.power(minimise.sigma_wall,3.)
            pb -= 1.2*np.power(minimise.sigma_wall,9.)*np.power(minimise.rminw,-10.)
            pb += 3.*np.power(minimise.sigma_wall,6.)*np.power(minimise.rminw,-4.)
            pa *= minimise.DFT.density[minimise.zmask]
            pb *= minimise.DFT.density[minimise.izmask]

            RHS = minimise.DFT.T*minimise.DFT.density[minimise.NiW] - \
                minimise.epsilon_wall*(np.sum(pa[:]) + np.sum(pb[:]))*minimise.DFT.dr

        # Calculate relative error
        Error = (abs(minimise.DFT.pressure-RHS)/minimise.DFT.pressure)

        # Print out if applicable
        if pout:
            print(f'\n----------------------------------------------------')
            print(f'Pressure = {minimise.DFT.pressure:.10f} Sum Rule = {RHS:.10f}')
            print(f'Relative Error = {Error:.10f}')
            print(f'----------------------------------------------------\n')

        # If applicable, write the result to the output file
        if fout:
            with open(minimise.output_file_name, 'a') as out:
                out.write(f'\nPressure = {minimise.DFT.pressure:.12f} kbTrho(Rs) + LJ Wall Terms = {RHS:.12f}\n')
                out.write(f'Relative Error = {Error:.12f}\n\n')

        return minimise.DFT.pressure, RHS, Error

    # Spherical geometry
    else:

        # Calculate attrative part for the appropriate external potential
        if minimise.wall_type == 'LJ':
            points = np.ones(minimise.DFT.N, dtype = bool)
            points[:minimise.NiW] = False
            points[minimise.DFT.end:] = False

            rR_plus = minimise.r[points]+minimise.Rs - minimise.DFT.dr
            rR_minus = minimise.r[points]-minimise.Rs + minimise.DFT.dr

            pressure = 1.2*np.power(minimise.sigma_wall,9.0)*(minimise.r[points]*minimise.r[points]*(np.power(rR_minus,-10.0)+np.power(rR_plus,-10.0))-\
                            minimise.r[points]*(np.power(rR_minus,-9.0) + np.power(rR_plus,-9.0)))
            pressure[:] += 3*np.power(minimise.sigma_wall,3.0)*((minimise.r[points]*(np.power(rR_minus,-3.0)+np.power(rR_plus,-3.0)))-\
                    (minimise.r[points]*minimise.r[points]*(np.power(rR_minus,-4.0)+np.power(rR_plus,-4.0))))
            pressure[:]*=minimise.DFT.density[points]*minimise.DFT.dr

        elif minimise.wall_type == 'SLJ':

            points = minimise.Vext<500
            points[minimise.DFT.end:] = False

            shifted_r = minimise.r[points] + (minimise.shift-minimise.NiW)*minimise.DFT.dr

            rR_plus = 1.0/(shifted_r + minimise.Rs)
            rR_minus = 1.0/(shifted_r - minimise.Rs)
            shifted_r = 1./shifted_r

            pressure = 1.2*np.power(minimise.sigma_wall,9.0)*((np.power(rR_minus,10.0)+np.power(rR_plus,10.0))-\
                            shifted_r*(np.power(rR_minus,9.0) + np.power(rR_plus,9.0)))
            pressure[:] += 3.*np.power(minimise.sigma_wall,3.0)*((shifted_r*(np.power(rR_minus,3.0)+np.power(rR_plus,3.0)))-\
                    ((np.power(rR_minus,4.0)+np.power(rR_plus,4.0))))

            pressure[:] *= minimise.DFT.density[points] * minimise.r[points] * minimise.r[points] * minimise.DFT.dr

        LHS,RHS = spherical_contact_sum_rule_b(minimise)
        pressure = minimise.epsilon_wall*np.sum(pressure[:]) /(minimise.Rs*minimise.Rs)
        LHS = pressure + minimise.DFT.T*minimise.DFT.density[minimise.NiW]

        # Calculate relative error
        Error = abs(LHS-RHS)/LHS

        # Print out if applicable
        if pout:
            print(f'\n----------------------------------------------------')
            print(f'Potential Side = {LHS:.10f} Sum Rule = {RHS:.10f}')
            print(f'Relative Error = {Error:.10f}')
            print(f'----------------------------------------------------\n')

        # If applicable, write the result to the output file
        with open(minimise.output_file_name, 'a') as out:
            out.write(f'\np + 2gamma/Rs + dgamma/dRs = {RHS:.12f} kbTrho(Rs) + LJ Wall Terms = {LHS:.12f}\n')
            out.write(f'Relative Error = {Error:.12f}\n\n')

        return LHS, RHS, Error


def slit_sum_rule(minimise, pout = False, fout = False):

    """
	Calculates the sum rule for a slit geometry.

	Args:
		Required:
			minimise (minimise_obj):  Planar or spherical minimisation object

		Optional:
			pout (bool): Print results of sum rule to terminal
						 Default is False
			fout (bool): Write results of sum rule to file
						 Default is False
	Returns:
			LHS, RHS and relative error of sum rule
    """

    params_L = minimise.DFT.copy_parameters()
    params_L["L"] += 2.*minimise.DFT.dr
    temp = minimisation.DFT(**params_L)

    params = minimise.copy_parameters()

    # Measure solvation force
    L_deriv = minimisation.slit(temp, **params,deriv=True)
    L_deriv.minimise()
    old_gamma = surface_tension(minimise)
    new_gamma = surface_tension(L_deriv)
    fs = -1.*minimise.DFT.beta*(new_gamma-old_gamma)/(temp.L-minimise.DFT.L)

    # Find equilibrium density profile of single wall geometries
    left = {"wall_type": params["left_wall_type"], "epsilon_wall":params["left_epsilon_wall"],
			"sigma_wall": params["left_sigma_wall"],"alpha":params["alpha"],
			"ng":params["ng"]}
    right = {"wall_type": params["right_wall_type"], "epsilon_wall":params["right_epsilon_wall"],
			"sigma_wall": params["right_sigma_wall"],"alpha":params["alpha"],
			"ng":params["ng"]}

    left_wall = minimisation.planar(minimise.DFT, **left, deriv=True)
    right_wall = minimisation.planar(minimise.DFT, **right, deriv=True)
    left_wall.minimise()
    right_wall.minimise()

    # Calculate each side of sum rule
    if left["wall_type"] == 'SLJ':

        # Calculate derivative of attractive potential
        shifted_r = minimise.r[:] + np.power(0.4,1./6.)*minimise.sigma_wall
        left_att = -1.2*(np.power(minimise.sigma_wall,9.)*np.power(shifted_r,-10.))
        left_att[:] += (3.*np.power(minimise.sigma_wall,3.)*np.power(shifted_r,-4.))

        # Calculate repulsive part of sum rule
        LHS = np.sum(left_att[minimise.rmask]*\
				 minimise.DFT.density[minimise.rmask])*minimise.DFT.dr*minimise.epsilon_wall

        LHS -= np.sum(left_att[minimise.rmask]*\
				 left_wall.DFT.density[minimise.rmask])*left_wall.DFT.dr*left_wall.epsilon_wall
        infinite_correction = (2./15.)*np.power(left_wall.sigma_wall,9.)*np.power(left_wall.DFT.L+\
									np.power(0.4,1./6.)*left_wall.sigma_wall,-9.)
        infinite_correction -= np.power(left_wall.sigma_wall,3.)*np.power(left_wall.DFT.L+\
									np.power(0.4,1./6.)*left_wall.sigma_wall,-3.)
        LHS += left_wall.DFT.bulk_density*left_wall.epsilon_wall*infinite_correction

        LHS *= minimise.DFT.beta

        # Calculate hard-sphere part of sum rule
        LHS = (minimise.DFT.density[minimise.NiW] - left_wall.DFT.density[left_wall.NiW]) - LHS

    elif left["wall_type"] == 'HW':
        LHS = minimise.DFT.density[minimise.NiW] - left_wall.DFT.density[left_wall.NiW]

    if right["wall_type"] == 'SLJ':

        shifted_r = (minimise.r[minimise.rmask] + np.power(0.4,1./6.)*right_wall.sigma_wall)
        right_att = 1.2*(np.power(minimise.right_sigma_wall,9.)*np.power(shifted_r,-10.))
        right_att -= 3.*np.power(minimise.right_sigma_wall,3.)*np.power(shifted_r,-4.)
        right_att *= -1.*minimise.right_epsilon_wall

        # Calculate attractive part of sum rule
        RHS = np.sum(np.flip(right_att[:])*\
				 minimise.DFT.density[minimise.rmask])*minimise.DFT.dr

        RHS -= np.sum(right_att[:]*\
				 right_wall.DFT.density[right_wall.rmask])*right_wall.DFT.dr

        infinite_correction = (2./15.)*np.power(right_wall.sigma_wall,9.)*np.power(right_wall.DFT.L+\
									np.power(0.4,1./6.)*right_wall.sigma_wall,-9.)
        infinite_correction -=np.power(right_wall.sigma_wall,3.)*np.power(right_wall.DFT.L+\
									np.power(0.4,1./6.)*right_wall.sigma_wall,-3.)
        RHS += right_wall.DFT.bulk_density*right_wall.epsilon_wall*infinite_correction
        RHS *= -1.*minimise.DFT.beta

        # Calculate repulsive part of sum rule
        RHS += (minimise.DFT.density[minimise.DFT.end-1] - right_wall.DFT.density[right_wall.NiW])

    elif right["wall_type"] == 'HW':
        RHS = minimise.DFT.density[minimise.DFT.end-1] - right_wall.DFT.density[right_wall.NiW]

    if (left_wall.wall_type != 'HW' and right_wall.wall_type != 'HW'):
        error = abs(LHS-RHS)/LHS
    else:
        error = abs(LHS-RHS)

    # Print out if applicable
    if pout:
        print(f'\n----------------------------------------------------')
        print(f'Left Wall = {LHS:.14f}\nRight Wall = {RHS:.14f}')
        print(f'Relative Error = {error:.14f}')
        print(f'------------------------------------------------------')

        if (left_wall.wall_type != 'HW' and right_wall.wall_type != 'HW'):
            print(f'Solvation Force = {fs:.14f}')
            print(f'Relative Errors: LHS: {abs(LHS-fs)/fs:.10f}')
            print(f'Relative Errors: RHS: {abs(RHS-fs)/fs:.10f}')
            print(f'----------------------------------------------------\n')

    # If applicable, write the result to the output file
    with open(minimise.output_file_name, 'a') as out:
            out.write(f'\nLeft Wall {LHS:.12f} Right Wall = {RHS:.12f}\n')
            out.write(f'Relative Error = {error:.12f}\n\n')
            if (left_wall.wall_type != 'HW' and right_wall.wall_type != 'HW'):
                out.write(f'Solvation Force = {fs:.14f}\n')
                out.write(f'Relative Error LHS: {abs(LHS-fs)/fs:.10f}')
                out.write(f' Relative Error RHS: {abs(RHS-fs)/fs:.10f}\n\n')

    return LHS,RHS,error

def liquid_vapour_surface_tension(temperature,vapour_density,liquid_density,
                                  functional='RF', cut_off = 2.5,
                                  file_path = './Surface_Tension/'):

    """
	Calculates the surface tension of a liquid-vapour interface for a truncated
	Lennard-Jones fluid.
	It is important to supply an accurate liquid and vapour density.

	Args:
		Required:
			temperature (float):    Temperature of bulk fluid
			vapour_density (float): Vapour density at coexistence
			liquid_density (float): Liquid density at coexistence

		Optional:
			functional (string): FMT functional to use.
								 Options:
									  'RF' = Rosenfeld
									  'WB' = White-Bear
									  'WBII' = White-BEar Mark II
								 Default is 'RF'
			cut_off (float):     Radius of truncation for attractive
							     fluid interactions
								 In units of fluid particle diameters
								 Default is 2.5
			file_path (string):        File path for output file

	Returns:
			liquid-vapour surface tension
    """

    # Set up DFT object
    DFT = minimisation.DFT(liquid_density, temperature, L = 200.0, dr =  0.005, functional = functional,
						   fluid_type =  'TLJ', cut_off = cut_off)
    DFT.vapour_density = vapour_density

    # Set up minimisation object
    planar = minimisation.planar(DFT, alpha = 0.005, ng = 1000, file_path = file_path ,
								  wall_type = 'LV', deriv=True)
    planar.output_filename = file_path + './LV_Surface_Tension_' + str(temperature) + '_' + functional

    # Perform numerical minimisation to find equilibrium density profile
    planar.attempts = 0
    while  planar.dev>1e-12 and planar.attempts<1000000 and not planar.fail:

        planar.weighted_densities()
        planar.correlation()
        planar.update()
        planar.attempts+=1

        if planar.attempts%100 == 0:
            print(f'{planar.attempts} complete. Deviation: {planar.dev}\n')

    if (planar.attempts<1000000):
        print(f'Convergence achieved in {planar.attempts} attempts.')
    else:
        print(f'Density profile failed to converge after {planar.attempts} attempts.')

    # Calculate surface tension
    lv_gamma =  surface_tension(planar)

    # Write final profile and simulation details to file
    pres = 0; dr = planar.DFT.dr;
    while dr<1:
         dr*=10; pres+=1;

    with open(planar.output_filename, 'a') as out:
        out.write(f'Produced {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
        out.write(f'Fluid Type = {planar.DFT.fluid_type}\n')
        if (planar.DFT.fluid_type == 'TLJ'):
            out.write(f'cut off radius: {planar.DFT.cut_off}sigma\n\n')

        out.write(f'Liquid Density = {liquid_density:.10f}\nVapour Density = {vapour_density:.10f}\n')
        out.write(f'Temperature = {temperature:.10f}\nR = {planar.DFT.R}\nSigma = {planar.DFT.sigma}\n')
        out.write(f'L = {planar.DFT.L}sigma\nN = {planar.DFT.N}\ndr = {planar.DFT.dr}\n\nFunctional = {planar.DFT.functional}\n')
        out.write(f'Liquid Excess Chemical Potential = {planar.DFT.mu:.12f}\n')
        out.write(f'Liquid Chemical Potential = {planar.DFT.mu + planar.DFT.T*np.log(planar.DFT.bulk_density):.12f}\n\n')
        out.write(f'Convergence in {planar.attempts} attempts.\n\n')
        out.write(f'i\tr\trho\n')
        for i in range(planar.DFT.N):
            out.write(f'{i}\t{planar.r[i]:.{pres}f}\t{planar.DFT.density[i]:.12f}\n')

        out.write(f'\nLiquid-Vapour Surface Tension = {lv_gamma:.12f}')

    # Print out surface tension
    print(f'Liquid Vapour Surface Tension: {lv_gamma:.12f}')
    return lv_gamma
