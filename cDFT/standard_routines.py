#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:22:37 2019

@author: mkcoe
"""

import numpy as np
import os
import cDFT.minimisation as minimisation
import cDFT.functionals as functionals

#mpl.use('Agg')

def excess_grand_potential(geometry, minimise):
     
    """
    Finds the grand potential from eq XXX in supporting documentation.
    """

    # The spherical sum rules require the full chemical potential (mu = mu_ideal + mu_excess) 
    # whereas the minimisation procedure requires only mu_excess therefore here we have to
    # define the full chemical potential to be used.
    mu = minimise.DFT.mu + minimise.DFT.Temp*np.log(minimise.DFT.rho_bulk)
    
    phi = minimise.DFT.Temp*minimise.DFT.free_energy(minimise.DFT)

    phi[minimise.NiW:minimise.DFT.end] += minimise.DFT.Temp*minimise.DFT.rho[minimise.NiW:minimise.DFT.end]*\
                                            (np.log(minimise.DFT.rho[minimise.NiW:minimise.DFT.end])-1.0)
    phi[minimise.NiW:minimise.DFT.end] += minimise.DFT.rho[minimise.NiW:minimise.DFT.end]*\
                                                (minimise.DFT.Vext[minimise.NiW:minimise.DFT.end] - mu)
    
    if geometry == 'spherical':
        phi[:]*=4.0*np.pi*minimise.r[:]*minimise.r[:]
    
    phi[:]*=minimise.DFT.dr
    
    return phi[:]
    
def surface_tension(geometry, minimise):
    
    """
    Returns the surface tension from eq XXX in the supporting documentation.
    """
    gamma = excess_grand_potential(geometry,minimise)
   
    # Surface tension requires full grand potential, therefore add ideal contribution of pV
    if geometry == 'spherical':
        gamma[minimise.NiW:minimise.DFT.end] += minimise.DFT.pressure*minimise.DFT.dr*(4.0*\
                 np.pi*minimise.r[minimise.NiW:minimise.DFT.end]*minimise.r[minimise.NiW:minimise.DFT.end])
    
    elif geometry == 'planar':
        gamma[minimise.NiW:minimise.DFT.end] += minimise.DFT.pressure*minimise.DFT.dr
        
    # Sum to find the grand potential. Surface tension requires division by dividing surface.
    gamma = np.sum(gamma[1:minimise.DFT.end])
   
    if geometry == 'spherical':
        gamma /= (4.0*np.pi*(minimise.Rs)**2)
    
    return gamma
    
def adsorption(geometry,minimise):
    
    """
    Finds adsorption from eq XXX in supporting documentation.
    Need to think more carefully about whether it is R^2 or R or R^3 to multiply adsorption by.
    """
    
    drho = minimise.DFT.rho[minimise.NiW:minimise.DFT.end]- minimise.DFT.rho_bulk
    
    if geometry == 'spherical':
        drho[:]*=minimise.r[minimise.NiW:minimise.DFT.end]*minimise.r[minimise.NiW:minimise.DFT.end]
        adsorption = minimise.DFT.dr*np.sum(drho)/((minimise.Rs)**2)
    elif geometry == 'planar':
        adsorption = np.sum(drho)*minimise.DFT.dr

    return adsorption   
    
def plot_by_wall_radius(DFT, alpha, file_path, mode):
    
    """
    Runs multiple radii to find either contact, surface tension or excess grand potential 
    as a function of the wall radii. Currently, the minimium wall is Rs = 3R as additional
    factors must be taken into account below this. 
    """
    file_path = file_path + 'spherical_' + str(DFT.eta) + '_' + str(DFT.functional) +'/'
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    max_wall = (0.75*DFT.N)//DFT.NiR; min_wall = 2.*DFT.R;
   
    Radii  = np.geomspace(min_wall,max_wall,num=40)
    
    results = np.zeros((40,2))
    if mode=='contact' or 'excess grand potential':
        results[:,0] = DFT.R/(Radii[:]+DFT.R)
    elif mode == 'surface tension':
        results[:,0] = DFT.R/(Radii[:])

    for indx,rad in enumerate(Radii):
        minimise = minimisation.spherical(DFT,rad,alpha,file_path)
        minimise.minimise()
        if mode=='contact':
            results[indx,1] = minimise.DFT.rho[minimise.NiW]*minimise.DFT.R*minimise.DFT.R*minimise.DFT.R
        else:
            # Excess grand potential found from eq XXX in supporting documentation
            if mode == 'surface tension':
                func_sum = surface_tension('spherical',minimise)
            elif mode == 'excess grand potential':
                func_sum = np.sum(excess_grand_potential('spherical',minimise))
            
            results[indx,1] = func_sum*minimise.DFT.R*minimise.DFT.R*minimise.DFT.beta
        del minimise
    
    return results
    
def spherical_derivative(DFT, Rs, alpha, mode, file_path, dR=None):
    
    """
    Finds the derivative of the surface tension or excess grand potential with respect to
    the radius of a spherical wall.
    """
    
    # First minimise for given Rs
    spherical = minimisation.spherical(DFT, Rs, alpha, file_path)
    spherical.minimise()
    
    # Find surface tension or excess grand potential of first minimisation.
    if mode == 'excess grand potential':
        old = excess_grand_potential('spherical', spherical)
        old = np.sum(old[1:spherical.DFT.end])
    elif mode == 'surface tension':
        old = surface_tension('spherical', spherical)
    elif mode == 'all':
        old = excess_grand_potential('spherical', spherical)
        old_omega = np.sum(old[1:spherical.DFT.end])
        old[spherical.NiW:spherical.DFT.end] += spherical.DFT.pressure*spherical.DFT.dr*(4.0*\
                 np.pi*spherical.r[spherical.NiW:spherical.DFT.end]**2.0)
        old_gamma = np.sum(old[1:spherical.DFT.end])/(4.0*np.pi*(spherical.Rs)**2.0)
    
    # Save the important data, and increase the radius by a small amount.    
    NiW_old = spherical.NiW
    rho_Rs = spherical.DFT.rho[spherical.NiW]
    Rs_new = Rs+(4*spherical.DFT.dr)
    del spherical
    
    # Perform the routine again for the slightly larger radius
    spherical = minimisation.spherical(DFT, Rs_new, alpha, file_path)
    spherical.minimise()
    
    # Find surface tension or excess grand potential of second minimisation.
    if mode == 'excess grand potential':
        new = excess_grand_potential('spherical', spherical)
        new = np.sum(new[1:spherical.DFT.end])
    elif mode == 'surface tension':
        new = surface_tension('spherical', spherical)
    elif mode == 'all':
        new = excess_grand_potential('spherical', spherical)
        new_omega = np.sum(new[1:spherical.DFT.end])
        new[spherical.NiW:spherical.DFT.end] += spherical.DFT.pressure*spherical.DFT.dr*(4.0*\
                 np.pi*spherical.r[spherical.NiW:spherical.DFT.end]**2.0)
        new_gamma = np.sum(new[1:spherical.DFT.end])/(4.0*np.pi*(spherical.Rs)**2.0)
    
    # Finally find the derivative. 
    if mode == 'all':
        deriv_gamma = (new_gamma-old_gamma)/(DFT.dr*(spherical.NiW-NiW_old))
        deriv_omega = (new_omega-old_omega)/(DFT.dr*(spherical.NiW-NiW_old)) 
    else:
         deriv =(new-old)/(DFT.dr*(spherical.NiW-NiW_old))
    
    del spherical 
   
    if mode == 'excess grand potential':
        return deriv, rho_Rs
    elif mode == 'surface tension':
        return deriv, old, rho_Rs
    elif mode == 'all':
        return deriv_omega, deriv_gamma, rho_Rs, old_gamma
     
def spherical_omega_sum_rule(DFT, Rs, alpha, file_path,dR=None):
    
    """
    Performs first sum rule eq XXX in supporting documentation (Byrk Rule). 
    """
    
    LHS, rho_Rs = spherical_derivative(DFT, Rs, alpha, 'excess grand potential', file_path,dR)
    RHS = 4.0*np.pi*(Rs+DFT.R)*(Rs+DFT.R)*rho_Rs
    
    print(f'\n--------------------------------------------')
    print(f'Spherical Excess Grand Potential Sum Rule Results:')
    print(f'beta*d(omega)/(dRb) = {LHS:.10f}')
    print(f'4piRs^2rho[Rs] = {RHS:.10f}')
    print(f'Relative Error = {abs(LHS-RHS)/RHS:.10f}')
    print(f'--------------------------------------------\n')
    
    return LHS, RHS

def spherical_surface_tension_sum_rule(DFT, Rs, alpha, file_path,dR=None):
    
    """
    Performs sum rule eq XXX in supporting documentation (Maria rule)
    """
    
    deriv, gamma_old, LHS = spherical_derivative(DFT, Rs, alpha, 'surface tension', file_path,dR)
    RHS = DFT.beta*(DFT.pressure + 2.0*gamma_old/(Rs+DFT.R) + deriv)
    
    print(f'\n--------------------------------------------')
    print(f'Spherical Surface Tension Sum Rule Results:')
    print(f'rho[Rs] = {LHS:.10f}\t sum_rule = {RHS:.10f}')
    print(f'Relative Error = {abs(LHS-RHS)/LHS:.10f}')
    print(f'--------------------------------------------\n')
    
    return LHS, RHS 

def spherical_surface_tension_and_omega_rule(DFT, Rs, alpha, file_path, dR=None):
    
    LHS_omega, deriv_gamma, LHS_gamma, old_gamma = spherical_derivative(DFT, Rs, alpha, 'all', file_path, dR)
    RHS_omega = 4.0*np.pi*(Rs+DFT.R)*(Rs+DFT.R)*LHS_gamma
    RHS_gamma = DFT.beta*(DFT.pressure + 2.0*old_gamma/(Rs+DFT.R) + deriv_gamma)
    
    print(f'\n----------------------------------------------------')
    print(f'Spherical Sum Rule Results:')
    print(f'Excess Grand Potential:')
    print(f'beta*d(omega)/(dRb) = {LHS_omega:.10f}')
    print(f'4piRs^2rho[Rs] = {RHS_omega:.10f}')
    print(f'Relative Error = {abs(LHS_omega-RHS_omega)/RHS_omega:.10f}')
    print(f'Surface Tension:')
    print(f'rho[Rs] = {LHS_gamma:.10f}\t sum_rule = {RHS_gamma:.10f}')
    print(f'Relative Error = {abs(LHS_gamma-RHS_gamma)/LHS_gamma:.10f}')
    print(f'-----------------------------------------------------\n')
    
    return LHS_omega, RHS_omega, LHS_gamma, RHS_gamma
    
def adsorption_sum_rule(DFT, alpha, file_path, geometry, Rs = None):
    
    if geometry == 'planar':
        minimise = minimisation.planar(DFT, alpha, file_path)
    elif geometry == 'spherical':
        minimise = minimisation.spherical(DFT, Rs, alpha, file_path)
        
    minimise.minimise()
    mu_old = minimise.DFT.mu + minimise.DFT.Temp*np.log(minimise.DFT.rho_bulk)
    if geometry == 'planar':    
        adsorp = adsorption('planar', minimise)
        gamma_old = surface_tension('planar',minimise)
    elif geometry == 'spherical':
        adsorp = adsorption('spherical',minimise)
        gamma_old = surface_tension('spherical',minimise)
    
    del minimise
   
    new_DFT = minimisation.DFT(DFT.eta*(1.00001),DFT.Vext_type,DFT.R,DFT.Temp,DFT.N,DFT.dr,DFT.functional)
    if geometry == 'planar':
        minimise = minimisation.planar(new_DFT, alpha, file_path)
    elif geometry == 'spherical':
        minimise = minimisation.spherical(new_DFT, Rs, alpha, file_path)
       
    mu_new = minimise.DFT.mu + minimise.DFT.Temp*np.log(minimise.DFT.rho_bulk)
    minimise.minimise()
    if geometry == 'planar':    
        gamma_new = surface_tension('planar',minimise)
    elif geometry == 'spherical':
        gamma_new = surface_tension('spherical',minimise)
    
    del minimise
  
    dmu = mu_new-mu_old
    deriv = -1.0*(gamma_new-gamma_old)/dmu
    
    print(f'\n----------------------------------------------------')
    print(f'Adsorption = {adsorp:.10f} -dgamma/dmu = {deriv:.10f}')
    print(f'Relative Error = {abs(deriv-adsorp)/adsorp:.10f}')
    print(f'----------------------------------------------------\n')
    return adsorp, deriv