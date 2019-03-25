#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program for Planar and Spherical Geometry.

Copyright Mary Coe m.k.coe@bristol.ac.uk

Created January 2019. Last Update January 2019.

This module contains details about the functionals which can be used in the cDFT.
The currently supported functionals are Rosenfeld, Whitebear, and Whitebear II.

The module is laid out as follows:
    

    Block 1: Common functions used by multiple functionals
    Block 2: Rosenfeld specific functions
    Block 3: Whitebear specific functions
    Block 4: WhitebearII specific functions
    
These blocks include respective pressures, chemical potentials, grand potentials and 
in the case of the WhitebearII, functions relating the phi2 and phi3 within the functional.

"""
import numpy as np

# Common functions and constants
pi = np.pi
pi4 = pi*4.0; pi8 = pi*8.0; pi12 = pi*12.0; pi36 = pi*36.0; pi6 = pi*6.0; pi24 = pi*24.0;

def calculate_dn0(DFT, nzp):
    
    """
    Calculates the n0 derivative. This is the same for the Rosenfeld, Whitebear 
    and WhitebearII functionals. 
    """

    DFT.n3neg[:] = 1.0-DFT.n3[:];
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
    """
    zero_points = DFT.n3 == 0. ; nzp = DFT.n3>0;
    zero_points[DFT.N-2*DFT.NiR:] = True; nzp[DFT.N-2*DFT.NiR:] = False;
    """
    #DFT.n3neg[zero_points] = 1.0
    DFT.n3neg[:] = 1.0-DFT.n3[:]
    free_energy = np.zeros(DFT.N)
    free_energy[:DFT.end] = -1.0*DFT.n0[:DFT.end]*np.log(DFT.n3neg[:DFT.end]) + \
                (DFT.n1[:DFT.end]*DFT.n2[:DFT.end]-DFT.n1v[:DFT.end]*DFT.n2v[:DFT.end])/(DFT.n3neg[:DFT.end]) + \
                        ((DFT.n2[:DFT.end]**3) - 3.0*DFT.n2[:DFT.end]*DFT.n2v[:DFT.end]**2)/(24.0*np.pi*DFT.n3neg[:DFT.end]**2)             
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
    zero_points[DFT.N-2*DFT.NiR:] = True; nzp[DFT.N-2*DFT.NiR:] = False;
    
    DFT.d2[zero_points] = 0.0; DFT.d3[zero_points] = 0.0; DFT.d2v[zero_points] = 0.0
    
    # Start by calculating constants used more than once
    DFT.n3neg[:] = 1.0 - DFT.n3[:]
    n3neg2 = np.zeros(DFT.N); n2v2 = np.zeros(DFT.N); 
    n3neg2[nzp] = DFT.n3neg[nzp]**2; n2v2[nzp] = DFT.n2v[nzp]**2;
    
    # We combine the derivatives in order to reduce the number of
    # operations (see accompanying documentation)
    DFT.d2[nzp] = calculate_dn0(DFT,nzp) / (pi4*(DFT.R)**2)
    DFT.d2[nzp] += calculate_dn1(DFT,nzp) / (pi4*DFT.R)
    DFT.d2[nzp] += DFT.n1[nzp]/DFT.n3neg[nzp] + (DFT.n2[nzp]**2 - n2v2[nzp])/(pi8*n3neg2[nzp])
    
    DFT.d3[nzp] = DFT.n0[nzp]/DFT.n3neg[nzp] + (DFT.n1[nzp]*DFT.n2[nzp] - DFT.n1v[nzp]*DFT.n2v[nzp])/(n3neg2[nzp])
    DFT.d3[nzp] += (DFT.n2[nzp]**3 - 3*DFT.n2[nzp]*n2v2[nzp])/(pi12*DFT.n3neg[nzp]**3)
    
    DFT.d2v[nzp] = calculate_dn1v(DFT,nzp)/(pi4*DFT.R)
    DFT.d2v[nzp] -= (DFT.n1v[nzp]/DFT.n3neg[nzp] + DFT.n2[nzp]*DFT.n2v[nzp]/(pi4*n3neg2[nzp]))


# Whitebear functional specific functions

def Whitebear_free_energy(DFT):
    
    nzp = DFT.n3!=0; nzp[DFT.N-4*DFT.NiR:] = False;
    
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
    zero_points[DFT.N-2*DFT.NiR:] = True; nzp[DFT.N-2*DFT.NiR:] = False;
    
    DFT.d2[zero_points] = 0.0; DFT.d3[zero_points] = 0.0; DFT.d2v[zero_points] = 0.0
   
    # Start by calculating constants used more than once
    DFT.n3neg[:] = 1.0 - DFT.n3[:]
    n3neg2 = np.zeros(DFT.N); n2v2 = np.zeros(DFT.N); n32 = np.zeros(DFT.N);
    n3neg2[nzp] = DFT.n3neg[nzp]**2; n2v2[nzp] = DFT.n2v[nzp]**2; n32[nzp] = DFT.n3[nzp]**2;
  
    # We combine the derivatives in order to reduce the number of operations 
    # (see accompanying documentation)
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
def WhitebearII_free_energy(DFT):
    
    free_energy = np.zeros(DFT.N)
    DFT.n3neg[:DFT.end] = 1.0-DFT.n3[:DFT.end]
    phi2 = calculate_WhitebearII_phi2(DFT)
    phi3 = calculate_WhitebearII_phi3(DFT)
    free_energy[:DFT.end] = -1.0*DFT.n0[:DFT.end]*np.log(DFT.n3neg[:DFT.end])
    free_energy[:DFT.end] += (1.0+(1.0/9.0)*phi2[:DFT.end]*DFT.n3[:DFT.end]**2.0)*(DFT.n1[:DFT.end]*DFT.n2[:DFT.end] - \
                                                   DFT.n1v[:DFT.end]*DFT.n2v[:DFT.end])/(DFT.n3neg[:DFT.end])
    free_energy[:DFT.end] += (1.0 -(4.0/9.0)*phi3[:DFT.end]*DFT.n3[:DFT.end])*(DFT.n2[:DFT.end]**3.0 - \
                                   3.0*DFT.n2[:DFT.end]*DFT.n2v[:DFT.end]**2.0)/(pi24*DFT.n3neg[:DFT.end]**2.0)
    
    return free_energy

def calculate_WhitebearII_pressure():
    pass

def calculate_WhitebearII_chemical_potential():
    pass

def calculate_WhitebearII_phi2(DFT):
    
    """
    Calculates the phi2 function in the WhitebearII functional
    """
    phi2 = 3.0*(DFT.n3[:]*(2-DFT.n3[:]) + 2.0*DFT.n3neg[:]*np.log(DFT.n3neg[:]))/(DFT.n3[:]**3)
    
    return phi2

def calculate_WhitebearII_phi3(DFT):
   
    """
    Calculates the phi2 function in the WhitebearII functional
    """
    
    phi3 = (3.0/(4.0*DFT.n3[:]**3))*(DFT.n3[:]*(2.0-3.0*DFT.n3[:]+2.0*DFT.n3[:]**2) + \
            2.0*(DFT.n3neg[:]**2)*np.log(DFT.n3neg[:]))
    
    return phi3

def calculate_WhitebearII_dphi2(DFT):
    
    """
    Calculates the phi2 function in the WhitebearII functional
    """
    
    dphi2 = -6.0*(DFT.n3[:] + np.log(DFT.n3neg[:])) * \
                        (((3.0*DFT.n3neg[:])/(DFT.n3[:]**4)) + (1.0/(DFT.n3[:]**3)))
    return dphi2

def calculate_WhitebearII_dphi3(DFT):
    
    """
    Calculates the phi2 function in the WhitebearII functional
    """
   
    dphi3 = (3.0/2.0*DFT.n3[:]**3.0)*(DFT.n3[:]*(3.0*DFT.n3[:]-2.0)-2.0*DFT.n3neg[:]*np.log(DFT.n3n3g[:])) -\
                    (DFT.n3[:]*(2.0-3.0*DFT.n3[:]+2.0*DFT.n3[:]**2.0) + \
                                         2.0*np.log(DFT.n3neg[:])*DFT.n3neg[:]**2)/(4.0*DFT.n3[:]**4)
    
    return dphi3

def calculate_WhitebearII_derivatives(DFT):
    
    # Start by calculating the additional factors required for WBII
    phi2 = calculate_WhitebearII_phi2(DFT)
    phi3 = calculate_WhitebearII_phi3(DFT)
    dphi2 = calculate_WhitebearII_dphi2(DFT)
    dphi3 = calculate_WhitebearII_dphi3(DFT)
    
    dphi2[:] = (1.0/3.0)*DFT.n3[:]*phi2[:] + (1.0/9.0)*(DFT.n3[:]**2)*dphi2[:]
    dphi3[:] = (4.0/9.0)*phi3[:] + (4.0/9.0)*DFT.n3[:]*dphi3[:]
    
    n3neg2 = DFT.n3neg[:]**2; n32 = DFT.n3[:]**2
    
    # We combine the derivatives in order to reduce the number of
    # operations (see accompanying documentation)
    DFT.d2[:] = calculate_dn0(DFT) / (pi4*(DFT.R)**2)
    DFT.d2[:] += calculate_dn1(DFT)*(1.0 + (1.0/9.0)*phi2[:])/ (pi4*DFT.R)
    DFT.d2[:] += (1.0+(1.0/9.0)*n32[:]*phi2[:])*DFT.n1[:]/DFT.n3neg[:]
    DFT.d2[:] += (1.0 - (4.0/9.0)*DFT.n3[:]*phi3[:])*(DFT.n1[:]/DFT.n3neg[:])
    DFT.d2[:] += (1.0 - (4.0/9.0)*DFT.n3[:]*phi3[:])*(DFT.n2[:]**2 - DFT.n2v[:]**2)/(pi8*n3neg2[:])
    
    DFT.d3[:] = DFT.n0[:]/DFT.n3neg[:]
    DFT.d3[:] += ((DFT.n1[:]*DFT.n2[:] - DFT.n1v[:]*DFT.n2v[:])/DFT.n3neg2[:]) * \
        ((1.0/9.0)*n32[:]*(phi2[:] + DFT.n3neg[:]*dphi2[:]) + (DFT.n3[:]/3.0)*DFT.n3neg[:]*phi2 + 1.0)
    DFT.d3[:] += (((DFT.n2[:]**3) - 3.0*DFT.n2[:]*DFT.n2v[:]**2)/(pi24*n3neg2[:])) * \
            ((4.0/9.0)*phi3[:]*(1.0-2.0*DFT.n3[:]) +(4.0/9.0)*DFT.n3[:]*dphi3[:] + 2.0)
    
    DFT.dn2v[:] = calculate_dn1v(DFT) * (1.0 + (1.0/9.0)*phi2[:])/(pi4*DFT.R) 
    DFT.dn2v[:] += (DFT.n1v[:]/DFT.n3neg[:]) * (1.0+(1.0/9.0)*(DFT.n3[:]**2)*phi2[:])
    DFT.dn2v[:] += ((DFT.n2[:]*DFT.n2v[:])/(pi4*n3neg2[:]))*(1.0-(4.0/9.0)*DFT.n3[:]*phi3[:])

# Testing functions. To be removed before upload.    
def test_WB_derivatives():
    
    """
    Run this to check the ouput of the derivatives for the WB functional match those
    of the test data provided by Roth. 
    """
    import cDFT.minimisation as minimisation
    test_data = minimisation.DFT(0.2,'Hard Wall',1.0,1.0,256,0.05,'Whitebear')
    test_wdens = np.genfromtxt('../Tests/Test_data/Roth_Weighted_Densities_WB')
    print(test_wdens.shape)
    test_data.n0[1:216] = test_wdens[:,3]
    test_data.n1[1:216] = test_wdens[:,4]
    test_data.n2[1:216] = test_wdens[:,5]
    test_data.n3[1:216] = test_wdens[:,6]
    test_data.n1v[1:216] = test_wdens[:,7]
    test_data.n2v[1:216] = test_wdens[:,8]
    test_data.n3neg[1:216] = 1.0 - test_wdens[:,6]
    
    del test_wdens
    
    calculate_Whitebear_derivatives(test_data)
    with open('../test_WB_derivatives','w') as divs:
        divs.write(f'i     rho            d2           d3           d2v          \n')
        for i in range(1,test_data.N):
            divs.write(f'{i:{3}} {test_data.rho[i]:{7}.{5}} {test_data.d2[i]:{12}.{10}} {test_data.d3[i]:{12}.{10}} {test_data.d2v[i]:{12}.{10}}\n')
    
def test_RF_derivatives():
    
    """
    Run this to check the output of the derivatives for the RF function match those
    of the test data provided by the 'spherical_working.c' Wilding program.

    """
    import cDFT.minimisation as minimisation
    test_data = minimisation.DFT(0.2,'Hard Wall',0.5,1.0,256,0.05,'Rosenfeld')
    test_wdens = np.genfromtxt('../Tests/Test_data/Wilding_Weighted_Densities_RF')
    print(test_wdens.shape)
    test_data.n0[:] = test_wdens[:,1]
    test_data.n1[:] = test_wdens[:,2]
    test_data.n2[:] = test_wdens[:,3]
    test_data.n3[:] = test_wdens[:,4]
    test_data.n1v[:] = test_wdens[:,5]
    test_data.n2v[:] = test_wdens[:,6]
    test_data.n3neg[:] = 1.0 - test_wdens[:,4]
    
    del test_wdens
    
    calculate_Rosenfeld_derivatives(test_data)
    with open('../test_RF_derivatives','w') as divs:
        divs.write(f'i     rho            d2           d3           d2v          \n')
        for i in range(1,test_data.N):
            divs.write(f'{i:{3}} {test_data.rho[i]:{7}.{5}} {test_data.d2[i]:{12}.{10}} {test_data.d3[i]:{12}.{10}} {test_data.d2v[i]:{12}.{10}}\n')

def test_RF_free_energy():

    import cDFT.minimisation as minimisation
    test_data = minimisation.DFT(0.2,'Hard Wall',0.5,1.0,1024,0.005,'Rosenfeld')
    print(test_data.n0.shape)
    test_wdens = np.genfromtxt('../Tests/Test_data/Wilding_Weighted_densities_RF_free_energy')
    test_data.n0[:] = test_wdens[:,1]
    test_data.n1[:] = test_wdens[:,2]
    test_data.n2[:] = test_wdens[:,3]
    test_data.n3[:] = test_wdens[:,4]
    test_data.n1v[:] = test_wdens[:,5]
    test_data.n2v[:] = test_wdens[:,6]
    test_data.n3neg[:] = 1.0 - test_wdens[:,4]
    test_data.rho[:] = test_wdens[:,7]
    free_energy = np.zeros(1024)
    free_energy = Rosenfeld_free_energy(test_data)
    free_energy_excess = np.sum(free_energy[:test_data.N-3*test_data.NiR]*0.005)
    free_energy_ideal = np.sum(test_data.Temp*test_data.rho[test_data.NiR:test_data.N-2*test_data.NiR]*(np.log(test_data.rho[test_data.NiR:test_data.N-2*test_data.NiR])-1.0))*0.005
    free_energy_ideal -= np.sum(test_data.rho[test_data.NiR:test_data.N-2*test_data.NiR]*(test_data.mu+np.log(0.38197)))*0.005
    print(f'Excess = {free_energy_excess:.6f}\t Ideal = {free_energy_ideal:.6f}\t total = {free_energy_excess+free_energy_ideal:.6f}\n')
  
    
if __name__ == "__main__":
    
    pass
    #test_WB_derivatives()
    #print("pressure is {}".format(calculate_Whitebear_pressure(0.2,1.0,1.0)))
    #print("chemical potential is {}".format(calculate_Whitebear_chemical_potential(0.2,1.0,1.0)))
    #test_RF_derivatives()
    #print("RF pressure is {}".format(calculate_Rosenfeld_pressure(0.2,1.0,0.5)))
    #print("Rosenfeld excess chemical potential is {}".format(calculate_Rosenfeld_chemical_potential(0.2,1.0)))
    
   
   
    