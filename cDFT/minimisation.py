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

import numpy as np
import pyfftw as fft
import cDFT.output as output
import os
import cDFT.diagnostics as diagnostics
import cDFT.functionals as functionals
import datetime
import copy

#Constants used regardless of input parameters are defined here
pi = np.pi
pi2 = np.pi*2.0
pi4 = np.pi*4.0
tolerance = 1e-12
maxiter = 500000

class DFT:
    
    def __init__(self,eta, Vext_type, R, Temp, N, dr, functional):
        
        """
        We start by defining the essential ingredients for the DFT. These are:
            eta:          Packing fraction
            Vext_type:    Type of external potential the system is subjected to. 
                          Currently only hard wall is supported.
            Rs:           Radius of hard wall. If using planar, set this to 0 
                          else you are free to choose this.
            R:            Radius of particle in system. 
            Temp:         Temperature of system. Note that kb is set to 1.0, so
                          Beta = 1.0/Temp.
            N:            Size of grid to perform DFT on. This should be a power
                          of 2 in order for fast fourier transforms to work. 
            dr:           Discretisation of the grid. For better accuracy, should
                          be set smaller. Typically about 0.005.
            geometry:     Planar or Spherical.
            functional:   Rosenfeld or Whitebear
        """
        
        # First we set the input parameters to be stored
        self.eta = eta
        self.Vext_type = Vext_type
        self.R = R
        self.Temp = Temp
        self.beta = 1.0/Temp
        self.N = N
        self.dr = dr
        self.functional = functional
        
        # Secondly, we calculate helpful constants. These are 
        #     rho_bulk:     Bulk fluid density, which is used to set intial
        #                   density profile
        #     NiR:          Number of grid points in radius of particle
        self.rho_bulk = ((3.0)/(pi4))* (1.0/(R**3)) * eta
        self.NiR = int(R/dr)
        self.end = N - 4*self.NiR
        
        if functional == 'Rosenfeld':
           self.calculate_derivatives = functionals.calculate_Rosenfeld_derivatives
           self.pressure = functionals.calculate_Rosenfeld_pressure(eta, Temp, R)
           self.mu = functionals.calculate_Rosenfeld_chemical_potential(eta, Temp)
           self.free_energy = functionals.Rosenfeld_free_energy
            
        elif functional == 'Whitebear':
            self.calculate_derivatives = functionals.calculate_Whitebear_derivatives
            self.pressure = functionals.calculate_Whitebear_pressure(eta, Temp, R)
            self.mu = functionals.calculate_Whitebear_chemical_potential(eta,Temp)
            self.free_energy = functionals.Whitebear_free_energy
        else:
            # Whitebear II currently in development
            self.calculate_derivatives = functionals.calculate_WhitebearII_derivatives
       
        # Thirdly, we initialise the required arrays for the DFT. These are split
        # into the weighted densities (n0,n1,n2,n3,n1v,n2v), the combined derivatives 
        # (dn2, dn3, dn2v), the correlation function (c), and the density profile
        # (rho). We also set the external potential, Vext.
        self.n2 = fft.empty_aligned(N, dtype='float64')
        self.n3 = fft.empty_aligned(N, dtype='float64')
        self.n2v = fft.empty_aligned(N, dtype='float64')
        
        # n3neg is a useful array to old 1.0-n3[i], a term used a lot in the functionals.
        # As fourier transforms are not applied to n0,n1,n1v, there is no need to align
        # their memory in a specific way.
        self.n3neg = np.zeros(N)
        self.n0 = np.zeros(N); self.n1 = np.zeros(N); self.n1v = np.zeros(N)
        
        self.d2 = fft.empty_aligned(N, dtype='float64')
        self.d3 = fft.empty_aligned(N, dtype='float64')
        self.d2v = fft.empty_aligned(N, dtype='float64')
        self.c2 = fft.empty_aligned(N, dtype='float64')
        self.c3 = fft.empty_aligned(N, dtype='float64')
        self.c2v = fft.empty_aligned(N, dtype='float64')
        self.c2v_dummy = fft.empty_aligned(N, dtype='float64')
        self.c = np.zeros(N)
        self.rho = fft.empty_aligned(N, dtype='float64')
        self.rho_old = fft.empty_aligned(N, dtype='float64')
        self.Vext = np.zeros(N)

        # Finally, we can set up the weight functions. These depend on the  
        # radius of particles. 
        w2 = fft.empty_aligned(N, dtype='float64')
        w3 = fft.empty_aligned(N, dtype='float64')
        w2v = fft.empty_aligned(N, dtype='float64')
        
        self.fw2 = fft.empty_aligned(int(N//2)+1, dtype='complex128')
        self.fw3 = fft.empty_aligned(int(N//2)+1, dtype='complex128')
        self.fw2v = fft.empty_aligned(int(N//2)+1, dtype='complex128')
        
        # Here we declare the fft object for the weights
        fft_weights = fft.FFTW(w2,self.fw2, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
        
        w2[:] = 0.0; w3[:] = 0.0; w2v[:] = 0.0;
        
        for i in range(0,self.NiR+1):
            w2[i] = pi2*R*dr
            w3[i] = pi*(R*R-i*dr*i*dr)*dr
            w2v[i] = pi2*i*dr*dr
            
            if i>0:
                w2[N-i] = w2[i]
                w3[N-i] = w3[i]
                w2v[N-i] = -1.0*w2v[i]
        
        # Modifying the weight functions slightly helps the numerics (Roth 2010)
        w2[self.NiR]*=3.0/8.0; w2[N-self.NiR] *=3.0/8.0; w3[self.NiR]*=3.0/8.0; 
        w3[N-self.NiR]*=3.0/8.0; w2v[self.NiR]*=3.0/8.0; w2v[N-self.NiR]*=3.0/8.0;
        
        w2[self.NiR-1]*=7.0/6.0; w2[N-self.NiR+1]*=7.0/6.0; w3[self.NiR-1]*=7.0/6.0;
        w3[N-self.NiR+1]*=7.0/6.0; w2v[self.NiR-1]*=7.0/6.0; w2v[N-self.NiR+1]*=7.0/6.0;
        
        w2[self.NiR-2]*=23.0/24.0; w2[N-self.NiR+2]*=23.0/24.0; w3[self.NiR-2]*=23.0/24.0;
        w3[N-self.NiR+2]*=23.0/24.0; w2v[self.NiR-2]*=23.0/24.0; w2v[N-self.NiR+2]*=23.0/24.0;
        
        # Finally we take the fourier transform. We use the same fftw object, and update arrays each
        # time. This is because we only need to calculate these once. 
        fft_weights.execute()
        fft_weights.update_arrays(w3,self.fw3); fft_weights.execute();
        fft_weights.update_arrays(w2v,self.fw2v); fft_weights.execute();

class minimisation:
    
    def __init__(self, cDFT, inplace=False):
        
        """
        In order to prevent redefinition of transforms throughout the program, we
        store all the important ingredients for minimisation here.
        """
        
        # The user is free to choose whether to overwrite the original DFT object or
        # to produce a local copy, accessed via the minimisation object. The default is
        # not to overwrite. This allows the user to define one DFT object and send it to
        # multiple minimisations. 
        if not inplace:
            self.DFT = copy.deepcopy(cDFT)
        else:
            self.DFT = copy.copy(cDFT)
        
        # We define aligned arrays to be used within the convolution
        self.rho = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.frho = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        
        self.fn2 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fn3 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fn2v = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            
        self.fd2 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fd3 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fd2v = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            
        self.fc2 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fc3 = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fc2v = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
        self.fc2v_dummy = fft.empty_aligned(int(self.DFT.N//2)+1, dtype='complex128')
            
        # We then set up the required FFTW objects to perform the fourier transforms.
        # As we will use these continuously, we define one per transform pair
        # The first is the transform of rho. We never need to directly inverse this, so provide
        # only the forward transform.
        self.fft_rho = fft.FFTW(self.rho,self.frho, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
            
        # For the weighted densities, we only require the inverse transform, as they are 
        # calculated in fourier space.
        self.ifft_n2 = fft.FFTW(self.fn2,self.DFT.n2, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
        self.ifft_n3 = fft.FFTW(self.fn3,self.DFT.n3, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
        self.ifft_n2v = fft.FFTW(self.fn2v,self.DFT.n2v, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
            
        # For the derivatives, we need the forward transforms.
        self.fft_d2 = fft.FFTW(self.DFT.d2,self.fd2, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
        self.fft_d3 = fft.FFTW(self.DFT.d3,self.fd3, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
        self.fft_d2v = fft.FFTW(self.DFT.d2v,self.fd2v, direction = 'FFTW_FORWARD', flags = ('FFTW_ESTIMATE',))
            
        # We perform the correlation function calculation in fourier space so only need
        # the backwards transform for this.
        self.ifft_c2 = fft.FFTW(self.fc2,self.DFT.c2, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
        self.ifft_c3 = fft.FFTW(self.fc3,self.DFT.c3, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
        self.ifft_c2v = fft.FFTW(self.fc2v,self.DFT.c2v, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
        self.ifft_c2v_dummy = fft.FFTW(self.fc2v_dummy, self.DFT.c2v_dummy, direction = 'FFTW_BACKWARD', flags = ('FFTW_ESTIMATE',))
        
        # Set the initial devation to an arbritrary number greater than the tolerance
        self.dev = 1.0
        
        # Set up any output directories
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        
    def update(self):
           
        """
        Updates the density profile and compares to the previous profile.
        """
            
        # The correlation function is given by the negative sum of the 
        # components.
        corr = np.zeros(self.DFT.N); rho_new = np.zeros(self.DFT.N);
        corr[1:] = -1.0*(self.DFT.c2[1:] + self.DFT.c3[1:] + self.DFT.c2v[1:])
            
        # The trial density profile is calculated using equation XXX
        rho_new[self.NiW:self.DFT.end] = self.DFT.rho_bulk*np.exp(corr[self.NiW:self.DFT.end] + \
                                           self.DFT.beta*(self.DFT.mu-self.DFT.Vext[self.NiW:self.DFT.end]))
            
        # The old density profile is saved for comparison later
        self.DFT.rho_old[self.NiW:self.DFT.end] = self.DFT.rho[self.NiW:self.DFT.end]
            
        # The new density profile is found using the Picard method.
        self.DFT.rho[self.NiW:self.DFT.end] = (1.0 - self.alpha)*self.DFT.rho[self.NiW:self.DFT.end] + \
                                                                    self.alpha*rho_new[self.NiW:self.DFT.end]
            
        # The maximum deviation is found.
        self.dev = max(abs(self.DFT.rho[self.NiW:self.DFT.end]-self.DFT.rho_old[self.NiW:self.DFT.end]))
            
    def minimise(self):
            
        """
        Minimises the density profile until the deviation between density profiles is less than
        1e-12 or until the maximum number of iterations is reached. This is by default set to 
        500000 however can be chosen by the user if required.
        """

        self.attempts = 0
        while self.dev>1e-12 and self.attempts<1000000:
            
            # In turn, we calculate the weighted densities, then the correlation
            # function, then update the density profile until convergence is
            # complete.
            self.calc_weighted_densities()
            self.calc_correlation()
            self.update()
            self.attempts+=1
            if self.attempts%1000 == 0:
                print(f'{self.attempts} complete. Deviation: {self.dev}\n')
            
        if (self.attempts<1000000):
            print(f'Convergence achieved in {self.attempts} attempts.')
            print(f'Contact Density is {self.DFT.rho[self.NiW]*self.DFT.R*self.DFT.R*self.DFT.R:.12f} (in format rho*R^3)')
            
        else:
            print(f'Density profile failed to converge after {self.attempts} attempts.')
            
            
    def output_simulation_data(self,output_file_name):
            
        """
        Outputs the key parameters of the minimisation as well as the final density profile.
        """
        # Works out the correct precision for the distance output.
        pres = 0; dr = self.DFT.dr;
        while dr<1:
            dr*=10; pres+=1;
            
        with open(output_file_name, 'a') as out:
            out.write(f'Produced {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
            out.write(f'Eta = {self.DFT.eta}\nR = {self.DFT.R}\nT = {self.DFT.Temp}\n')
            out.write(f'N = {self.DFT.N}\ndr = {self.DFT.dr}\n\nFunctional = {self.DFT.functional}\n')
            out.write(f'Pressure = {self.DFT.pressure}\nExcess Chemical Potential = {self.DFT.mu}\n\n')
            out.write(f'Contact Density = {self.DFT.rho[self.NiW]:.14f}\t In terms of R^3 = \
                                           {self.DFT.rho[self.NiW]*self.DFT.R*self.DFT.R*self.DFT.R:.14f}\n')
            out.write(f'Convergence in {self.attempts} attempts.\n\n')
            out.write(f'i\tr\trho\t\trho/rho_bulk\n')
            for i in range(self.DFT.N):
                out.write(f'{i}\t{self.r[i]:.{pres}f}\t{self.DFT.rho[i]:.10f}\t{self.DFT.rho[i]/self.DFT.rho_bulk:.10f}\n')
            
class planar(minimisation):
    
    def __init__(self, cDFT, alpha, file_path, inplace=False):
            
        self.alpha = alpha
        self.file_path = file_path
        minimisation.__init__(self, cDFT, inplace)
            
        self.NiW = 2*self.DFT.NiR
       
        self.DFT.rho[0:self.NiW] = 0.0 
        self.DFT.rho[self.NiW:self.DFT.N-2*self.DFT.NiR] = self.DFT.rho_bulk
        self.DFT.rho[self.DFT.N-2*self.DFT.NiR:] = 0.0
        
        self.r = fft.empty_aligned(self.DFT.N, dtype = 'float64')
       
        for i in range(self.DFT.N):
            self.r[i] = i * self.DFT.dr
            
        #plan to add try...except statements here to catch if user inputs incorrect thing
        if self.DFT.Vext_type == 'Hard Wall':
            self.DFT.Vext[0:self.NiW] = 500
            
    def calc_weighted_densities(self):
            
        """
        Calculates the weighted densities in fourier space and inverses to real space. 
        """
             
        # We then calculate the fourier transform of r*rho
        self.rho[:] = self.DFT.rho[:]
        self.rho[self.NiW]*=0.5   #Using heaviside function here
        self.fft_rho()
             
        # Then we calculate the fourier transform of the weighted densities
        #  and then invert and divide by r
        self.fn2[:] = self.frho[:]*self.DFT.fw2[:]
        self.ifft_n2() 
        self.DFT.n2[:self.NiW-self.DFT.NiR+1] = 0.0
             
        self.fn3[:] = self.frho[:]*self.DFT.fw3[:]
        self.ifft_n3() 
        self.DFT.n3[:self.NiW-self.DFT.NiR+1] = 0.0
             
        self.fn2v[:] = (self.frho[:]*self.DFT.fw2v[:])
        self.ifft_n2v() 
        self.DFT.n2v[:self.NiW-self.DFT.NiR+1] = 0.0
             
        # From these weighted densities, we can then employ simple relations
        # to calculate the rest
        self.DFT.n1[:] = self.DFT.n2[:]/(pi4*self.DFT.R)
        self.DFT.n0[:] = self.DFT.n2[:]/(pi4*self.DFT.R**2)
        self.DFT.n1v[:] = self.DFT.n2v[:]/(pi4*self.DFT.R)
        
    def calc_correlation(self):
            
        """
        Calculates the correlation function to be used in determining the new
        density profile.
        """
            
        # First we calculate the derivatives needed for the correlation func
        self.DFT.calculate_derivatives(self.DFT)
        self.DFT.d2[0] = 0.0; self.DFT.d3[0] = 0.0; self.DFT.d2v[0] = 0.0;
            
        # Then we take the fourier transform and perform the convolutions in
        # fourier space
        self.fft_d2(); self.fft_d3(); self.fft_d2v();
            
        # The scalar parts are straight forward 
        self.fc2[:] = self.fd2[:]*self.DFT.fw2[:]
        self.fc3[:] = self.fd3[:]*self.DFT.fw3[:]
            
        self.ifft_c2(); self.ifft_c3();
            
        # The vector correlation function is made of 2 convolutions and
        # therefore contains 2 fft
        self.fc2v[:] = self.fd2v[:]*(-1.0*self.DFT.fw2v[:])
            
        self.ifft_c2v(); 
            
        self.DFT.c2[0] = 0.0; self.DFT.c3[0] = 0.0; self.DFT.c2v[0] = 0.0;
          
    def minimise(self):
        
        output_file_name = self.file_path + 'planar_' + str(self.DFT.eta) + '_' + str(self.DFT.dr) + '_' + str(self.DFT.functional)
        super().minimise()
        with open(output_file_name, 'w') as out:
            out.write(f'Planar Geometry.\n')
        self.output_simulation_data(output_file_name)

class spherical(minimisation):
    
    def __init__(self, cDFT, Rs, alpha, file_path, inplace=False):
        """
        In order to prevent redefinition of transforms throughout the program, we
        store all the important ingredients for minimisation here.
        """
        
        self.alpha = alpha
        self.file_path = file_path
        minimisation.__init__(self, cDFT, inplace)
        
        # We must set the initial density profile and Vext. For testing purposes,
        # I am using the standard style grid. Later, I hope to reduce this to
        # only 2 convolutions worth in order to be able to run at larger radii.
        self.Rs = Rs*cDFT.R + self.DFT.R
        self.r = fft.empty_aligned(self.DFT.N, dtype = 'float64')
        self.NiW = int(round((self.Rs)/self.DFT.dr))
       
        for i in range(self.DFT.N):
            self.r[i] = i * self.DFT.dr
            
        self.DFT.rho[0:self.NiW] = 0.0 
        self.DFT.rho[self.NiW:self.DFT.N-2*self.DFT.NiR] = self.DFT.rho_bulk
        self.DFT.rho[self.DFT.N-2*self.DFT.NiR:] = 0.0
        
        #plan to add try...except statements here to catch if user inputs incorrect thing
        if self.DFT.Vext_type == 'Hard Wall':
            self.DFT.Vext[0:self.NiW] = 500
           
    def calc_weighted_densities(self):
            
        """
        Calculates the weighted densities in fourier space and inverses to real space. 
        """
             
        # We then calculate the fourier transform of r*rho
        self.rho[:] = self.DFT.rho[:]*self.r[:]
        self.rho[self.NiW]*=0.5   #Using heaviside function here
        self.fft_rho()
             
        # Then we calculate the fourier transform of the weighted densities
        #  and then invert and divide by r
        self.fn2[:] = self.frho[:]*self.DFT.fw2[:]
        self.ifft_n2() 
        self.DFT.n2[1:self.DFT.N-2*self.DFT.NiR]/=self.r[1:self.DFT.N-2*self.DFT.NiR]
        self.DFT.n2[:self.NiW-self.DFT.NiR+1] = 0.0
             
        self.fn3[:] = self.frho[:]*self.DFT.fw3[:]
        self.ifft_n3() 
        self.DFT.n3[1:self.DFT.N-2*self.DFT.NiR]/=self.r[1:self.DFT.N-2*self.DFT.NiR]
        self.DFT.n3[:self.NiW-self.DFT.NiR+1] = 0.0
             
        self.fn2v[:] = (self.frho[:]*self.DFT.fw2v[:])
        self.ifft_n2v() 
        self.DFT.n2v[1:self.DFT.N-2*self.DFT.NiR]/=self.r[1:self.DFT.N-2*self.DFT.NiR]
        self.DFT.n2v[1:self.DFT.N -2*self.DFT.NiR] += self.DFT.n3[1:self.DFT.N -2*self.DFT.NiR]/self.r[1:self.DFT.N -2*self.DFT.NiR]
        self.DFT.n2v[:self.NiW-self.DFT.NiR+1] = 0.0
             
        # From these weighted densities, we can then employ simple relations
        # to calculate the rest
        self.DFT.n1[:] = self.DFT.n2[:]/(pi4*self.DFT.R)
        self.DFT.n0[:] = self.DFT.n2[:]/(pi4*self.DFT.R**2)
        self.DFT.n1v[:] = self.DFT.n2v[:]/(pi4*self.DFT.R)
                 
    def calc_correlation(self):
            
        """
        Calculates the correlation function to be used in determining the new
        density profile.
        """
            
        # First we calculate the derivatives needed for the correlation func
        self.DFT.calculate_derivatives(self.DFT)
        self.DFT.d2[0] = 0.0; self.DFT.d3[0] = 0.0; self.DFT.d2v[0] = 0.0;
        self.DFT.d2[:]*=self.r[:]; self.DFT.d3[:]*=self.r[:];
            
        # Then we take the fourier transform and perform the convolutions in
        # fourier space
        self.fft_d2(); self.fft_d3(); self.fft_d2v();
            
        # The scalar parts are straight forward 
        self.fc2[:] = self.fd2[:]*self.DFT.fw2[:]
        self.fc3[:] = self.fd3[:]*self.DFT.fw3[:]
            
        self.ifft_c2(); self.ifft_c3();
        #cDFT.c2[:cDFT.NiW-2*cDFT.NiR] = 0.0; cDFT.c3[:cDFT.NiW-2*cDFT.NiR] = 0.0;
        self.DFT.c2[1:]/=self.r[1:]; self.DFT.c3[1:]/=self.r[1:];
            
        # The vector correlation function is made of 2 convolutions and
        # therefore contains 2 fft
        self.fc2v[:] = self.fd2v[:]*self.DFT.fw3[:]
        self.DFT.d2v[:]*=self.r[:]
        self.fft_d2v()
        self.fc2v_dummy[:] = self.fd2v[:]*self.DFT.fw2v[:]
            
        self.ifft_c2v(); self.ifft_c2v_dummy();
            
        #cDFT.c2v[:cDFT.NiW-2*cDFT.NiR] = 0.0
        self.DFT.c2v[1:] -= self.DFT.c2v_dummy[1:]
        self.DFT.c2v[1:]/=self.r[1:]
            
        self.DFT.c2[0] = 0.0; self.DFT.c3[0] = 0.0; self.DFT.c2v[0] = 0.0;
  
    def minimise(self):
        
        output_file_name = self.file_path + 'spherical_' + str(self.DFT.eta) + '_' + str(self.DFT.dr) + '_' + str(self.Rs) 
        super().minimise()
        with open(output_file_name, 'w') as out:
            out.write(f'Spherical Geometry.\nRs = {self.Rs:.6f}\n')
        self.output_simulation_data(output_file_name)
        

if __name__ == "__main__":
    pass