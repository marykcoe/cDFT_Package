#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Package.
Copyright Mary Coe m.k.coe@bristol.ac.uk

Created March 2019. Last Update March 2019.

Classical Density Functional Theory (cDFT) is a method to find the equilibrium
density profile of fluid subjected to an external potential. A good review of
the method can be found in Roth R. 2010. J. Phys.:Condens. Matter 22 063102.

This program provides a quick tutorial on some of the basic functions of the 
package. It acts as the documentation until the full documentation is uploaded
in the next few weeks.

This package is not complete. This is a working version, and therefore does
not include error messages. Therefore, caution is advised. The most common 
error, is making the grid too small. The program requires the grid to be
greater than the number of points in the wall plus four times the number of
points in the radii of the particles. Also, some sum rules only work if the 
radii is 1.0.

The package currently supports:
    1. Planar and Spherical Hard Walls
    2. Rosenfeld and White-Bear functionals
    3. Planar adsorption and contact density sum rules
    4. Spherical adsorption, surface tension, and grand potential sum rules.
These are all described below. 

Over the coming months, this package will be improved to also support:
    1. Cylindrical Hard Walls
    2. White-Bear Mark II functional
    3. Lennard-Jones Potentials
    4. Multiprocessing

The required python modules to run this package are:
    1. pyfftw
    2. numpy
    3. matplotlib

In the mean time, enjoy playing around with cDFT!
"""

# To use the package, we import the various files
import cDFT.minimisation as minimisation
import cDFT.output as output
import cDFT.standard_routines as standard_routines
import numpy as np
import matplotlib.pyplot as plt


# The DFT object is specified as packing fraction, interaction, radius of spheres, 
# temperature, number of grid points (recommended to be a power of 2), grid discretisation \
# and functional.
RF = minimisation.DFT(0.2,'Hard Wall',1.0,1.0,2**14,0.005,'Rosenfeld')

# To initiate a planar minimisation, you must give a DFT, a mixing parameter (recommended
# to be between 0.01 and 0.1) and an output filepath. To minimise, then just call the
# minimise routine.
planar = minimisation.planar(RF, 0.1, './RF_planar_example/')
planar.minimise()

# To see the density profile, send the minimisation object to output. This is
# saved to a pdf in the output directory './RF_planar_example/'.
output.plot_single_density(planar)

# For the planar geometry, the surface tension, grand potential and adsorption can be found.
# We can also explore the accuracy of the minimisation using some sum rules relating to these.
# To find the surface tension, or the excess grand potential, use the standard_routines package, 
# specifying first the geometry as a string, and then sending the minimisation object.
surface_tension = standard_routines.surface_tension('planar', planar)
print(f'\nSurface tension is {surface_tension:.6f}')

# The excess grand potential returns are array, so must be summed to find the full excess
# grand potential.
grand_potential = np.sum(standard_routines.excess_grand_potential('planar', planar))
print(f'\nExcess Grand Potential is {grand_potential:.6f}')

# For the planar geometry, the supported sum rules are the contact sum rule and the
# adsorption sum rule. The contact sum rule is beta*p = rho[contact], so can be found
# by comparing the pressure and contact density. The minimisation object, when initiated,
# creates a copy of the DFT, hence to access properties of the minimised object we use
# planar.DFT. For the adsorption sum rule we again use the standard_routines library, 
# specifying the DFT object, the mixing parameter, the output file path and the geometry.
print(f'\n---------------------------------------')
print(f'Contact Sum Rule:\nPressure = {planar.DFT.pressure:.6f} rho[Contact] = {planar.DFT.rho[planar.NiW]}')
print(f'Relative error is {abs(planar.DFT.pressure-planar.DFT.rho[planar.NiW])/planar.DFT.pressure:.6f}')
print(f'---------------------------------------\n')

standard_routines.adsorption_sum_rule(RF,0.1,'./RF_planar_example/','planar')

# The package also supports the White-Bear functional
WB = minimisation.DFT(0.2,'Hard Wall',1.0,1.0,2**14,0.005,'Whitebear')

# As well as a spherical wall. Here, the arguments are DFT, bulk wall radius 
# (note that the density profile is measured from the centre of the hard spheres
# hence the first non-zero density point will be at R_s = R_bulk + R), the mixing
# paramater and the output file path. We minimise in the same way and output in the
# same way.
spherical = minimisation.spherical(WB, 2.0, 0.1,'./WB_spherical_example/')
spherical.minimise()
output.plot_single_density(spherical)

# Like before, we can find the surface tension and excess grand potential. We 
# also have access to a wider range of sum rules. For example, the adsorption
# sum rule is supported. Note there is now an extra argument, R_bulk.
standard_routines.adsorption_sum_rule(WB, 0.1,'./WB_spherical_example/','spherical', 2.0)

# As well as a surface tension (gamma) sum rule beta*(pressure + 2*gamma/Rs +dgamma/dRs = rho[contact])
# and an excess grand potential (omega) sum rule (beta*domega/dRs = 4*pi*Rs^2*rho[contact]). 
# Note, these are currently only supported for R = 1.0. We can access them individually
# using
standard_routines.spherical_surface_tension_sum_rule(WB, 2.0, 0.1, './WB_spherical_example/')
standard_routines. spherical_omega_sum_rule(WB, 2.0, 0.1, './WB_spherical_example/')

# or both using
standard_routines.spherical_surface_tension_and_omega_rule(WB, 2.0, 0.1, './WB_spherical_example/')

# There is also a routine to see how the 'surface tension', 'excess grand potential'or
# 'contact' value varies with wall radius. This returns an array of results which can be
# plotted. Note: This will take a few minutes to run and will output a lot of data to 
# the terminal which can be ignored.
results_1 = standard_routines.plot_by_wall_radius(WB, 0.1, './WB_spherical_example/', 'surface tension')
results_2 = standard_routines.plot_by_wall_radius(WB, 0.1, './WB_spherical_example/', 'contact')

plt.figure(1)
ax1 = plt.subplot(211)
ax1.plot(results_1[:,0], results_1[:,1], color='plum', marker='o', markerfacecolor='purple', \
         markeredgecolor='purple', markersize=3)
ax1.set_title('Surface Tension')
ax1.set_ylabel(r'$\beta\gamma(R_{bulk}) (R^2)$')
ax1.set_xlabel(r'$R/R_{bulk}$')
ax1.set_xlim(min(results_1[:,0]), max(results_1[:,0]))

ax2 = plt.subplot(212)
ax2.plot(results_2[:,0], results_2[:,1], color='turquoise', marker='*', markerfacecolor='teal', \
         markeredgecolor='teal', markersize=3)
ax2.set_title('Density at Contact')
ax2.set_ylabel(r'$\rho(R_s)(R^3)$')
ax2.set_xlabel(r'$R/R_s$')
ax2.set_xlim(min(results_2[:,0]), max(results_2[:,0]))
plt.tight_layout()
plt.savefig('./WB_spherical_example/spherical_radii_examples.pdf')
plt.close(1)

# Finally, we can compare density profiles.
RF_2 = minimisation.DFT(0.3,'Hard Wall',1.0,1.0,2**14,0.005,'Rosenfeld')
planar_2 = minimisation.planar(RF_2,0.05,'./RF_planar_example/')
planar_2.minimise()

output.plot_multiple_density([planar,planar_2], './RF_planar_example/planar_comparison.pdf')

# Over the next few months, this package will be getting a lot of new features, so check
# back for the latest version! 