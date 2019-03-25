#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program for Planar and Spherical Geometry.

Copyright Mary Coe m.k.coe@bristol.ac.uk

Created January 2019. Last Update January 2019.

This module outputs the data to a standard format and plots graphs.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
c = ['purple','teal','grey','green','maroon','navy']
l = ['solid','dashed',':','_.']
m = ['o','x','^','*','d','s']

def plot_single_density(minimise):
    
    """
    Plots the final density profile of the minimisation with respect to the bulk. 
    The main plot is a close up of the region near the wall whilst the inset axes are the full
    density profile.
    The origin is shifted to be at the wall.
    """
  
    ax = plt.subplot(111)
    ax.plot(minimise.r[minimise.NiW:minimise.NiW + 15*minimise.DFT.NiR]-minimise.NiW*minimise.DFT.dr, \
            minimise.DFT.rho[minimise.NiW:minimise.NiW + 15*minimise.DFT.NiR]/minimise.DFT.rho_bulk, \
            color = 'purple')
    ax.set_xlim(0,15*minimise.DFT.R)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho/\rho_b$')
    axins = inset_axes(ax, width="60%", height="50%", loc=1)
    axins.plot(minimise.r[minimise.NiW:minimise.DFT.end] - minimise.NiW*minimise.DFT.dr, \
               minimise.DFT.rho[minimise.NiW:minimise.DFT.end]/minimise.DFT.rho_bulk,color='teal')
    axins.set_xlabel('r')
    axins.set_ylabel(r'$\rho/\rho_b$')
    
    plt.savefig(minimise.file_path + 'density_profile_' + str(minimise.DFT.eta) + '_' + str(minimise.DFT.functional) + '.pdf')
    plt.close()


def plot_multiple_density(minimises, output_filename):
    
  
    ax = plt.subplot(111)
    for i,minimise in enumerate(minimises):
        ax.plot(minimise.r[minimise.NiW:minimise.NiW + 15*minimise.DFT.NiR]-minimise.NiW*minimise.DFT.dr,\
                minimise.DFT.rho[minimise.NiW:minimise.NiW + 15*minimise.DFT.NiR]/minimise.DFT.rho_bulk,\
                color = c[i], linestyle=l[i%4], label = f'{minimise.DFT.functional} $\eta$ {minimise.DFT.eta}')
        
    ax.set_xlim(0,15*minimise.DFT.R)
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\rho/\rho_b$')
    axins = inset_axes(ax, width="60%", height="50%", loc=1)
    for i,minimise.DFT in enumerate(minimises):
        axins.plot(minimise.r[minimise.NiW:minimise.DFT.end] - minimise.NiW*minimise.DFT.dr, \
                   minimise.DFT.rho[minimise.NiW:minimise.DFT.end]/minimise.DFT.rho_bulk,color=c[i],linestyle=l[i%4])
    axins.set_xlabel('r')
    axins.set_ylabel(r'$\rho/\rho_b$')
    
    ax.legend(fontsize=7)
    plt.savefig(output_filename)
    plt.close()


