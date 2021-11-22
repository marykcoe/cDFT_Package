#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Density Functional Theory Program.
Supports one-component hard-sphere and truncated Lennard-Jones
fluids incontact with homogenous planar surfaces, spherical
solutes and confined to a slit with homogeneous surfaces.

Created October 2020. Last Update November 2021.

This program utilises FMT and supports the Rosenfeld,
White-Bear and White-Bear Mark II functionals.

For information on how to use this package please consult
the accompanying tutorials. Information on how the package
works can be found in Chapter 4 and Appendix A-C of the
following thesis (link available December 2021)

This module contains the exception objects.
"""

class Error(Exception):
    """Base class for exceptions"""
    pass

class UnsupportedFunctionalError(Error):

    """
	Exception raised for unsupported input functional.

    Attributes:
        functional(string): input functional which caused the error
        message(string): explanation of supported functionals
    """

    def __init__(self, functional, message="Functional not supported."):
        self.functional = functional
        message = "This functional is not supported.\n"
        message += "Please choose a supported functional.\n"
        message += "The supported functionals are RF (Rosenfeld), WB (White-Bear), WBII (White-Bear Mark II)."
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f' functional = {self.functional}.\n{self.message}'



class UnsupportedFluidError(Error):

    """
	Exception raised for unsupported input fluid_type.

    Attributes:
        fluid_type(string):  input fluid_type which caused the error
        message(string): explanation of supported fluid types
    """

    def __init__(self, fluid_type, message=""):
        self.fluid_type = fluid_type
        message = "This fluid type is not supported.\n"
        message += "Please choose a supported fluid type.\n"
        message += "The supported fluid types are HS (hard-sphere) or TLJ (truncated Lennard-Jones)."
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f' fluid_type = {self.fluid_type}.\n{self.message}'


class UnsupportedWallTypeError(Error):

    """
	Exception raised for unsupported input wall_type.

    Attributes:
        wall_type(string): input wall_type which caused the error
        message(string): explanation of supported wall types
    """

    def __init__(self, wall_type, message=""):
        self.wall_type = wall_type
        message = "This wal type is not supported.\n"
        message += "Please choose a supported wall type.\n"
        if self.Rs is not None:
            message += "The supported wall types are HW (hard wall), "
            message += "LJ (Lennard-Jones) and SLJ (Lennard-Jones with "
            message += "minimum shifted to occur at the surface of the wall)."
        else:
            message += "The supported wall types are HW (hard wall), "
            message += "LJ (Lennard-Jones), SLJ (Lennard-Jones with "
            message += "minimum shifted to occur at the surface of the wall) "
            message += "and WCALJ (Lennard-Jones with WCA splitting)."

        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f' wall_type = {self.wall_type}.\n{self.message}'