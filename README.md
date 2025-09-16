# Atomic T and H Cross Section Calculator

This package can be used to evaluate the cross sections for elastic and inelastic scattering of cold atomic 
hydrogen and tritium.  

It accompanies the paper: 

## Reference
If you use this code, please cite the paper:

"Elastic and Spin-Changing Cross Sections of Spin-Polarized Atomic Tritium"
M.G. Elliott and B.J.P. Jones

## Structure

The source code is in the /src/ folder, and the /examples/ folder contains a set of Jupyter notebooks 
that illustrate the use of code for various calculations.  

## Examples

The examples provided with the codeare the following:

**Elastic Cross Section**
Calculate the elastic scattering dd->dd cross section as a function of magnetic field and temperature

**Dipolar Losses**
Calculate the dipole (eg. dd->aa) loss rates in the zero temperature limit

**Spin Exchange**
Calculate the spin exchange (e.g. cc->aa) loss rates in the zero temperature limit.

**Potentials**
Compare the various potential shapes used in the calculations (see potentials.py to enable a different set)

**Potential Comparison**
Check the effects of the potential on some of the the cross sections

**Hyperfine levels**
Plot the Breit Rabi levels of H and T

**Higher partial waves**
Perform finite temperature calculations including the effects of higher partial waves than l=0

**Literature comparison**
Make a set of plots to compare predictions to published data

**Adiabatic correction**
Check the effect of the adiabatic correction to the potential

**Make Tables**
Generate cross section tables as a function of channel, B field and temperature

**Plot Tables**
Pull data from the above tables and plot it

**Fractional Uncertainty**
Evaluate error bars due to various sources of systematic uncertainty

