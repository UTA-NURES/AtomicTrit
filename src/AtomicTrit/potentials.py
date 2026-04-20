# ================================================================================
# the various potentials that can be used in the subsequent calculations.
# ================================================================================

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from AtomicTrit.constants import *

path=os.path.dirname(os.path.abspath(__file__))


#=====================================
# Composites
#=====================================

bohr = BohrInAng/hcInEVAngstrom

#Read a potential from the file
def ReadPotential(path):
    dat=np.genfromtxt(path, delimiter=',', skip_header=1)
    interp = interp1d(dat[:, 0]*bohr, (dat[:, 1]+1)*HartreeInEV, kind='cubic', bounds_error=False,fill_value='extrapolate')
    return interp

# Stitch together N potentials, separated at points demarcated as "splits"
def CompositePotential(R,Potentials,Splits):
    return (sum([((Splits[i]<=R) * (R<Splits[i+1]))*Potentials[i](R) for i in range(0,len(Splits)-1)]))

# Extend any potential to large distance with an R^-6 term
def VanDerWaalsExtension(R,Potential, split):
    return (R<=split)* Potential(R) + (R>split)*Potential(split)*(split/R)**6

def HFD(Rho):
    R = Rho * hcInEVAngstrom / BohrInAng
    return (-6.5/R**6-124/R**8-3285/R**10)

# Extend any potential to large distance with an R^-6 term
def HFDExtension(R,Potential, split):
    return (R<=split)* Potential(R) + (R>split)*Potential(split)/HFD(split)*HFD(R)

# Read these files
SingletFiles =   ['Singlet_Kolos1965.csv',
                 'Singlet_Kolos1974.csv',
                 'Singlet_Kolos1975.csv',
                 'Singlet_Wolniewicz1993.csv',
                 'Singlet_Jamieson2000.csv']

TripletFiles =   ['Triplet_Kolos1965.csv',
                 'Triplet_Kolos1974.csv',
                 'Triplet_Jamieson2000.csv']


# Fill these dictionaries with interpolators
SingletInterps={}
TripletInterps={}
for i in range(0,len(SingletFiles)):
    SingletInterps[SingletFiles[i]]=ReadPotential(path+"/InputData/"+SingletFiles[i])

for i in range(0,len(TripletFiles)):
    TripletInterps[TripletFiles[i]]=ReadPotential(path+"/InputData/"+TripletFiles[i])


#===========================
# Kolos Potentials
#===========================

#The Journal of Chemical Physics. 1965 Oct 1;43(7):2429-41.
def Kolos_Singlet1_VDW(R):
    return VanDerWaalsExtension(R,SingletInterps['Singlet_Kolos1965.csv'],9.5*bohr)

def Kolos_Triplet1_HFD(R):
    return HFDExtension(R,TripletInterps['Triplet_Kolos1965.csv'],9.5*bohr)

#Kolos 1974, Chemical Physics Letters.  Volume 24, Issue 4, 15 February 1974, Pages 457-460
def Kolos_Singlet2(R):
    return CompositePotential(R, [SingletInterps['Singlet_Kolos1965.csv'], SingletInterps['Singlet_Kolos1974.csv']], [0, 6.0 * bohr, 1e6])

def Kolos_Singlet2_VDW(R):
    return VanDerWaalsExtension(R,Kolos_Singlet2,11.5*bohr)

def Kolos_Triplet2(R):
    return CompositePotential(R, [TripletInterps['Triplet_Kolos1965.csv'], TripletInterps['Triplet_Kolos1974.csv']], [0, 6.0 * bohr, 1e6])

def Kolos_Triplet2_HFD(R):
    return HFDExtension(R,Kolos_Triplet2,11.5*bohr)




#==========================
# Wolniewicz potentials
#==========================

def Wolniewicz_Singlet(R):
    return (SingletInterps['Singlet_Wolniewicz1993.csv'](R))

def Wolniewicz_Singlet_VDW(R):
    return (VanDerWaalsExtension(R,Wolniewicz_Singlet,12*bohr))

#==========================
# Jamieson Potentials
#==========================
# from Physical Review A. 2000 Mar 6;61(4):042705.

def Jamieson_Singlet(R):
    return CompositePotential(R, [SingletInterps['Singlet_Kolos1965.csv'], SingletInterps['Singlet_Jamieson2000.csv']], [0, 1.5 * bohr, 1e6])

def Jamieson_Triplet(R):
    return CompositePotential(R, [TripletInterps['Triplet_Kolos1965.csv'], TripletInterps['Triplet_Jamieson2000.csv']], [0, 1.5 * bohr, 1e6])

def Jamieson_Singlet_VDW(R):
    return VanDerWaalsExtension(R,Jamieson_Singlet,20*bohr)

def Jamieson_Triplet_HFD(R):
    return HFDExtension(R,Jamieson_Triplet,20*bohr)

def Jamieson_Triplet_VDW(R):
    return VanDerWaalsExtension(R,Jamieson_Triplet,20*bohr)


# ==========================================
#For illustration only -
# radial dependence of the dipole potential
#===========================================

# From Stoof et al, Physical Review B 38.7 (1988): 4688.
def DipoleRadialPart(rho):
    muel = np.sqrt(4 * np.pi * finestructure) / (2 * meeV)
    return muel**2/(4*np.pi*rho**3)*(4*np.pi/5)



#=====================================
# Silvera potentials in analytic form
#=====================================

# From European Scientific Journal. 2012 Oct 1;8(24).
#  Based on Silvera modified by Fried and Etters,
def Silvera_Triplet(Rho):
    rmin=4.16
    x = Rho * hcInEVAngstrom / rmin
    D = 1.28
    F=(x>D)+(x<D)*np.exp(-(D/x-1)**2)
    return 6.46 * K2eV * (4.889e4*np.exp(0.0968-8.6403*x-2.427*x**2)-(1.365/x**6+0.425/x**8+0.183/x**10)*F)

# From Reviews of Modern Physics, 52(2), p.393. and Progress in Low Temperature Physics. 1986 Jan 1;10:139-370.
def Silvera_Triplet2(Rho):
    R = Rho * hcInEVAngstrom / BohrInAng
    P = np.exp(0.09678-1.10173*R-0.03945*R**2)+np.exp(-(10.0378/R-1)**2)*(-6.5/R**6-124/R**8-3285/R**10)
    return P * HartreeInEV


# From Reviews of Modern Physics, 52(2), p.393.
def Silvera_J(Rho):
    R = Rho * hcInEVAngstrom / BohrInAng
    P = np.exp(-.288-.275*R-.176*R**2+.0068*R**3)
    return P * HartreeInEV

# From Reviews of Modern Physics, 52(2), p.393
# Note - Silvera says this is not an accurate representation, use with extreme caution
def Silvera_Singlet(R):
    return Silvera_Triplet(R) - Silvera_J(R)

#==========================
# The adiabatic correction
#==========================

# Adiabatic correction from  Journal of Molecular Spectroscopy. 1990 Oct 1;143(2):237-50.
def ReadAdiabaticCorrection(file):
    dat_Adiabatic = pd.read_csv(file)
    Hvals = np.array(dat_Adiabatic.Hprime)
    DVals = np.array(dat_Adiabatic.D)
    Corr = ((Hvals[0:-1] - Hvals[-1]) / (DVals[0:-1] - DVals[-1]))
    CorrInterp = interp1d(dat_Adiabatic.R[0:-1] * BohrInAng / hcInEVAngstrom, Corr, bounds_error=False, kind='linear',
                          fill_value=(Corr[0], Corr[-1]))
    return CorrInterp

TripletCorrection=ReadAdiabaticCorrection(path + "/InputData/AdiabaticTripletCorrection_Kolos1990.csv")
SingletCorrection=ReadAdiabaticCorrection(path + "/InputData/AdiabaticSingletCorrection_Wolniewicz1993.csv")

def ApplyCorrection(R, Potential, Correction, scale):
    return Potential(R)*(1+scale*Correction(R))


# ==========================================
# With adiabatic corrections
#===========================================


# Kolos has no adiabatic correction, so we apply with scale +1/3 for T and +1 for H
def Kolos_Triplet2_HFD_T(R):
    return ApplyCorrection(R,Kolos_Triplet2_HFD,TripletCorrection,+1/3)
def Kolos_Singlet2_VDW_T(R):
    return ApplyCorrection(R,Kolos_Singlet2_VDW,SingletCorrection,+1/3)
def Kolos_Triplet2_HFD_H(R):
    return ApplyCorrection(R,Kolos_Triplet2_HFD,TripletCorrection,+1)
def Kolos_Singlet2_VDW_H(R):
    return ApplyCorrection(R,Kolos_Singlet2_VDW,SingletCorrection,+1)
def Kolos_Triplet1_HFD_T(R):
    return ApplyCorrection(R,Kolos_Triplet1_HFD,TripletCorrection,+1/3)
def Kolos_Singlet1_VDW_T(R):
    return ApplyCorrection(R,Kolos_Singlet1_VDW,SingletCorrection,+1/3)
def Kolos_Triplet1_HFD_H(R):
    return ApplyCorrection(R,Kolos_Triplet1_HFD,TripletCorrection,+1)
def Kolos_Singlet1_VDW_H(R):
    return ApplyCorrection(R,Kolos_Singlet1_VDW,SingletCorrection,+1)

# Jamieson and Wolniewicz both have the H correction, so we scale -2/3 for T and not for H
def Jamieson_Triplet_HFD_T(R):
    return ApplyCorrection(R,Jamieson_Triplet_HFD,TripletCorrection,-2/3)

def Jamieson_Singlet_VDW_T(R):
    return ApplyCorrection(R,Jamieson_Singlet_VDW,SingletCorrection,-2/3)
def Wolniewicz_Singlet_VDW_T(R):
    return ApplyCorrection(R,Wolniewicz_Singlet_VDW,SingletCorrection,-2/3)
def Jamieson_Triplet_HFD_H(R):
    return ApplyCorrection(R,Jamieson_Triplet_HFD,TripletCorrection,0)
def Jamieson_Singlet_VDW_H(R):
    return ApplyCorrection(R,Jamieson_Singlet_VDW,SingletCorrection,0)
def Wolniewicz_Singlet_VDW_H(R):
    return ApplyCorrection(R,Wolniewicz_Singlet_VDW,SingletCorrection,0)



# As far as we can tell, Silvera does not have the correction.
def Silvera_Triplet_T(R):
    return ApplyCorrection(R, Silvera_Triplet, TripletCorrection, +1 / 3)
def Silvera_Triplet2_T(R):
    return ApplyCorrection(R, Silvera_Triplet2, TripletCorrection, +1 / 3)
def Silvera_Triplet_H(R):
    return ApplyCorrection(R, Silvera_Triplet, TripletCorrection, +1)
def Silvera_Triplet2_H(R):
    return ApplyCorrection(R, Silvera_Triplet2, TripletCorrection, +1)



# ==========================================
# Compilation of useful potentials
#===========================================

Triplets = {#"Kolos 65":Kolos_Triplet1_HFD,
            "Kolos 74":Kolos_Triplet2_HFD,
            "Silvera":Silvera_Triplet,
            #"Silvera2":Silvera_Triplet2,
            "Jamieson":Jamieson_Triplet_HFD}

Singlets = {#"Kolos 65":Kolos_Singlet1_HFD,
            "Kolos 74":Kolos_Singlet2_VDW,
            #"Silvera":Silvera_Singlet,
            "Wolniewicz":Wolniewicz_Singlet_VDW,
            "Jamieson":Jamieson_Singlet_VDW}

TripletsT = {#"Kolos 65":Kolos_Triplet1_HFD_T,
            "Kolos 74":Kolos_Triplet2_HFD_T,
            "Silvera":Silvera_Triplet_T,
            "Jamieson":Jamieson_Triplet_HFD_T}

SingletsT = {#"Kolos 65":Kolos_Singlet1_HFD_T,
            "Kolos 74":Kolos_Singlet2_VDW_T,
            "Wolniewicz":Wolniewicz_Singlet_VDW_T,
            "Jamieson":Jamieson_Singlet_VDW_T}

TripletsH = {#"Kolos 65":Kolos_Triplet1_HFD_H,
            "Kolos 74":Kolos_Triplet2_HFD_H,
            "Silvera":Silvera_Triplet_H,
            "Jamieson":Jamieson_Triplet_HFD_H}

SingletsH = {#"Kolos 65":Kolos_Singlet1_HFD_H,
            "Kolos 74":Kolos_Singlet2_VDW_H,
            "Wolniewicz":Wolniewicz_Singlet_VDW_H,
            "Jamieson":Jamieson_Singlet_VDW_H}


#============================================
# Jochemsen Potentials (R2) in analytic form
#============================================
# Canadian Journal of Physics, 62(8), 751–759

def R2(Rho):
    # Atomic units are used in the potential
    R = np.asarray(Rho) / BohrInEV
    V = np.zeros_like(Rho)

    A       = 2.5725              # Hartree
    R0      = 6.0471              # Bohr
    Alpha   = 1.7902              # Bohr ^-1
    A0      = A*np.exp(-R0*Alpha) # Hartree
    Epsilon = 2.20496e-5          # Hartree
    R1      = 8.21763             # Bohr
    C6      = 2.8187              # Hartree / Bohr^3
    C8      = 41.7257             # Hartree / Bohr^8
    C10     = 865.7844            # Hartree / Bohr^10

    # Masks
    mask1 = R <= R0
    mask2 = (R > R0) & (R <= R1)
    mask3 = R > R1

    # Region 1: exponential repulsion
    V[mask1] = A*np.exp(-Alpha*R[mask1]) - A0

    # Region 2: Lennard-Jones-like mid region
    V[mask2] = 4*Epsilon*((R0/R[mask2])**12 - (R0/R[mask2])**6)

    # Region 3: long-range attractive tail
    V[mask3] = -(C6/R[mask3]**6 + C8/R[mask3]**8 + C10/R[mask3]**10)

    return V * HartreeInEV

#============================================
# Scoles Potentials (HFD-B) in analytic form
#============================================
# Canadian Journal of Physics, 62(8), 751–759

def HFD_B(_r):
    _r = _r/ BohrInEV
    def V_SCF(_r):
        alpha = 2.1829 # Bohr ^ -1
        beta  = 1.9
        A     = 1.2712 # Hartree / Bohr ^ beta
        return A*_r**beta * np.exp(-alpha * _r)

    def g(_r, _n):
        rho = 1.1696 # Bohr ^ -1
        return np.power(1 - np.exp(-2.1 * rho *_r/_n - 0.109 * rho**2 *_r**2/np.sqrt(_n)), _n)

    def f(_r):
        rho = 1.1696 # Bohr ^ -1
        return 1 - np.power(rho * _r, 1.68  ) * np.exp(-0.78 * rho * _r)

    def V_CORR(_r):
        C6   = 0.061922613 * 45.536241158
        C8   = 0.25702478  * 162.608373042
        C10  = 1.4894502   * 580.668986083
        C12  = 12.388198   * 2073.549258817
        C14  = 144.0574    * 7404.574089179
        summ = C6/(_r**6)* g(_r, 6) + C8/(_r**8)* g(_r, 8) + C10/(_r**10)* g(_r, 10) + C12/(_r**12)* g(_r, 12) +C14/(_r**14)* g(_r, 14)
        return -f(_r) * summ

    return (V_SCF(_r) + V_CORR(_r)) * HartreeInEV
