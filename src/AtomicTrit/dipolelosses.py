# ================================================================================
# this contains the important functions for calculating dipole loss rates
# ================================================================================

from AtomicTrit import constants
from AtomicTrit import elastic
from AtomicTrit import potentials
from AtomicTrit import hyperfine
from AtomicTrit import spinbasis
import sympy
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from sympy.physics.wigner import gaunt


# Dipole channel list. Note, the code is not restricted to these channels, but
#  this is the most important set for atomic H and T trapping.

DipoleChannels=[]
DipoleChannels.append({'alpha':'d','beta':'d','alphaprime':'a','betaprime':'a'})
DipoleChannels.append({'alpha':'d','beta':'d','alphaprime':'a','betaprime':'c'})
DipoleChannels.append({'alpha':'d','beta':'d','alphaprime':'a','betaprime':'d'})
DipoleChannels.append({'alpha':'d','beta':'d','alphaprime':'c','betaprime':'c'})
DipoleChannels.append({'alpha':'d','beta':'d','alphaprime':'c','betaprime':'d'})


#================================
# Kinematic helper functions
#================================

# The mean momentum for a molecule at temperature T.
def p_of_temp(mu, T):
    return np.sqrt((4/np.pi)* mu * constants.kb * constants.J2eV * T)

# If a collision occurs at temp T, additional energy is releasted from the hyperfine states.
# This gives the final state momentum.
def pprime(pin, epsa, epsb, epsprimea, epsprimeb, mu):
    E = pin ** 2 / (2 * mu)
    Eprime = E + epsa + epsb - epsprimea - epsprimeb
    pprime = np.sqrt(2 * mu * Eprime)
    return pprime

# This gives the mean momentum in the channel, used in approximations described above
#  Eq 40, right column, Stoof et al.
def p_abs(mu, pin, epsa, epsb, epsprimea, epsprimeb):
    psquared = pin**2 + mu * (epsa + epsb - epsprimea - epsprimeb)
    return np.sqrt(psquared)


#=====================================
# Components of square matrix elements
#=====================================


# This function evaluates the matrix element of the 1/r^3 operator between two plane wave solutions.
#  It provides the main ingredient of the spatial part of the wavefunction for dipolar scattering.
def GetIntegral(rhos,alphain, betain, alphaout, betaout, mu, temp, potential, lin=0, lout=2, how_to_int='Radau'):

    P1 = p_of_temp(mu, temp)
    P2 = pprime(P1, alphain, betain, alphaout, betaout, mu)

    InState =  np.array(elastic.Wave_Function(rhos, P1, lin, mu, potential, how_to_int)[0])
    OutState = np.array(elastic.Wave_Function(rhos, P2, lout, mu, potential, how_to_int)[0])

    Integrand = interp1d(rhos, InState * OutState / rhos ** 3, kind='quadratic')
    Integral = quad(Integrand, rhos[0], rhos[-1])[0] / (P1 * P2)
    return Integral


# This function gets the spatial part of the loss rate for dipolar losses.
# Follows Eq. 40  Stoof et al, Physical Review B 38.7 (1988): 4688.
def GetSpatialPart(channel=DipoleChannels[0], B_value=1e-5, consts=constants.HydrogenConstants, Temperature=5e-4, potential=potentials.Silvera_Triplet,lin=0,lout=2,rhos=np.linspace(1e-9,0.75,2000),how_to_int='Radau'):
    HFLevels=hyperfine.AllHFLevels(B_value, consts)

    aHf =  HFLevels[channel['alpha']]
    bHf =  HFLevels[channel['beta']]
    apHf = HFLevels[channel['alphaprime']]
    bpHf = HFLevels[channel['betaprime']]

    Pin = p_of_temp(consts.mu, Temperature)
    Pout = pprime(Pin, aHf, bHf, apHf, bpHf, consts.mu)

    Integral    = GetIntegral(rhos,aHf, bHf, apHf, bpHf, consts.mu, Temperature, potential, lin,lout,how_to_int)
    SpatialPart = Pout * consts.mu * Integral**2

    return(SpatialPart)

# This function gets the spin part of the loss rate for dipolar losses.
# Follows Eq. 40  Stoof et al, Physical Review B 38.7 (1988): 4688.
def GetSpinPart(channel=DipoleChannels[0], B_value=1e-5, consts=constants.HydrogenConstants):
    NormDiff = 4*np.sqrt(6)
    Rets=spinbasis.GetRotatedElements()
    theta = hyperfine.Theta(2*consts.delW, B_value, consts.gam)
    value = 0
    for m in Rets.keys():
        El = ( spinbasis.GetElement(Rets[m], channel['alpha'], channel['beta'], 1, channel['alphaprime'], channel['betaprime'], 1)) **2
        try:
            value += El.subs(spinbasis.sr2, np.sqrt(2)) \
                    .subs(spinbasis.sr3, np.sqrt(3)) \
                    .subs(spinbasis.c, np.cos(theta)) \
                    .subs(spinbasis.s, np.sin(theta))
        except:
            value += 0
    SpinPart = value*NormDiff**2
    return(SpinPart)

# This function evaluates the Gaunt functions that come from the angular
#  spatial integrals in the dipole rate
def GetGauntTerm(l1,l2,dm):
    SumGaunt=0
    for m1 in range(-l1, l1 + 1):
        Gaunt2 = float(2 * np.sqrt(np.pi) * gaunt(l1, l2, 2, m1, -m1 + dm, -dm))
        SumGaunt += Gaunt2 ** 2
    return SumGaunt

#==================================
# Collision rate functions
#==================================

# This function gets G Factor for dipolar losses.
# Follows Eq. 34  Stoof et al, Physical Review B 38.7 (1988): 4688.
def GetGFactor( channel=DipoleChannels[0],  B_value=1e-5, consts=constants.HydrogenConstants, Temperature=5e4, potential=potentials.Silvera_Triplet,lin=0,lout=2,dm=2,rhos=np.linspace(1e-9,0.75,2000)):
    Pre_Factor = 1 / (5 * np.pi) * constants.mu_dip_couple ** 4 * constants.NatUnits_cm3sm1

    SpatialMatrixElementSq = GetSpatialPart( channel, B_value, consts, Temperature, potential, lin, lout, rhos, 'Radau')
    SpinMatrixElementSq    = GetSpinPart(    channel, B_value, consts)
    GauntElementSq         = GetGauntTerm(lin, lout, dm)

    return(Pre_Factor * GauntElementSq * SpatialMatrixElementSq * SpinMatrixElementSq)

# Here we sum over the first few partial waves, sufficient for 5% level calculation
# of cross section up to approx 100 K.
def GetSummedGFactor( channel=DipoleChannels[0],  B_value=1e-5, consts=constants.HydrogenConstants, Temperature=5e4, potential=potentials.Silvera_Triplet, PWaves= [[0, 2], [2, 0], [2, 2], [2, 4], [4, 2], [4, 4], [4, 6]], dm=2, rhos=np.linspace(1e-9,0.75,2000)):
    G=0
    for pi in range(0,len(PWaves)):
        G+=GetGFactor(channel,  B_value, consts, Temperature, potential,PWaves[pi][0],PWaves[pi][1],dm,rhos)
    return G

# A useful function for comparing to Stoof et al
def B_Naught(B_Values):
    return (1 + B_Values/3.17e-3)
#... and back again
def Invert_B_Naught(B_Values):
    return (B_Values-1)*3.17e-3

# Work out how many m levels will contribute to a given scatter
def CalculateDegeneracy(l1,l2,dm):
    degen=0
    for m in np.arange(-l1,l1+1,1):
        if(np.abs(m-dm)<=l2):
            degen+=1
    return degen


def E_com(T):
    return T*2/np.pi

def GetCrossSection(G,m,E_com):
    A=1e-6 / (constants.BohrInAng*1e-10)**2 * np.sqrt((m*constants.amu/2)/(2*E_com*constants.kb))
    return 2.*A*G

