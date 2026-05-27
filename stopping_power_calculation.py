# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:24:51 2024

@author: diana.bachiller
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

%matplotlib inline

energy = 3000 # (keV)
energy_step = 0.5 # (keV)
charge_fluence = 1 # (µC/cm2) OPTIONAL to calculate dose

mode = "LivePIXE" # Set elements and concentration in mode "manual" or "LivePIXE"
calculate_range = "yes" # If "yes", set target_density 
target_density = 2.1 # (g/cm³) OPTIONAL to calculate ion range

# Manual, up to 6 elements
# EXAMPLE: PbCO3 (Cerussite)
Z1 = 82
n1 = 1
Z2 = 6
n2 = 1
Z3 = 8
n3 = 3
Z4 = 1
n4 = 0
Z5 = 1
n5 = 0
Z6 = 2
n6 = 0

all_elements_Z = np.loadtxt("coefficients_stopping_power.txt", usecols=0, dtype=int)
all_elements_atomic_weight = np.loadtxt("atomic_weight.txt", usecols=1)
all_elements_coefficients_stopping_power = np.loadtxt("coefficients_stopping_power.txt", usecols=(1,2,3,4,5,6,7,8))


if mode == "LivePIXE":
# LivePIXE's file with elemental composition
    matrix_elements_Z = np.loadtxt("elemental_composition_LivePIXE.txt", skiprows=1, usecols=0, dtype=int)
    matrix_elements_concentration = np.loadtxt("elemental_composition_LivePIXE.txt", skiprows=1, usecols=1)
if mode == "manual":
    matrix_elements_Z = np.array([Z1,Z2,Z3,Z4,Z5,Z6])
    matrix_elements_n = np.array([n1,n2,n3,n4,n5,n6])
    matrix_elements_atomic_weight = all_elements_atomic_weight[matrix_elements_Z]
    matrix_elements_mass = matrix_elements_atomic_weight*matrix_elements_n
    total_mass = np.sum(matrix_elements_mass)
    matrix_elements_concentration = matrix_elements_mass/np.sum(matrix_elements_mass)

sp_all = []
all_energies = []
initial_energy = np.copy(energy)

while energy > 25:
    S_all = []
    for n in range(np.shape(matrix_elements_Z)[0]):
        Z = matrix_elements_Z[n]
        concentration = matrix_elements_concentration[n]
        #print(Z,concentration,np.where(all_elements_Z==Z)[0][0])
        atomic_weight = all_elements_atomic_weight[np.where(all_elements_Z==Z)[0][0]]
        #print(atomic_weight)
        coefficients_sp = all_elements_coefficients_stopping_power[np.where(all_elements_Z==Z)[0][0]]
        SL = coefficients_sp[0]*energy**coefficients_sp[1]+coefficients_sp[2]*energy**coefficients_sp[3]
        SH = coefficients_sp[4]/energy**coefficients_sp[5]*np.log(coefficients_sp[6]/energy+coefficients_sp[7]*energy)
        S = SL*SH/(SL+SH)*6.0222e5/atomic_weight
        S_all.append(S)
    sp = np.sum(np.array(S_all)*matrix_elements_concentration)
    sp_all.append(sp)
    all_energies.append(energy)
    energy = energy - energy_step

sp_all = np.array(sp_all) # Stopping power in keV·cm²/g
print("Stopping Power entrance = ",np.round(sp_all[0]/1000,2), " MeV·cm²/g")
print("Stopping Power max = ", np.round(np.max(sp_all)/1000,2), " MeV·cm²/g")
print("Dose entrance = ", np.round(sp_all[0]*charge_fluence/1e3,2), " kGy")
print("Dose max = ", np.round(np.max(sp_all)*charge_fluence/1e3,2), " kGy")

plt.figure(figsize=(7,5))
plt.scatter(all_energies, sp_all/1000, color="b", )
plt.xlim((initial_energy, 0))
plt.xlabel("Energy (keV)", fontsize = 14)
plt.ylabel("Stopping Power (MeV·cm²/g)", fontsize = 14)
plt.tick_params(axis='both', which='both', labelsize=14)
plt.title("Stopping Power vs Proton Energy", fontsize = 14)
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(all_energies, sp_all*charge_fluence/1e3, color="b", )
plt.xlim((initial_energy, 0))
plt.xlabel("Energy (keV)", fontsize = 14)
plt.ylabel("Dose (kGy)", fontsize = 14)
plt.tick_params(axis='both', which='both', labelsize=14)
plt.title("Dose vs Proton Energy", fontsize = 14)
plt.show()

dx=0
depth = []

if calculate_range == "yes":
    sp_keV_per_cm = sp_all*target_density  # keV/cm
    dx_all = energy_step/sp_keV_per_cm # cm
    for n in range(np.shape(sp_keV_per_cm)[0]):
        dx = dx + energy_step/sp_keV_per_cm[n] # cm 
        depth.append(dx*1e4) # µm
    dx = dx*1e4 # µm
    print("Ion range = ",np.round(dx,2), " µm")
    
    plt.figure(figsize=(7,5))
    plt.scatter(depth, sp_all/1000, color="b", )
    plt.xlabel("Depth (µm)", fontsize = 14)
    plt.ylabel("Stopping Power (MeV·cm²/g)", fontsize = 14)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.title("Stopping Power vs Depth", fontsize = 14)
    plt.show()
    
    plt.figure(figsize=(7,5))
    plt.scatter(depth, sp_all*charge_fluence/1e3, color="b", )
    # plt.xlim((0, 100))
    plt.xlabel("Depth (µm)", fontsize = 14)
    plt.ylabel("Dose (kGy)", fontsize = 14)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.title("Dose vs Depth", fontsize = 14)
    plt.show()
