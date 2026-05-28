# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:24:51 2024
@author: diana.bachiller
"""

"""
Modified May 2026
@author: Laurent Pichon 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import sys
import csv
plt.ioff()

def process_Gy(lst_arg):

    _quanti_path_file = lst_arg[0]
    _initial_energy = lst_arg[1]
    _final_energy = lst_arg[2]
    _energy_step = lst_arg[3]
    _charge_fluence = lst_arg[4]

    try:
        _calculate_range_Y_N = lst_arg[5]
    except:
        _calculate_range_Y_N = "No"
    try: 
        _target_density = lst_arg[6]
    except:
        _target_density = 2.1  # Default value
    try:
        _mode = lst_arg[7] # Set elements and concentration in mode "manual" or "LivePIXE"
    except:
        _mode = "LivePIXE" # Default value

    print("Final Energy = ", _final_energy, " keV")
    print("Initial Energy = ", _initial_energy, " keV")
    print("Energy step = ", _energy_step, " keV")
    print("Charge fluence = ", _charge_fluence, " µC/cm2")
    print("Calculate range = ", _calculate_range_Y_N)
    print("Target density = ", _target_density, " g/cm³")  
    print("Mode calculation = ", _mode)
        
    # Manual, up to 6 elements
    # EXAMPLE: PbCO3 (Cerussite)
    Z1 = 82
    n1 = 1
    Z2 = 6
    n2 = 1
    Z3 = 8
    n3 = 3
    # Z4 = 1
    # n4 = 0
    # Z5 = 1
    # n5 = 0
    # Z6 = 2
    # n6 = 0

    all_elements_Z = np.loadtxt("coefficients_stopping_power.txt", usecols=0, dtype=int)
    all_elements_atomic_weight = np.loadtxt("atomic_weight.txt", usecols=1)
    all_elements_coefficients_stopping_power = np.loadtxt("coefficients_stopping_power.txt", usecols=(1,2,3,4,5,6,7,8))


    if _mode.upper() == "LIVEPIXE":
    # LivePIXE's file with elemental composition
        matrix_elements_Z = np.loadtxt(_quanti_path_file, skiprows=1, usecols=0, dtype=int)
        matrix_elements_concentration = np.loadtxt(_quanti_path_file , skiprows=1, usecols=1)
        
    if _mode.upper() == "MANUAL":
        matrix_elements_Z = np.array([Z1,Z2,Z3])#,Z4,Z5,Z6])
        #idx_matrix_elements_Z = matrix_elements_Z - 1
        matrix_elements_n = np.array([n1,n2,n3])#,n4,n5,n6])
        matrix_elements_atomic_weight = all_elements_atomic_weight[matrix_elements_Z -1] # Z-1 because index starts at 0
        matrix_elements_mass = matrix_elements_atomic_weight*matrix_elements_n
        total_mass = np.sum(matrix_elements_mass)
        matrix_elements_concentration = matrix_elements_mass/np.sum(matrix_elements_mass)

    sp_all = []
    all_energies = []
    
  
    # 1) Indices des éléments une seule fois
    idx = np.searchsorted(all_elements_Z, matrix_elements_Z)

    atomic_weight = all_elements_atomic_weight[idx]                       # shape (N_el,)
    coefficients_sp = all_elements_coefficients_stopping_power[idx, :]   # shape (N_el, 8)

    # 2) Tableau d'énergies (3000 -> 25, pas -_energy_step)
    all_energies = np.arange(_initial_energy, _final_energy, -_energy_step, dtype=float)   # [3000, 2999, ..., 25]
    # shape (N_E,)

    # 3) Broadcasting sur (N_el, N_E)
    E = all_energies[None, :]              # shape (1, N_E)
    E = np.broadcast_to(E, (len(idx), E.size))  # shape (N_el, N_E)

    a0, a1, a2, a3, a4, a5, a6, a7 = coefficients_sp.T  # chacun shape (N_el, 1)
    a0 = a0[:, None]      # (17,) → (17,1)
    a1 = a1[:, None]      # idem
    a2 = a2[:, None]
    a3 = a3[:, None]
    a4 = a4[:, None]
    a5 = a5[:, None]
    a6 = a6[:, None]
    a7 = a7[:, None]

    SL = a0 * E**a1 + a2 * E**a3
    SH = a4 / E**a5 * np.log(a6 / E + a7 * E)
    S = SL * SH / (SL + SH) * 6.0222e5 / atomic_weight[:, None]  # shape (N_el, N_E)

    sp_all = (S * matrix_elements_concentration[:, None]).sum(axis=0)     # shape (N_E,)
   
    spe = np.round(sp_all[0]/1000,2)
    spm = np.round(np.max(sp_all)/1000,2)
    dpe = np.round(sp_all[0]*_charge_fluence/1e3,2)
    dpm = np.round(np.max(sp_all)*_charge_fluence/1e3,2)

    print("Stopping Power entrance = ",spe, " MeV·cm²/g")
    print("Stopping Power max = ", spm, " MeV·cm²/g")
    print("Dose entrance = ", dpe, " kGy")
    print("Dose max = ", dpm, " kGy")

    with open('output_results.csv', mode='w', newline='' ) as csvfile:
        writer = csv.writer(csvfile,delimiter=';')
        writer.writerow(['Stopping Power entrance (MeV·cm²/g)', 'Stopping Power max (MeV·cm²/g)', 'Dose entrance (kGy)', 'Dose max (kGy)'])
        writer.writerow([spe, spm, dpe, dpm])

    fig1=plt.figure(figsize=(7,5))
    plt.scatter(all_energies, sp_all/1000, color="g", linewidth=0.1,marker=".")
    plt.xlim((  _initial_energy, 0))
    plt.xlabel("Energy (keV)", fontsize = 14)
    plt.ylabel("Stopping Power (MeV·cm²/g)", fontsize = 14)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.title("Stopping Power vs Proton Energy", fontsize = 14)
    fig1.savefig('StoppingPowervsProtonEnergy.png', bbox_inches='tight')
    # plt.show()

    fig2=plt.figure(figsize=(7,5))
    plt.scatter(all_energies, sp_all*_charge_fluence/1e3, color="b",linewidth=0.1, marker=".",linestyle='dotted')
    plt.xlim(( _initial_energy, 0))
    plt.xlabel("Energy (keV)", fontsize = 14)
    plt.ylabel("Dose (kGy)", fontsize = 14)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.title("Dose vs Proton Energy", fontsize = 14)
    fig2.savefig('DosevsProtonEnergy.png', bbox_inches='tight')
    # plt.show()

    with open('stopping_power_calculation_results_energy.csv', mode='w', newline='' ) as csvfile:
        writer = csv.writer(csvfile,delimiter=';')
        writer.writerow(['Energy (keV)', 'Stopping Power (MeV·cm²/g)', 'Dose (kGy)'])
        for energy, sp, dose in zip(all_energies, sp_all/1000, sp_all*_charge_fluence/1e3,):
            writer.writerow([energy, sp, dose])


    dx=0
    depth = []
    if _calculate_range_Y_N == "yes":
        sp_keV_per_cm = sp_all*_target_density  # keV/cm
        dx_all = _energy_step/sp_keV_per_cm # cm

        for n in range(np.shape(sp_keV_per_cm)[0]):
            dx = dx + _energy_step/sp_keV_per_cm[n] # cm 
            depth.append(dx*1e4) # µm
        dx = dx*1e4 # µm
        print("Ion range = ",np.round(dx,2), " µm")
        
        fig3=plt.figure(figsize=(7,5))
        plt.scatter(depth, sp_all/1000, color="g", linewidth=0.1,marker=".")
        plt.xlabel("Depth (µm)", fontsize = 14)
        plt.ylabel("Stopping Power (MeV·cm²/g)", fontsize = 14)
        plt.tick_params(axis='both', which='both', labelsize=14)
        plt.title("Stopping Power vs Depth", fontsize = 14)
        fig3.savefig('StoppingPowervsDepth.png', bbox_inches='tight')

        # plt.show()
        
        fig4=plt.figure(figsize=(7,5))
        plt.scatter(depth, sp_all*_charge_fluence/1e3, color="b", linewidth=0.1,marker=".")
        # plt.xlim((0, 100))
        plt.xlabel("Depth (µm)", fontsize = 14)
        plt.ylabel("Dose (kGy)", fontsize = 14)
        plt.tick_params(axis='both', which='both', labelsize=14)
        plt.title("Dose vs Depth", fontsize = 14)
        fig4.savefig('DosevsDepth.png', bbox_inches='tight')
        # plt.show()
    
        with open('stopping_power_calculation_results_depth.csv', mode='w', newline='' ) as csvfile:
            writer = csv.writer(csvfile,delimiter=';')
            writer.writerow(['Depth (µm)', 'Stopping Power (MeV·cm²/g)', 'Dose (kGy)'])
            for depth_val, sp, dose in zip(depth, sp_all/1000, sp_all*_charge_fluence/1e3):
                writer.writerow([depth_val, sp, dose])

def main(self=None):
      
    if len(sys.argv) < 4:
        print("Usage:  <arg1:Path of elemental composition file from GUPIX or LIVEPIXE> <arg2: initial Energy (keV)> <arg3: final Energy (keV)>"
            " <arg4:Energy step (keV)> <arg5:Charge fluence (µC/cm²)>  <arg6:Calculate range (yes/no)> "
            "<arg7:Target density (g/cm³)> <arg8:modecalculation (LivePIXE/manual)>")
             
        lst_arg = ["C:\\LivePIXE\\MAT_X0\\LAST_MTX.OUT", 3000,25, 0.50, 2.1, "yes", 2,"Manual"]
        print("Processing with default parameters for testing...")
       # lst_arg = ["C:\\LivePIXE\\MAT_X0\\LAST_MTX.OUT", 3000,25, 0.50, 2]
        process_Gy(lst_arg)
    else:
        process_Gy(lst_arg)

if __name__ == "__main__":
   lst_arg = sys.argv
   main()





