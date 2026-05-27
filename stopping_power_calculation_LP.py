# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:24:51 2024
@author: diana.bachiller
"""

"""
Modified May 2026
@author: Laurent Pichon 
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import sys
import csv
plt.ioff()

def process_gray_calculation(lst_arg):
    _energy = 3000 # (keV)
    _energy_step = 0.5 # (keV)
    _charge_fluence = 1 # (µC/cm2) OPTIONAL to calculate dose
    _quanti_path_file = lst_arg[0]
    _energy = lst_arg[1]
    _energy_step = lst_arg[2]
    _charge_fluence = lst_arg[3]
    _calculate_range_Y_N = lst_arg[4]
    _target_density = lst_arg[5]
    print("Energy = ", _energy, " keV")
    print("Energy step = ", _energy_step, " keV")
    print("Charge fluence = ", _charge_fluence, " µC/cm2")
    print("Calculate range = ", _calculate_range_Y_N)
    print("Target density = ", _target_density, " g/cm³")  
   
    mode = "LivePIXE" # Set elements and concentration in mode "manual" or "LivePIXE"
       

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
        # matrix_elements_Z = np.loadtxt("elemental_composition_LivePIXE.txt", skiprows=1, usecols=0, dtype=int)
        # matrix_elements_concentration = np.loadtxt("elemental_composition_LivePIXE.txt", skiprows=1, usecols=1)
        matrix_elements_Z = np.loadtxt(_quanti_path_file, skiprows=1, usecols=0, dtype=int)
        matrix_elements_concentration = np.loadtxt(_quanti_path_file , skiprows=1, usecols=1)
        
    if mode == "manual":
        matrix_elements_Z = np.array([Z1,Z2,Z3,Z4,Z5,Z6])
        matrix_elements_n = np.array([n1,n2,n3,n4,n5,n6])
        matrix_elements_atomic_weight = all_elements_atomic_weight[matrix_elements_Z]
        matrix_elements_mass = matrix_elements_atomic_weight*matrix_elements_n
        total_mass = np.sum(matrix_elements_mass)
        matrix_elements_concentration = matrix_elements_mass/np.sum(matrix_elements_mass)

    sp_all = []
    all_energies = []
    initial_energy = np.copy(_energy)
    time_st1 = time.time()
    # 1) Indices des éléments une seule fois
    idx = np.searchsorted(all_elements_Z, matrix_elements_Z)

    atomic_weight = all_elements_atomic_weight[idx]                       # shape (N_el,)
    coefficients_sp = all_elements_coefficients_stopping_power[idx, :]   # shape (N_el, 8)

    # 2) Tableau d'énergies (3000 -> 25, pas -1)
    all_energies = np.arange(initial_energy, 24, -_energy_step, dtype=float)   # [3000, 2999, ..., 25]
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

    # 4) Combinaison par les concentrations
    sp_all = (S * matrix_elements_concentration[:, None]).sum(axis=0)     # shape (N_E,)

    # time_lp =  time.time() - time_st1
    # print("Time for stopping power calculation with LP = ", np.round(time_lp, 2), " seconds")

    # time_st2 = time.time()
    # sp_all = []
    # all_energies = []
    # initial_energy = np.copy(_energy)

    # while _energy > 25:
    #     S_all = []
    #     for n in range(np.shape(matrix_elements_Z)[0]):
    #         Z = matrix_elements_Z[n]
    #         concentration = matrix_elements_concentration[n]
    #         #print(Z,concentration,np.where(all_elements_Z==Z)[0][0])
    #         atomic_weight = all_elements_atomic_weight[np.where(all_elements_Z==Z)[0][0]]
    #         #print(atomic_weight)
    #         coefficients_sp = all_elements_coefficients_stopping_power[np.where(all_elements_Z==Z)[0][0]]
    #         SL = coefficients_sp[0]*_energy**coefficients_sp[1]+coefficients_sp[2]*_energy**coefficients_sp[3]
    #         SH = coefficients_sp[4]/_energy**coefficients_sp[5]*np.log(coefficients_sp[6]/_energy+coefficients_sp[7]*_energy)
    #         S = SL*SH/(SL+SH)*6.0222e5/atomic_weight
    #         S_all.append(S)
    #     sp = np.sum(np.array(S_all)*matrix_elements_concentration)
    #     sp_all.append(sp)
    #     all_energies.append(_energy)
    #     _energy = _energy - _energy_step



    sp_all = np.array(sp_all) # Stopping power in keV·cm²/g
    # time_diana =  time.time() - time_st2
    # print("Time for stopping power calculation with Diana = ", np.round(time_diana, 2), " seconds")

    print("Stopping Power entrance = ",np.round(sp_all[0]/1000,2), " MeV·cm²/g")
    print("Stopping Power max = ", np.round(np.max(sp_all)/1000,2), " MeV·cm²/g")
    print("Dose entrance = ", np.round(sp_all[0]*_charge_fluence/1e3,2), " kGy")
    print("Dose max = ", np.round(np.max(sp_all)*_charge_fluence/1e3,2), " kGy")

    

    fig1=plt.figure(figsize=(7,5))
    plt.scatter(all_energies, sp_all/1000, color="g", linewidth=0.1,marker=".")
    plt.xlim((initial_energy, 0))
    plt.xlabel("Energy (keV)", fontsize = 14)
    plt.ylabel("Stopping Power (MeV·cm²/g)", fontsize = 14)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.title("Stopping Power vs Proton Energy", fontsize = 14)
    fig1.savefig('StoppingPowervsProtonEnergy.png', bbox_inches='tight')
    # plt.show()

    fig2=plt.figure(figsize=(7,5))
    plt.scatter(all_energies, sp_all*_charge_fluence/1e3, color="b",linewidth=0.1, marker=".",linestyle='dotted')
    plt.xlim((initial_energy, 0))
    plt.xlabel("Energy (keV)", fontsize = 14)
    plt.ylabel("Dose (kGy)", fontsize = 14)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.title("Dose vs Proton Energy", fontsize = 14)
    fig2.savefig('DosevsProtonEnergy.png', bbox_inches='tight')
    # plt.show()

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

        with open('stopping_power_calculation_results_energy.csv', mode='w', newline='' ) as csvfile:
            writer = csv.writer(csvfile,delimiter=';')
            writer.writerow(['Energy (keV)', 'Stopping Power (MeV·cm²/g)', 'Dose (kGy)'])
            for energy, sp, dose in zip(all_energies, sp_all/1000, sp_all*_charge_fluence/1e3,):
                writer.writerow([energy, sp, dose])

        with open('stopping_power_calculation_results_depth.csv', mode='w', newline='' ) as csvfile:
            writer = csv.writer(csvfile,delimiter=';')
            writer.writerow(['Depth (µm)', 'Stopping Power (MeV·cm²/g)', 'Dose (kGy)'])
            for depth_val, sp, dose in zip(depth, sp_all/1000, sp_all*_charge_fluence/1e3):
                writer.writerow([depth_val, sp, dose])

def main(self=None):
    print("hello master")
    
    _path_lst = ""
    _fnct = "maps"
    _path_ascii = ""
    _ibil = "" 

    if len(sys.argv) < 2:
        print("Usage:  <arg1:Path of elemental composition from LIVEPIXE> <arg2: Energy> <arg3:Energy step> <arg4:Charge fluence>  <arg5:Calculate range (yes/no)> <arg6:Target density (g/cm³)>")
        lst_arg = ["C:\\LivePIXE\\MAT_X0\\LAST_MTX.OUT", 3000, 0.50, 1, "yes", 2.9]
        process_gray_calculation(lst_arg)
      
    else:
        process_gray_calculation(lst_arg)

if __name__ == "__main__":
   lst_arg = sys.argv
   main()





