# Instructions for Running KMC-MD Simulation:
# This script was created for the execution of Kinetic Monte Carlo (KMC) coupled with Molecular Dynamics (MD) simulations,
# for studying the dehydrochlorination (DHC) process of PVC. 

# Key Configuration Parameters:
# 1. PVC chains are differentiated by chain length using the following residue names:
#    - PVA for N=5, PVB for N=20, PVC for N=40, PVD for N=60, PVE for N=100, PVF for N=120.
#    Additional residue names include NAS (sodium), OH (hydroxide), NAC (sodium chloride), and SOL (water).

# 2. Simulation Temperature and Initial Structure file: 
#    - Set the simulation temperature (in Kelvin) at line 66 and initial structure (npt.gro) at line 174.

# 3. Reaction Rate Catalog: 
#    - Edit reaction rates between lines 149-161.

# 4. Cutoff Radius for Rate Calculation: 
#    - Adjust the cutoff radius (in Angstroms) for rate calculations at line 166.

# 5. Forcefield Atom Types: 
#    - Modify the [atomtypes] section of the forcefield.itp file at line 1236 as required.

# 6. Initial NaOH Molecules: 
#    - Edit the initial number of NaOH molecules at line 1518.

# 7. Iteration settings: 
#    - Set the maximum number of iterations at line 1521.
#    - Define the rate threshold (in /s) below which the simulation halts at line 1522.

# 8. Initial number of PVC Chains: 
#    - Edit the initial count of PVC chains at line 1525.

# 9. Simulation Directory: 
#    - Assign the base directory for simulation output at lines 1528 and 1545.

# 10. % DHC Calculation: 
#     - Adjust the calculation method for % DHC at line 1644.

# 11. MD Execution: 
#     - Adjust the commands for the MD simulation between lines 1354 and 1383
#     - Set the frequency of MD stage execution relative to KMC stages at line 1653. 
#       e.g. 'if iteration % 1 == 0' signifies MD stage execution after every KMC stage.

# Ensure that all modifications align with the specific requirements of your simulation.


import os
import time
import subprocess
import shutil
from shutil import copy
import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds
from MDAnalysis import Universe
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.core.topologyattrs import Bonds
from MDAnalysis.core.groups import AtomGroup
import glob
import re
import pyfiglet
from tqdm import tqdm
import time
import itertools
import threading

k_B = 1.380649E-23 # Boltzmann constant in J/K
T = 300 # Temperature in Kelvin

# Display on terminal
def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

# Display on terminal
def run_spinner(description):
    spinner = spinning_cursor()
    print(f"\n{description}... ", end='', flush=True)
    for _ in range(50):  
        print(f'\r{description}... {next(spinner)}', end='', flush=True)
        time.sleep(0.1)
    print('\r', end='\n')

# Parse topology info from ITP files    
def parse_itp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    bonds_dict = {}
    read_bonds = False
    
    for line in lines:
        line = line.strip() 

        if line.startswith(';'):
            continue

        if "[ bonds ]" in line:
            read_bonds = True
            continue
        elif "[" in line and "]" in line:
            read_bonds = False
        
        if read_bonds:
            atoms = line.split()[:2]
            if atoms:
                atom1, atom2 = map(lambda x: int(x)-1, atoms) 
                if atom1 in bonds_dict:
                    bonds_dict[atom1].append(atom2)
                else:
                    bonds_dict[atom1] = [atom2]
                if atom2 in bonds_dict:
                    bonds_dict[atom2].append(atom1)
                else:
                    bonds_dict[atom2] = [atom1]
    return bonds_dict

# Register the bond info from ITP files
def create_bonds(u, itp_files):
    bond_info = {}
    for itp_file in itp_files:
        # Extract the molecule type from the ITP filename
        molecule_type = itp_file.split('.')[0].upper()

        # Use the parse_itp function to read bond info from ITP file and store it in bond_info
        bond_info[molecule_type] = parse_itp(itp_file)

    all_bonds = []
    for residue in u.residues:
        molecule_type = residue.resname.upper()
        if molecule_type in bond_info:
            bonds_dict = bond_info[molecule_type]

            # Create a mapping of atom indices to atom names for atoms in this residue
            atom_idxs_to_names = {atom.ix: atom.name for atom in residue.atoms}

            # Iterate through each bond in the bond dictionary
            for atom_index, bonded_atoms in bonds_dict.items():
                for bonded_atom in bonded_atoms:

                    # Check if the bond between these atoms has not already been added
                    if not ((residue.atoms[atom_index].index, residue.atoms[bonded_atom].index) in all_bonds or
                            (residue.atoms[bonded_atom].index, residue.atoms[atom_index].index) in all_bonds):
                        all_bonds.append((residue.atoms[atom_index].index, residue.atoms[bonded_atom].index))

    # Add the bonds as a topology attribute to the universe
    u.add_TopologyAttr('bonds', all_bonds)

# Reaction catalog
reactions = {
    1: {'Ea': 60380, 'k': 3.8E+07},
    2: {'Ea': 62560, 'k': 1.40E+07},
    3: {'Ea': 58520, 'k': 3.34E+07},
    4: {'Ea': 63220, 'k': 2.69E+08},
    5: {'Ea': 64650, 'k': 5.91E+06},
    6: {'Ea': 63750, 'k': 5.04E+05},
    7: {'Ea': 105140, 'k': 1.05E+06},
    8: {'Ea': 78800, 'k': 1.24E+04},
    9: {'Ea': 100090, 'k': 1.28E+03},
    10: {'Ea': 76570, 'k': 2.84E+06},
    11: {'Ea': 92080, 'k': 2.89E+04},
    12: {'Ea': 84300, 'k': 1.69E+05}
}

# Rate calculation
def calculate_rate(distance, reaction):
    r_cut = 4.0 # in Angstroms
    Ea = reactions[reaction]['Ea']
    k_prefactor = reactions[reaction]['k']
    exp_term = np.where(distance > r_cut, 0, np.exp(-distance/r_cut))
    rate = k_prefactor * np.exp(-Ea/(k_B * T * 6.02E+23)) * exp_term
    return rate

# Initialize sets to keep track of various atoms and residues involved in reactions
u = mda.Universe('npt.gro') # Name of the initial structure file
H_atoms = u.select_atoms("name H* and resname PV*")
H_states = np.zeros(len(H_atoms), dtype=int)
reacted_OH_atoms = set()
reacted_Cl_atoms = set()
double_bonded_C_atoms = set()
hydrogen_bonded_to_OH_atoms = set()
selected_H_atoms = set()
target_Na_atoms = set()
OH_bonded_to_C_atoms = set()
hydrogen_bonded_to_OH_bonded_to_C_atoms = set()
reacted_residue_names = set()
new_residue_names = set()
reacted_terminal_carbons = set()

# Identify the terminal carbon of the PVC chain which is participating in the reaction
def find_terminal_carbon(residue):
    for carbon in residue.atoms.select_atoms("name C*"):
        if len([atom for atom in carbon.bonded_atoms if atom.name.startswith('H')]) == 3:
            return carbon
    for carbon in residue.atoms.select_atoms("name C*"):
        bonded_hydrogens = [atom for atom in carbon.bonded_atoms if atom.name.startswith('H')]
        if len(bonded_hydrogens) == 2 and carbon.index in double_bonded_C_atoms:
            return carbon
    return None

# Store info of atoms that participated in reaction 1
def reaction1(selected_H, u):
    C_bonded_to_H = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    terminal_carbon = find_terminal_carbon(selected_H.residue)
    print("Terminal carbon:", terminal_carbon)
    if C_bonded_to_H == terminal_carbon:
        other_double_bonded_C = next((atom for atom in terminal_carbon.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    else:
        max_distance = 0
        other_double_bonded_C = None
        for C_neighbor in C_bonded_to_H.bonded_atoms:
            if C_neighbor.name.startswith('C') and C_neighbor != C_bonded_to_H and not C_neighbor.name.startswith('Cl'):
                distance = np.linalg.norm(terminal_carbon.position - C_neighbor.position)
                if distance > max_distance:
                    max_distance = distance
                    other_double_bonded_C = C_neighbor

        if not other_double_bonded_C:
            return False
    closest_Cl = next((atom for atom in other_double_bonded_C.bonded_atoms if atom.name.startswith('Cl')), None)
    if not closest_Cl:
        return False
    selected_H_atoms.add(selected_H.index)
    reacted_Cl_atoms.add(closest_Cl.index)
    distances_to_OH = distance_array(selected_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(distances_to_OH)
    closest_OH = OH_atoms[closest_OH_idx]
    reacted_OH_atoms.add(closest_OH.index)
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(closest_Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    target_Na = Na_atoms[np.argmin(Na_distances)]
    target_Na_atoms.add(target_Na.index)
    double_bonded_C_atoms.add(C_bonded_to_H.index)
    double_bonded_C_atoms.add(other_double_bonded_C.index)
    if terminal_carbon:
        reacted_terminal_carbons.add(terminal_carbon.index)
    hydrogen_bonded_to_OH = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_atoms.add(hydrogen_bonded_to_OH.index)
    return True

# Store info of atoms that participated in reaction 2
def reaction2(selected_H, u):
    C_bonded_to_H = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    terminal_carbon = find_terminal_carbon(selected_H.residue)
    if C_bonded_to_H == terminal_carbon:
        other_double_bonded_C = next((atom for atom in terminal_carbon.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    else:
        other_double_bonded_C = next((C_neighbor for C_neighbor in C_bonded_to_H.bonded_atoms if C_neighbor.name.startswith('C') and not C_neighbor.name.startswith('Cl') and any(atom.name.startswith('Cl') for atom in C_neighbor.bonded_atoms)), None)

    if not other_double_bonded_C:
        return False
        
    closest_Cl = next((atom for atom in other_double_bonded_C.bonded_atoms if atom.name.startswith('Cl')), None)
    if not closest_Cl:
        return False
    selected_H_atoms.add(selected_H.index)
    reacted_Cl_atoms.add(closest_Cl.index)
    distances_to_OH = distance_array(selected_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(distances_to_OH)
    closest_OH = OH_atoms[closest_OH_idx]
    reacted_OH_atoms.add(closest_OH.index)
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(closest_Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    target_Na = Na_atoms[np.argmin(Na_distances)]
    target_Na_atoms.add(target_Na.index)
    double_bonded_C_atoms.add(C_bonded_to_H.index)
    double_bonded_C_atoms.add(other_double_bonded_C.index)
    if terminal_carbon:
        reacted_terminal_carbons.add(terminal_carbon.index)
    hydrogen_bonded_to_OH = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_atoms.add(hydrogen_bonded_to_OH.index)
    return True        

# Store info of atoms that participated in reaction 3
def reaction3(selected_H, u):
    C_bonded_to_H = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    terminal_carbon = find_terminal_carbon(selected_H.residue)
    if C_bonded_to_H == terminal_carbon:
        other_double_bonded_C = next((atom for atom in terminal_carbon.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and any(atom.name.startswith('Cl') for atom in atom.bonded_atoms)), None)
    else:
        max_distance = 0
        other_double_bonded_C = None
        for C_neighbor in C_bonded_to_H.bonded_atoms:
            if C_neighbor.name.startswith('C') and not C_neighbor.name.startswith('Cl'):
                distance = np.linalg.norm(terminal_carbon.position - C_neighbor.position)
                if distance > max_distance and any(atom.name.startswith('Cl') for atom in C_neighbor.bonded_atoms):
                    max_distance = distance
                    other_double_bonded_C = C_neighbor

    if not other_double_bonded_C:
        return False
        
    closest_Cl = next((atom for atom in other_double_bonded_C.bonded_atoms if atom.name.startswith('Cl')), None)
    if not closest_Cl:
        return False
    selected_H_atoms.add(selected_H.index)
    reacted_Cl_atoms.add(closest_Cl.index)
    distances_to_OH = distance_array(selected_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(distances_to_OH)
    closest_OH = OH_atoms[closest_OH_idx]
    reacted_OH_atoms.add(closest_OH.index)
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(closest_Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    target_Na = Na_atoms[np.argmin(Na_distances)]
    target_Na_atoms.add(target_Na.index)
    double_bonded_C_atoms.add(C_bonded_to_H.index)
    double_bonded_C_atoms.add(other_double_bonded_C.index)
    if terminal_carbon:
        reacted_terminal_carbons.add(terminal_carbon.index)
    hydrogen_bonded_to_OH = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_atoms.add(hydrogen_bonded_to_OH.index)
    return True        

# Store info of atoms that participated in reaction 4
def reaction4(selected_H, u):
    C_bonded_to_H = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    terminal_carbon = find_terminal_carbon(selected_H.residue)
    if C_bonded_to_H == terminal_carbon:
        other_double_bonded_C = next((atom for atom in terminal_carbon.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and any(atom.name.startswith('Cl') for atom in atom.bonded_atoms)), None)
    else:
        max_distance = 0
        other_double_bonded_C = None
        for C_neighbor in C_bonded_to_H.bonded_atoms:
            if C_neighbor.name.startswith('C') and not C_neighbor.name.startswith('Cl'):
                distance = np.linalg.norm(terminal_carbon.position - C_neighbor.position)
                if distance > max_distance and any(atom.name.startswith('Cl') for atom in C_neighbor.bonded_atoms):
                    max_distance = distance
                    other_double_bonded_C = C_neighbor

    if not other_double_bonded_C:
        return False
        
    closest_Cl = next((atom for atom in other_double_bonded_C.bonded_atoms if atom.name.startswith('Cl')), None)
    if not closest_Cl:
        return False
    selected_H_atoms.add(selected_H.index)
    reacted_Cl_atoms.add(closest_Cl.index)
    distances_to_OH = distance_array(selected_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(distances_to_OH)
    closest_OH = OH_atoms[closest_OH_idx]
    reacted_OH_atoms.add(closest_OH.index)
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(closest_Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    target_Na = Na_atoms[np.argmin(Na_distances)]
    target_Na_atoms.add(target_Na.index)
    double_bonded_C_atoms.add(C_bonded_to_H.index)
    double_bonded_C_atoms.add(other_double_bonded_C.index)
    if terminal_carbon:
        reacted_terminal_carbons.add(terminal_carbon.index)
    hydrogen_bonded_to_OH = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_atoms.add(hydrogen_bonded_to_OH.index)
    return True

# Store info of atoms that participated in reaction 5
def reaction5(selected_H, u):
    C_bonded_to_H = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    terminal_carbon = find_terminal_carbon(selected_H.residue)
    if C_bonded_to_H == terminal_carbon:
        other_double_bonded_C = next((atom for atom in terminal_carbon.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    else:
        other_double_bonded_C = next((C_neighbor for C_neighbor in C_bonded_to_H.bonded_atoms if C_neighbor.name.startswith('C') and not C_neighbor.name.startswith('Cl') and any(atom.name.startswith('Cl') for atom in C_neighbor.bonded_atoms)), None)

    if not other_double_bonded_C:
        return False
        
    closest_Cl = next((atom for atom in other_double_bonded_C.bonded_atoms if atom.name.startswith('Cl')), None)
    if not closest_Cl:
        return False
    selected_H_atoms.add(selected_H.index)
    reacted_Cl_atoms.add(closest_Cl.index)
    distances_to_OH = distance_array(selected_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(distances_to_OH)
    closest_OH = OH_atoms[closest_OH_idx]
    reacted_OH_atoms.add(closest_OH.index)
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(closest_Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    target_Na = Na_atoms[np.argmin(Na_distances)]
    target_Na_atoms.add(target_Na.index)
    double_bonded_C_atoms.add(C_bonded_to_H.index)
    double_bonded_C_atoms.add(other_double_bonded_C.index)
    if terminal_carbon:
        reacted_terminal_carbons.add(terminal_carbon.index)
    hydrogen_bonded_to_OH = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_atoms.add(hydrogen_bonded_to_OH.index)
    return True

# Store info of atoms that participated in reaction 6    
def reaction6(selected_H, u):
    C1 = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C1 is None:
        return False
    C2 = next((atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom.index in double_bonded_C_atoms), None)
    if C2 is None:
        return False
    C3 = next((atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')) and C3 != C2 and C3 != C1, None)
    Cl = next((atom for atom in C1.bonded_atoms if atom.name.startswith('Cl')), None)
    if Cl is None:
        return False
    OH_atoms = u.select_atoms("name Oh and resname OH")
    OH_distances = distance_array(C1.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(OH_distances)
    closest_OH = OH_atoms[closest_OH_idx]
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    closest_Na = Na_atoms[np.argmin(Na_distances)]
    double_bonded_C_atoms.remove(C3.index)
    double_bonded_C_atoms.add(C1.index)   
    reacted_Cl_atoms.add(Cl.index)        
    target_Na_atoms.add(closest_Na.index) 
    OH_bonded_to_C_atoms.add((closest_OH.index, C1.index))
    hydrogen_bonded_to_OH_bonded_to_C = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_bonded_to_C_atoms.add(hydrogen_bonded_to_OH_bonded_to_C.index)
    return True

# Store info of atoms that participated in reaction 7
def reaction7(selected_H, u):
    C1 = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C1 is None:
        return False
    C2 = next((atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom.index in double_bonded_C_atoms), None)
    if C2 is None:
        return False
    C3 = next((atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    Cl = next((atom for atom in C3.bonded_atoms if atom.name.startswith('Cl')), None)
    if Cl is None:
        return False
    OH_atoms = u.select_atoms("name Oh and resname OH")
    OH_distances = distance_array(C1.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(OH_distances)
    closest_OH = OH_atoms[closest_OH_idx]
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    closest_Na = Na_atoms[np.argmin(Na_distances)]
    double_bonded_C_atoms.remove(C1.index)
    double_bonded_C_atoms.add(C3.index)   
    reacted_Cl_atoms.add(Cl.index)        
    reacted_OH_atoms.add(closest_OH.index)
    target_Na_atoms.add(closest_Na.index) 
    OH_bonded_to_C_atoms.add((closest_OH.index, C1.index))
    hydrogen_bonded_to_OH_bonded_to_C = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_bonded_to_C_atoms.add(hydrogen_bonded_to_OH_bonded_to_C.index)
    return True

# Store info of atoms that participated in reaction 8
def reaction8(selected_H, u):
    OH_atoms = u.select_atoms("name Oh and resname OH")
    OH_distances = distance_array(C1.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(OH_distances)
    closest_OH = OH_atoms[closest_OH_idx]
    C1 = [atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')][0]
    C2 = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != selected_H][0]
    C3 = [atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C1][0]
    C4 = [atom for atom in C3.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C2 and atom.index in double_bonded_C_atoms][0]
    C5 = [atom for atom in C4.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C3][0]
    Cl_bonded_to_C5 = [atom for atom in C5.bonded_atoms if atom.name.startswith('Cl')][0]
    OH_bonded_to_C_atoms.add((closest_OH.index, C1.index))
    hydrogen_bonded_to_OH_bonded_to_C = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_bonded_to_C_atoms.add(hydrogen_bonded_to_OH_bonded_to_C.index)
    double_bonded_C_atoms.add(C5.index) 
    double_bonded_C_atoms.remove(C1.index)
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(Cl_bonded_to_C5.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    closest_Na = Na_atoms[np.argmin(Na_distances)]
    reacted_Cl_atoms.add(Cl_bonded_to_C5.index)
    target_Na_atoms.add(closest_Na.index)

# Store info of atoms that participated in reaction 9
def reaction9(selected_H, u):
    C_bonded_to_H = [atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')][0]
    OH_atoms = u.select_atoms("name Oh and resname OH")
    OH_distances = distance_array(C_bonded_to_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(OH_distances)
    closest_OH = OH_atoms[closest_OH_idx]
    Cl_bonded_to_C = [atom for atom in C_bonded_to_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')][0]
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(Cl_bonded_to_C.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    closest_Na = Na_atoms[np.argmin(Na_distances)]
    OH_bonded_to_C_atoms.add((closest_OH.index, C_bonded_to_H.index))
    hydrogen_bonded_to_OH_bonded_to_C = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_bonded_to_C_atoms.add(hydrogen_bonded_to_OH_bonded_to_C.index)
    reacted_Cl_atoms.add(Cl_bonded_to_C.index)
    target_Na_atoms.add(closest_Na.index)

# Store info of atoms that participated in reaction 10
def reaction10(selected_H, u):
    C_bonded_to_H = next((atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if not C_bonded_to_H or C_bonded_to_H.index in double_bonded_C_atoms:
        return False
    closest_Cl = None
    for C_neighbor in C_bonded_to_H.bonded_atoms:
        if C_neighbor.name.startswith('C') and not atom.name.startswith('Cl'):
            Cl_atoms_bonded_to_neighbor = [atom for atom in C_neighbor.bonded_atoms if atom.name.startswith('Cl')]
            if Cl_atoms_bonded_to_neighbor:
                closest_Cl = Cl_atoms_bonded_to_neighbor[0]
                break
    if not closest_Cl:
        return False
    reacted_Cl_atoms.add(closest_Cl.index)
    distances_to_OH = distance_array(selected_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(distances_to_OH)
    closest_OH = OH_atoms[closest_OH_idx]
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(closest_Cl.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    target_Na = Na_atoms[np.argmin(Na_distances)]
    double_bonded_C_atoms.add(C_bonded_to_H.index)
    double_bonded_C_atoms.add(closest_Cl.bonded_atoms[0].index)  
    hydrogen_bonded_to_OH_atoms.add(hydrogen_bonded_to_OH.index)
    reacted_OH_atoms.add(closest_OH.index)
    selected_H_atoms.add(selected_H.index)
    target_Na_atoms.add(target_Na.index)
    return True

# Store info of atoms that participated in reaction 11
def reaction11(selected_H, u):
    C_bonded_to_H = [atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')][0]
    OH_atoms = u.select_atoms("name Oh and resname OH")
    OH_distances = distance_array(C_bonded_to_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(OH_distances)
    closest_OH = OH_atoms[closest_OH_idx]
    Cl_bonded_to_C = [atom for atom in C_bonded_to_H.bonded_atoms if atom.name.startswith('Cl')][0]
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(Cl_bonded_to_C.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    closest_Na = Na_atoms[np.argmin(Na_distances)]
    OH_bonded_to_C_atoms.add((closest_OH.index, C_bonded_to_H.index))
    hydrogen_bonded_to_OH_bonded_to_C = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_bonded_to_C_atoms.add(hydrogen_bonded_to_OH_bonded_to_C.index)
    reacted_Cl_atoms.add(Cl_bonded_to_C.index)
    target_Na_atoms.add(closest_Na.index)

# Store info of atoms that participated in reaction 12
def reaction12(selected_H, u):
    C_bonded_to_H = [atom for atom in selected_H.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')][0]
    OH_atoms = u.select_atoms("name Oh and resname OH")
    OH_distances = distance_array(C_bonded_to_H.position[np.newaxis, :], OH_atoms.positions, box=u.dimensions)[0]
    closest_OH_idx = np.argmin(OH_distances)
    closest_OH = OH_atoms[closest_OH_idx]
    Cl_bonded_to_C = [atom for atom in C_bonded_to_H.bonded_atoms if atom.name.startswith('Cl')][0]
    Na_atoms = u.select_atoms("name Na and resname NAS")
    Na_distances = distance_array(Cl_bonded_to_C.position[np.newaxis, :], Na_atoms.positions, box=u.dimensions)[0]
    closest_Na = Na_atoms[np.argmin(Na_distances)]
    OH_bonded_to_C_atoms.add((closest_OH.index, C_bonded_to_H.index))
    hydrogen_bonded_to_OH_bonded_to_C = u.select_atoms(f"resid {closest_OH.resid} and name H*").atoms[0]
    hydrogen_bonded_to_OH_bonded_to_C_atoms.add(hydrogen_bonded_to_OH_bonded_to_C.index)
    reacted_Cl_atoms.add(Cl_bonded_to_C.index)
    target_Na_atoms.add(closest_Na.index)

# Assess local environment of each H site to determine if Reaction 1 could occur at each site    
def is_valid_for_reaction1(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C_bonded_to_H = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C_bonded_to_H is None or C_bonded_to_H.index in double_bonded_C_atoms:
        return False

    if any(atom.name.startswith('Cl') for atom in C_bonded_to_H.bonded_atoms):
        return False
    current_C = C_bonded_to_H
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)

    def check_no_double_bonded_chain(C, depth):
        if depth > 5:
            return True
        for next_C in C.bonded_atoms:
            if next_C.name.startswith('C') and not next_C.name.startswith('Cl') and next_C != C:
                if next_C.index in double_bonded_C_atoms:
                    return False
                if not check_no_double_bonded_chain(next_C, depth + 1):
                    return False
        return True

    return check_no_double_bonded_chain(C1, 1)
    valid_chain = False
    for C2 in C1.bonded_atoms:
        if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1 and C2.index not in double_bonded_C_atoms:
            for C3 in C2.bonded_atoms:
                if C3.name.startswith('C') and not C3.name.startswith('Cl') and C3 != C2 and C3 != C1 and C3.index not in double_bonded_C_atoms:
                    for C4 in C3.bonded_atoms:
                        if C4.name.startswith('C') and not C4.name.startswith('Cl') and C4 != C3 and C4 != C2 and C4 != C1 and C4.index not in double_bonded_C_atoms:
                            for C5 in C4.bonded_atoms:
                                if C5.name.startswith('C') and not C5.name.startswith('Cl') and C5 != C4 and C5 != C3 and C5 != C2 and C5 != C1 and C5.index not in double_bonded_C_atoms:
                                    for C6 in C5.bonded_atoms:
                                        if C6.name.startswith('C') and not C6.name.startswith('Cl') and C6 != C5 and C6 != C4 and C6 != C3 and C6 != C2 and C6 != C1 and C6.index not in double_bonded_C_atoms:
                                            valid_chain = True
                                            break
                                    if valid_chain:
                                        break
                            if valid_chain:
                                break
                    if valid_chain:
                        break
            if valid_chain:
                break
    if not valid_chain:
        return False
    for _ in range(4):
        next_Cs = [atom for atom in current_C.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom.index != H_atom.index]
        if not next_Cs:
            return False
        next_C = next_Cs[0]
        if next_C.index in double_bonded_C_atoms:
            return False
        current_C = next_C
    return True

# Assess local environment of each H site to determine if Reaction 2 could occur at each site   
def is_valid_for_reaction2(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C1 is None or C1.index in double_bonded_C_atoms or any(atom.name.startswith('Cl') for atom in C1.bonded_atoms):
        return False
    C2_atoms = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')]
    if len(C2_atoms) != 2:
        return False
    C2_with_Cl = any(atom.name.startswith('Cl') for atom in C2_atoms[0].bonded_atoms) or any(atom.name.startswith('Cl') for atom in C2_atoms[1].bonded_atoms)
    C2_double_bonded = C2_atoms[0].index in double_bonded_C_atoms or C2_atoms[1].index in double_bonded_C_atoms
    if not (C2_with_Cl and C2_double_bonded):
        return False
    if all(C2.index in double_bonded_C_atoms for C2 in C2_atoms):
        return False
    for C2 in C2_atoms:
        if C2.index in double_bonded_C_atoms and C2 != C1:
            for C3 in C2.bonded_atoms:
                if C3.name.startswith('C') and not C3.name.startswith('Cl') and C3 != C2 and C3 != C1 and C3.index in double_bonded_C_atoms:
                    for C4 in C3.bonded_atoms:
                        if C4.name.startswith('C') and not C4.name.startswith('Cl') and C4 != C3 and C4 != C2 and C4 != C1 and C4.index not in double_bonded_C_atoms and any(atom.name.startswith('Cl') for atom in C4.bonded_atoms):
                            return True
    return False

# Assess local environment of each H site to determine if Reaction 3 could occur at each site   
def is_valid_for_reaction3(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C1 is None or C1.index in double_bonded_C_atoms:
        return False
    C2_atoms = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C1]
    has_Cl_bonded_to_C2 = any(any(atom.name.startswith('Cl') for atom in C2.bonded_atoms) for C2 in C2_atoms)
    if not has_Cl_bonded_to_C2:
        return False
    double_bonded_C3_found = False
    double_bonded_C3_C4_sequence_found = False
    for C2 in C2_atoms:
        C3_atoms = [atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C2 and atom != C1]
        for C3 in C3_atoms:
            if C3.index in double_bonded_C_atoms:
                double_bonded_C3_found = True
                C4_atoms = [atom for atom in C3.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C3 and atom != C2 and atom != C1]
                for C4 in C4_atoms:
                    if C4.index in double_bonded_C_atoms:
                        double_bonded_C3_C4_sequence_found = True
                        C5_atoms = [atom for atom in C4.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                        for C5 in C5_atoms:
                            if C5.index in double_bonded_C_atoms:
                                return False
    return double_bonded_C3_found and double_bonded_C3_C4_sequence_found

# Assess local environment of each H site to determine if Reaction 4 could occur at each site   
def is_valid_for_reaction4(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C1 is None or C1.index in double_bonded_C_atoms:
        return False
    C2_atoms = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C1]
    has_Cl_bonded_to_C2 = any(any(atom.name.startswith('Cl') for atom in C2.bonded_atoms) for C2 in C2_atoms)
    if not has_Cl_bonded_to_C2:
        return False
    double_bonded_C3_to_C7_sequence_found = False
    for C2 in C2_atoms:
        C3_atoms = [atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C2 and atom != C1]
        for C3 in C3_atoms:
            if C3.index in double_bonded_C_atoms:
                C4_atoms = [atom for atom in C3.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C3 and atom != C2 and atom != C1]
                for C4 in C4_atoms:
                    if C4.index in double_bonded_C_atoms:
                        C5_atoms = [atom for atom in C4.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                        for C5 in C5_atoms:
                            if C5.index in double_bonded_C_atoms:
                                C6_atoms = [atom for atom in C5.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C5 and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                                for C6 in C6_atoms:
                                    if C6.index in double_bonded_C_atoms:
                                        C7_atoms = [atom for atom in C6.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C6 and atom != C5 and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                                        for C7 in C7_atoms:
                                            if C7.index not in double_bonded_C_atoms:
                                                double_bonded_C3_to_C7_sequence_found = True
                                                break
                                        if double_bonded_C3_to_C7_sequence_found:
                                            break
                                    if double_bonded_C3_to_C7_sequence_found:
                                        break
                                if double_bonded_C3_to_C7_sequence_found:
                                    break
                            if double_bonded_C3_to_C7_sequence_found:
                                break
                        if double_bonded_C3_to_C7_sequence_found:
                            break
                    if double_bonded_C3_to_C7_sequence_found:
                        break
                if double_bonded_C3_to_C7_sequence_found:
                    break
        if double_bonded_C3_to_C7_sequence_found:
            break
    return double_bonded_C3_to_C7_sequence_found

# Assess local environment of each H site to determine if Reaction 5 could occur at each site   
def is_valid_for_reaction5(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if C1 is None or C1.index in double_bonded_C_atoms or any(atom.name.startswith('Cl') for atom in C1.bonded_atoms):
        return False
    C2_atoms = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')]
    if len(C2_atoms) != 2:
        return False
    for C2 in C2_atoms:
        if C2.index in double_bonded_C_atoms and C2 != C1:
            C3_atoms = [atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C2 and atom != C1]
            for C3 in C3_atoms:
                if C3.index in double_bonded_C_atoms:
                    C4_atoms = [atom for atom in C3.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C3 and atom != C2 and atom != C1]
                    for C4 in C4_atoms:
                        if C4.index in double_bonded_C_atoms:
                            C5_atoms = [atom for atom in C4.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                            for C5 in C5_atoms:
                                if C5.index in double_bonded_C_atoms:
                                    C6_atoms = [atom for atom in C5.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C5 and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                                    for C6 in C6_atoms:
                                        if C6.index not in double_bonded_C_atoms:
                                            return True
    return False

# Assess local environment of each H site to determine if Reaction 6 could occur at each site       
def is_valid_for_reaction6(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C_bonded_to_H = [atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')]
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if not C1 or not any(atom.name.startswith('Cl') for atom in C1.bonded_atoms):
        return False
    if not C_bonded_to_H:
        return False
    C1 = C_bonded_to_H[0]
    if C1.index in double_bonded_C_atoms:
        for C2 in C1.bonded_atoms:
            if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1 and C2.index in double_bonded_C_atoms:
                return True
    valid_chain = False            
    for C2 in C1.bonded_atoms:
        if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1 and C2.index in double_bonded_C_atoms:
            for C3 in C2.bonded_atoms:
                if C3.name.startswith('C') and not C3.name.startswith('Cl') and C3 != C2 and C3 != C1 and C3.index not in double_bonded_C_atoms:
                    for C4 in C3.bonded_atoms:
                        if C4.name.startswith('C') and not C4.name.startswith('Cl') and C4 != C3 and C4 != C2 and C4 != C1 and C4.index not in double_bonded_C_atoms:
                            for C5 in C4.bonded_atoms:
                                if C5.name.startswith('C') and not C5.name.startswith('Cl') and C5 != C4 and C5 != C3 and C5 != C2 and C5 != C1 and C5.index not in double_bonded_C_atoms:
                                    valid_chain = True
                                    break
                            if valid_chain:
                                break
                    if valid_chain:
                        break
            if valid_chain:
                break
    if not valid_chain:
        return False
    return True

# Assess local environment of each H site to determine if Reaction 7 could occur at each site   
def is_valid_for_reaction7(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if not C1 or C1.index in double_bonded_C_atoms:
        return False
    C2_atoms = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C1]
    if len(C2_atoms) != 2 or not all(C2.index in double_bonded_C_atoms for C2 in C2_atoms):
        return False
    for C2 in C2_atoms:
        C3_atoms = [atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C2 and atom != C1]
        if len(C3_atoms) != 2:
            return False
        C3_with_Cl = any(atom.name.startswith('Cl') for atom in C3_atoms[0].bonded_atoms) or any(atom.name.startswith('Cl') for atom in C3_atoms[1].bonded_atoms)
        C3_double_bonded = C3_atoms[0].index in double_bonded_C_atoms or C3_atoms[1].index in double_bonded_C_atoms
        if not (C3_with_Cl and C3_double_bonded):
            return False
        for C3 in C3_atoms:
            C4_atoms = [atom for atom in C3.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C3 and atom != C2 and atom != C1]
            for C4 in C4_atoms:
                if C4.index in double_bonded_C_atoms:
                    return False
    return True

# Assess local environment of each H site to determine if Reaction 8 could occur at each site   
def is_valid_for_reaction8(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if not C1 or not C1.index in double_bonded_C_atoms:
        return False
    C2_atoms = [atom for atom in C1.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C1]
    if not all(C2.index in double_bonded_C_atoms for C2 in C2_atoms):
        return False
    for C2 in C2_atoms:
        C3_atoms = [atom for atom in C2.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C2 and atom != C1]
        if not all(C3.index in double_bonded_C_atoms for C3 in C3_atoms):
            return False
        for C3 in C3_atoms:
            C4_atoms = [atom for atom in C3.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C3 and atom != C2 and atom != C1]
            if not all(C4.index in double_bonded_C_atoms for C4 in C4_atoms):
                return False
            for C4 in C4_atoms:
                C5_atoms = [atom for atom in C4.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom != C4 and atom != C3 and atom != C2 and atom != C1]
                if any(C5.index in double_bonded_C_atoms for C5 in C5_atoms):
                    return False
    return True

# Assess local environment of each H site to determine if Reaction 9 could occur at each site   
def is_valid_for_reaction9(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C_bonded_to_H = [atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')]
    if not C_bonded_to_H:
        return False
    C_bonded_to_H = C_bonded_to_H[0]
    if C_bonded_to_H.index in double_bonded_C_atoms:
        return False
    if any(atom.name.startswith('Cl') for atom in C_bonded_to_H.bonded_atoms):
        return False
    current_C = C_bonded_to_H
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    def check_no_double_bonded_chain(C, depth):
        if depth > 5:
            return True
        for next_C in C.bonded_atoms:
            if next_C.name.startswith('C') and not next_C.name.startswith('Cl') and next_C != C and next_C.index not in double_bonded_C_atoms:
                if not check_no_double_bonded_chain(next_C, depth + 1):
                    return False
        return True
    return check_no_double_bonded_chain(C1, 1)
    valid_chain = False
    for C2 in C1.bonded_atoms:
        if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1 and C2.index not in double_bonded_C_atoms:
            for C3 in C2.bonded_atoms:
                if C3.name.startswith('C') and not C3.name.startswith('Cl') and C3 != C2 and C3 != C1 and C3.index not in double_bonded_C_atoms:
                    for C4 in C3.bonded_atoms:
                        if C4.name.startswith('C') and not C4.name.startswith('Cl') and C4 != C3 and C4 !=C2 and C4 != C1 and C4.index not in double_bonded_C_atoms:
                            for C5 in C4.bonded_atoms:
                                if C5.name.startswith('C') and not C5.name.startswith('Cl') and C5 != C4 and C5 != C3 and C5 != C2 and C5 != C1 and C5.index not in double_bonded_C_atoms:
                                    for C6 in C5.bonded_atoms:
                                        if C6.name.startswith('C') and not C6.name.startswith('Cl') and C6 != C5 and C6 != C4 and C6 != C3 and C6 != C2 and C6 != C1 and C6.index not in double_bonded_C_atoms:
                                            valid_chain = True
                                            break
                                    if valid_chain:
                                        break
                            if valid_chain:
                                break
                    if valid_chain:
                        break
            if valid_chain:
                break
    if not valid_chain:
        return False
    for _ in range(4):
        next_Cs = [atom for atom in current_C.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl') and atom.index != H_atom.index]
        if not next_Cs:
            return False
        next_C = next_Cs[0]
        if next_C.index in double_bonded_C_atoms:
            return False
        current_C = next_C
    return True

# Assess local environment of each H site to determine if Reaction 10 could occur at each site   
def is_valid_for_reaction10(H_atom, OH_bonded_to_C_atoms):
    if H_atom.index in selected_H_atoms:
        return False
    C_bonded_to_H = [atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')]
    if not C_bonded_to_H:
        return False
    C1 = C_bonded_to_H[0]
    if C1.index in double_bonded_C_atoms or any(atom.name.startswith('Cl') for atom in C1.bonded_atoms):
        return False
    for C2 in C1.bonded_atoms:
        if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1:
            if any((OH_atom.index, C2.index) in OH_bonded_to_C_atoms for OH_atom in C2.bonded_atoms):
                for other_C2 in C1.bonded_atoms:
                    if other_C2.name.startswith('C') and not other_C2.name.startswith('Cl') and other_C2 != C2 and any(atom.name.startswith('Cl') for atom in other_C2.bonded_atoms):
                        return True
    return False

# Assess local environment of each H site to determine if Reaction 11 could occur at each site   
def is_valid_for_reaction11(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C_bonded_to_H = [atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')]
    if not C_bonded_to_H:
        return False
    C1 = C_bonded_to_H[0]
    if C1.index not in double_bonded_C_atoms and any(atom.name.startswith('Cl') for atom in C1.bonded_atoms):
        for C2 in C1.bonded_atoms:
            if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1 and C2.index in double_bonded_C_atoms:
                return True
    return False

# Assess local environment of each H site to determine if Reaction 12 could occur at each site   
def is_valid_for_reaction12(H_atom):
    if H_atom.index in selected_H_atoms:
        return False
    C1 = next((atom for atom in H_atom.bonded_atoms if atom.name.startswith('C') and not atom.name.startswith('Cl')), None)
    if not C1 or not any(atom.name.startswith('Cl') for atom in C1.bonded_atoms):
        return False
    valid_chain = False
    for C2 in C1.bonded_atoms:
        if C2.name.startswith('C') and not C2.name.startswith('Cl') and C2 != C1 and C2.index in double_bonded_C_atoms:
            for C3 in C2.bonded_atoms:
                if C3.name.startswith('C') and not C3.name.startswith('Cl') and C3 != C2 and C3 != C1 and C3.index in double_bonded_C_atoms:
                    for C4 in C3.bonded_atoms:
                        if C4.name.startswith('C') and not C4.name.startswith('Cl') and C4 != C3 and C4 != C2 and C4.index in double_bonded_C_atoms:
                            for C5 in C4.bonded_atoms:
                                if C5.name.startswith('C') and not C5.name.startswith('Cl') and C5 != C4 and C5 != C3 and C5.index in double_bonded_C_atoms:
                                    for C6 in C5.bonded_atoms:
                                        if C6.name.startswith('C') and not C6.name.startswith('Cl') and C6 != C5 and C6 != C4 and C6 != C3 and C6.index not in double_bonded_C_atoms:
                                            for C7 in C6.bonded_atoms:
                                                if C7.name.startswith('C') and not C7.name.startswith('Cl') and C7 != C6 and C7 != C5 and C7 != C4 and C7.index not in double_bonded_C_atoms:
                                                    valid_chain = True
                                                    break
                                            if valid_chain:
                                                break
                                    if valid_chain:
                                        break
                            if valid_chain:
                                break
                    if valid_chain:
                        break
            if valid_chain:
                break
    if not valid_chain:
        return False
    else:
        return True

def custom_pvc_sort(resname):
    """
    Custom sort function for PVC residue names.
    Sorts alphabetically first, then numerically for names starting with 'PV'.
    """
    if resname.startswith("PV") and resname[2:].isdigit():
        return (1, int(resname[2:]))
    else:
        return (0, resname)

# Update the structure of the system (.gro file)
def write_gro_file(u, filename1, reacted_residue_name, reacted_residue_number, reacted_OH_atoms, hydrogen_bonded_to_OH_atoms, selected_H_atoms, target_Na_atoms, reacted_Cl_atoms, OH_bonded_to_C_atoms, hydrogen_bonded_to_OH_bonded_to_C_atoms):
    u = mda.Universe('npt.gro')
    atoms = u.atoms

    def adjust_indices_for_mdanalysis(indices):
        return [u.atoms[i] for i in indices]

    def adjust_indices_for_gro(atom_index):
        return atom_index
      

    reacted_OH_atoms = adjust_indices_for_mdanalysis(reacted_OH_atoms)
    hydrogen_bonded_to_OH_atoms = adjust_indices_for_mdanalysis(hydrogen_bonded_to_OH_atoms)
    selected_H_atoms = adjust_indices_for_mdanalysis(selected_H_atoms)
    target_Na_atoms = adjust_indices_for_mdanalysis(target_Na_atoms)
    reacted_Cl_atoms = adjust_indices_for_mdanalysis(reacted_Cl_atoms)
    OH_bonded_to_C_atoms = [(u.atoms[i[0] - 1], u.atoms[i[1] - 1]) for i in OH_bonded_to_C_atoms]

    selected_H_indices = {atom.index + 1 for atom in selected_H_atoms}
    target_Na_indices = {atom.index + 1 for atom in target_Na_atoms}
    reacted_Cl_indices = {atom.index + 1 for atom in reacted_Cl_atoms}
    hydrogen_bonded_to_OH_indices = {atom.index + 1 for atom in hydrogen_bonded_to_OH_atoms}
    reacted_OH_indices = {atom.index + 1 for atom in reacted_OH_atoms}
    
    reacted_atom_indices = set()
    for index_set in [reacted_OH_indices, hydrogen_bonded_to_OH_indices, selected_H_indices, target_Na_indices, reacted_Cl_indices]:
        reacted_atom_indices.update(index_set)

    reacted_atom_indices_list = list(reacted_atom_indices)
    reacted_atoms = u.atoms[reacted_atom_indices_list]

    existing_PV_residues = [res.resname for res in u.residues if res.resname.startswith("PV") and res.resname[2:].isdigit()]
    existing_PV_residues.sort(key=lambda x: int(x[2:]))
    if existing_PV_residues:
        last_residue_number = int(existing_PV_residues[-1][2:])
        residue_rename_counter = last_residue_number + 1
    else:
        residue_rename_counter = 1

    selected_H_resname = selected_H_atoms[0].resname if selected_H_atoms else None
    
    PVC_atoms = u.select_atoms("resname PV*")
    unreacted_atoms = u.select_atoms("not bynum {} and not (resname {} and resid {})".format(' '.join(map(str, reacted_atom_indices_list)), reacted_residue_name, reacted_residue_number))
    pv_atoms = unreacted_atoms.select_atoms("resname PV*")
    residue_names = [atom.residue.resname for atom in pv_atoms]

    unique_residue_names = list(set(residue_names))
    unique_residue_names.sort(key=custom_pvc_sort)

    nas_atoms = unreacted_atoms.select_atoms("resname NAS")
    oh_atoms = unreacted_atoms.select_atoms("resname OH")
    nac_atoms = unreacted_atoms.select_atoms("resname NAC")
    sol_atoms = unreacted_atoms.select_atoms("resname SOL")   
    residue_atoms = PVC_atoms.select_atoms("resname {} and resid {} and not bynum {}".format(reacted_residue_name, reacted_residue_number, ' '.join(map(str, reacted_atom_indices_list))))
    unreacted_Na_OH_atoms = u.select_atoms("resname NAS or resname OH").select_atoms("not bynum {}".format(' '.join(map(str, reacted_atom_indices_list))))
    
    with open(filename1, 'w') as f:
        f.write("Generated by MDAnalysis\n")

        total_atoms = len(unreacted_atoms) + len(residue_atoms) + len(reacted_atom_indices_list)
        f.write(f"{total_atoms}\n")

        atom_counter = 1

        for resname in unique_residue_names:
            current_residue_atoms = unreacted_atoms.select_atoms(f"resname {resname}")
        
            for atom in current_residue_atoms:
                if atom_counter > 99999:
                    atom_counter = 0
                gro_index = adjust_indices_for_gro(atom.index)
                formatted_line = f"{atom.resid:5d}{atom.resname:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
                f.write(formatted_line)
                atom_counter += 1
                
        for atom in residue_atoms:
            if atom.resname.startswith("PV") and atom.resname[2:].isdigit():
                residue_name = atom.resname
            elif selected_H_resname and selected_H_resname.startswith("PV") and selected_H_resname[2:].isdigit() and atom.index in reacted_atom_indices:
                residue_name = selected_H_resname
            else:
                residue_name = f"PV{residue_rename_counter}"
                new_residue_names.add(residue_name)
            gro_index = adjust_indices_for_gro(atom.index)
            formatted_line = f"{atom.resid:5d}{residue_name:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
            f.write(formatted_line)
            atom_counter += 1

        for atom in nac_atoms:
            if atom_counter > 99999:
                atom_counter = 0
            gro_index = adjust_indices_for_gro(atom.index)
            formatted_line = f"{atom.resid:5d}{atom.resname:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
            f.write(formatted_line)
            atom_counter += 1
        
        for atom in target_Na_atoms + reacted_Cl_atoms:
            if atom in target_Na_atoms:
                reacted_Cl_atom = reacted_Cl_atoms[0]
                dist_vector = distance_array(atom.position, reacted_Cl_atom.position, box=u.dimensions)[0]
                new_Na_position = reacted_Cl_atom.position + 1.45 * dist_vector / np.linalg.norm(dist_vector)
                atom.position = new_Na_position
            if atom_counter > 99999:
                atom_counter = 0
            gro_index = adjust_indices_for_gro(atom.index)
            formatted_line = f"{atom.resid:5d}{atom.resname:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
            f.write(formatted_line)
            atom_counter += 1

        for atom in sol_atoms:
            if atom_counter > 99999:
                atom_counter = 0
            gro_index = adjust_indices_for_gro(atom.index)
            formatted_line = f"{atom.resid:5d}{atom.resname:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
            f.write(formatted_line)
            atom_counter += 1
            
        for atom in reacted_OH_atoms + hydrogen_bonded_to_OH_atoms + selected_H_atoms:
            if atom in selected_H_atoms:
                reacted_OH_atom = reacted_OH_atoms[0]
                dist_vector = distance_array(atom.position, reacted_OH_atom.position, box=u.dimensions)[0]
                new_H_position = reacted_OH_atom.position + 0.58 * dist_vector / np.linalg.norm(dist_vector)
                atom.position = new_H_position
            if atom_counter > 99999:
                atom_counter = 0
            gro_index = adjust_indices_for_gro(atom.index)
            formatted_line = f"{atom.resid:5d}{atom.resname:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
            f.write(formatted_line)
            atom_counter += 1

        for atom in nas_atoms + oh_atoms:
            if atom_counter > 99999:
                atom_counter = 0
            gro_index = adjust_indices_for_gro(atom.index)
            formatted_line = f"{atom.resid:5d}{atom.resname:<5}{atom.name:>5}{gro_index:5d}{atom.position[0]/10:8.3f}{atom.position[1]/10:8.3f}{atom.position[2]/10:8.3f}\n"
            f.write(formatted_line)
            atom_counter += 1
            
        box_dimensions = u.dimensions[:3] / 10
        f.write(f"{box_dimensions[0]:10.1f}{box_dimensions[1]:10.1f}{box_dimensions[2]:10.1f}\n")

# Update the topology of the reacted PVC chain 
def update_topology(selected_H_atoms, reacted_Cl_atoms, double_bonded_C_atoms, itp_filepath, u):
    new_residue_name = os.path.basename(itp_filepath).split('.')[0].upper()
    selected_H_names = {u.atoms[i].name for i in selected_H_atoms}
    reacted_Cl_names = {u.atoms[i].name for i in reacted_Cl_atoms}
    double_bonded_C_names = {u.atoms[i].name for i in double_bonded_C_atoms}

    itp_atom_numbers = {}
    double_bonded_C_itp_numbers = {}

    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    new_lines = []
    atom_mapping = {}
    new_atom_id = 1
    in_section = None

    for line in lines:
        parts = line.split()
        if not parts or parts[0].startswith(';'):
            new_lines.append(line)
            continue

        if '[' in line and ']' in line:
            in_section = parts[1]
            new_lines.append(line)
            continue

        if in_section == 'moleculetype':
            if len(parts) >= 2 and parts[1].isdigit():
                parts[0] = new_residue_name
            new_lines.append(' '.join(parts) + '\n')
            continue

        if in_section == 'atoms':
            atom_id = int(parts[0])
            atom_name = parts[4]

            if atom_name in double_bonded_C_names:
                double_bonded_C_itp_numbers[atom_name] = atom_id

            if atom_name in selected_H_names or atom_name in reacted_Cl_names:
                itp_atom_numbers[atom_name] = atom_id
                continue

            parts[3] = new_residue_name
            atom_mapping[atom_id] = new_atom_id
            parts[0] = str(new_atom_id)
            new_atom_id += 1
            new_lines.append(' '.join(parts) + '\n')

        elif in_section in ['bonds', 'angles', 'dihedrals', 'pairs']:
            num_atoms = 2 if in_section == 'bonds' or in_section == 'pairs' else 3 if in_section == 'angles' else 4
            atom_ids = [int(part) for part in parts[:num_atoms]]

            if any(itp_atom_numbers.get(atom_name) in atom_ids for atom_name in selected_H_names.union(reacted_Cl_names)):
                continue

            updated_atom_ids = [str(atom_mapping.get(atom_id, atom_id)) for atom_id in atom_ids]
            parts[:num_atoms] = updated_atom_ids

            if any(atom_id in double_bonded_C_itp_numbers.values() for atom_id in atom_ids):
                if in_section == 'bonds':
                    parts[3], parts[4] = '0.1340', '459403.2'
                elif in_section == 'angles':
                    parts[4], parts[5] = '120.000', '585.76'
                elif in_section == 'dihedrals':
                    parts[5:] = ['58.576', '0.000', '-58.576', '0.000', '0.000', '0.000']

            new_lines.append(' '.join(parts) + '\n')

    with open(itp_filepath, 'w') as file:
        file.writelines(new_lines)

    return itp_filepath

# Remove any duplicates of ITP numbers on lines in ITP files
def remove_duplicates_from_topology_sections(itp_filepath):
    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    new_lines = []
    in_relevant_section = False

    for line in lines:
        if in_relevant_section and not line.startswith(';'):
            parts = line.split()
            if len(parts) > 1 and len(set(parts[:4])) != len(parts[:4]):
                continue
        new_lines.append(line)

        if line.startswith('[ bonds ]') or line.startswith('[ angles ]') or line.startswith('[ dihedrals ]') or line.startswith('[ pairs ]'):
            in_relevant_section = True
        elif line.startswith('[') and in_relevant_section:
            in_relevant_section = False

    with open(itp_filepath, 'w') as file:
        file.writelines(new_lines)

# Prepare files for MD simulation execution
def run_amber_and_process_files(initial_NAS, total_reacted_OH, run_number, reacted_residue_names, new_residue_names):

    atomtypes_section = []
    in_atomtypes = False
    new_lines = []
    remove_defaults = False

    base_dir = os.path.abspath(os.getcwd())
    md_dir = os.path.join(base_dir, f"md{run_number}")
    topology_dir = os.path.join(md_dir, 'topology')
    if not os.path.exists(topology_dir):
        os.makedirs(topology_dir)

    base_itp_files = ['na.itp', 'o.itp', 'thf.itp']
    for file_name in base_itp_files:
        shutil.copyfile(os.path.join(base_dir, file_name), os.path.join(topology_dir, file_name))

    for reacted_residue in reacted_residue_names:
        base_residue = reacted_residue
        base_itp_path = os.path.join(base_dir, f'{base_residue.lower()}.itp')

    if new_residue_names:
        for new_residue in new_residue_names:
            new_itp = new_residue
            itp_filename = f'{new_itp.lower()}.itp'
            updated_itp_path = os.path.join(topology_dir, itp_filename)
    else:
        selected_residue_itp = selected_H.resname.lower() + '.itp'
        updated_itp_path = os.path.join(topology_dir, selected_residue_itp)

    for itp_file in glob.glob(os.path.join(base_dir, '*.itp')):
        shutil.copy(itp_file, topology_dir)
 
        shutil.copyfile(base_itp_path, updated_itp_path)
        update_topology(selected_H_atoms, reacted_Cl_atoms, double_bonded_C_atoms, updated_itp_path, u)
        modify_itp_charges(updated_itp_path, double_bonded_C_atoms, OH_bonded_to_C_atoms, hydrogen_bonded_to_OH_bonded_to_C_atoms, u)

    with open(os.path.join(md_dir, 'topology', 'forcefield.itp'), 'a') as f:
        f.write("#include \"oplsaa.ff/forcefield.itp\"\n\n")

    # Modify the [atomtypes] section depending on molecules in your system        
    with open(os.path.join(md_dir, 'topology', 'forcefield.itp'), 'a') as f:
        f.write("[ atomtypes ]\n")
        f.write("ppa_AAA    C0    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppa_AAB    H1     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppa_AAC   Cl2    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppa_AAD   Cl3    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppa_AAE    C4    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppa_AAF    H5     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppb_AAA    H0     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppb_AAB    C1    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppb_AAC   Cl2    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppb_AAD    C3    12.0110     0.000    A    3.45000E-01   3.47272E-01\n")
        f.write("ppb_AAE    H4     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppb_AAF    C5    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppb_AAG   Cl6    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppc_AAA    H0     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppc_AAB    C1    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppc_AAC   Cl2    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppc_AAD    C3    12.0110     0.000    A    3.45000E-01   3.47272E-01\n")
        f.write("ppc_AAE    H4     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppc_AAF    C5    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppc_AAG   Cl6    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppd_AAA    H0     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("ppd_AAB    C1    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppd_AAC   Cl2    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppd_AAD   Cl3    35.4500     0.000    A    3.40000E-01   1.25520E+00\n")
        f.write("ppd_AAE    C4    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("ppd_AAF    H5     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("nopls_803  Na    23.000    0.000    A    0.221737   1.47\n")
        f.write("hopls_800  Oh    15.9999   0.000    A    0.365   0.251\n")
        f.write("hopls_801  Ho   1.0080    0.000    A    0.1443   0.183989\n")
        f.write("copls_801  Na   23.000    0.000    A    0.252   0.123\n")
        f.write("copls_802  Cl   35.450    0.000    A    0.385   1.352\n")
        f.write("sopls_809  H809     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_808  H808     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_800  C800    12.0110     0.000    A    3.45000E-01   3.47272E-01\n")
        f.write("sopls_802  H802     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_811  H811     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_812  H812     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_805  H805     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_803  C803    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("sopls_804  H804     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("sopls_806  O806    15.9990     0.000    A    2.90000E-01   5.85760E-01\n")
        f.write("sopls_807  C807    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("sopls_810  C810    12.0110     0.000    A    3.50000E-01   2.76144E-01\n")
        f.write("sopls_801  H801     1.0080     0.000    A    2.50000E-01   1.25520E-01\n")
        f.write("\n")
        f.write("[ nonbond_params ]\n")
        f.write("nopls_803    hopls_800    1 0.238725   0.7934\n")
        f.write("copls_801    copls_802    1 0.31850   0.4078\n")
        f.write("\n")

    NAS = initial_NAS - total_reacted_OH
    OH = NAS
    H2O = total_reacted_OH
    initial_NAS = NAS
    
    def custom_pvc_sort(residue_name):
        match = re.match(r"PV(\d+)", residue_name)
        if match:
            return (1, int(match.group(1)))
        else:
            return (0, residue_name)    
            
    pvc_residue_names = [os.path.splitext(os.path.basename(itp_file))[0].upper() for itp_file in glob.glob(os.path.join(topology_dir, 'pv*.itp'))]
    pvc_residue_names.sort(key=custom_pvc_sort)
    
    if updated_itp_path:
        updated_residue_name = os.path.splitext(os.path.basename(updated_itp_path))[0].upper()
        if updated_residue_name in pvc_residue_names:
            pvc_residue_names.remove(updated_residue_name)
            pvc_residue_names.append(updated_residue_name)
            
    with open(os.path.join(md_dir, 'topology', 'forcefield.itp'), 'a') as f:
    
        pv_itp_files = glob.glob(os.path.join(md_dir, 'topology', 'pv*.itp'))
        for pv_file in pv_itp_files:
            pv_filename = os.path.basename(pv_file)
            f.write(f"#include \"{pv_filename}\"\n")
            pv_residue_name = os.path.basename(pv_file).split('.')[0].upper()
            if pv_residue_name not in initial_PVC_counts:
                initial_PVC_counts[pv_residue_name] = 1
                print("PV* counts:", initial_PVC_counts)
        
        f.write("#include \"nacl.itp\"\n")

        f.write("#include \"oplsaa.ff/spce.itp\"\n")

        for file_name in base_itp_files:
            f.write(f"#include \"{file_name.lower()}\"\n")
            
        f.write("\n")
        f.write("[ system ]\n")
        f.write("SYS\n")
        f.write("\n")
        f.write("[ molecules ]\n")

        for residue_name in pvc_residue_names:
            if residue_name in initial_PVC_counts and initial_PVC_counts[residue_name] > 0:
                f.write(f"{residue_name}     {initial_PVC_counts[residue_name]}\n")
                
        f.write(f"NAC     {total_reacted_OH}\n")
        f.write(f"SOL     {total_reacted_OH}\n")
        f.write(f"NAS     {NAS}\n")
        f.write(f"OH      {OH}\n")

        
    min_dir = os.path.join(md_dir, 'min')
    if not os.path.exists(min_dir):
       os.mkdir(min_dir)
    nvt_dir = os.path.join(md_dir, 'nvt')
    if not os.path.exists(nvt_dir):
        os.mkdir(nvt_dir)
    npt_dir = os.path.join(md_dir, 'npt')
    if not os.path.exists(npt_dir):
        os.mkdir(npt_dir)
        
    subprocess_steps = [
        ("Setting up topology and preparing minimization", [
            ['gmx', 'editconf', '-f', os.path.join(base_dir, f'output_{iteration-1}.gro'), '-o', os.path.join(min_dir, 'initial.gro')],
            ['cp', os.path.join(base_dir, 'min1.mdp'), os.path.join(min_dir, 'min1.mdp')],
            ['cp', os.path.join(base_dir, 'min2.mdp'), os.path.join(min_dir, 'min2.mdp')],
            ['cp', os.path.join(base_dir, 'min.top'), os.path.join(min_dir, 'min.top')]
        ]),
        ("Running first energy minimization and NVT equilibration", [
            ['gmx', 'grompp', '-f', os.path.join(min_dir, 'min1.mdp'), '-c', os.path.join(min_dir, 'initial.gro'), '-p', os.path.join(min_dir, 'min.top'), '-o', os.path.join(min_dir, 'em'), '-maxwarn', '4'],
            ['gmx', 'mdrun', '-v', '-deffnm', os.path.join(min_dir, 'em')],
            ['cp', os.path.join(min_dir, 'em.gro'), os.path.join(nvt_dir, 'nvt.gro')],
            ['cp', os.path.join(base_dir, 'nvt.top'), os.path.join(nvt_dir, 'nvt.top')],
            ['cp', os.path.join(base_dir, 'nvt.mdp'), os.path.join(nvt_dir, 'nvt.mdp')],
            ['gmx', 'grompp', '-f', os.path.join(nvt_dir, 'nvt.mdp'), '-c', os.path.join(nvt_dir, 'nvt.gro'), '-p', os.path.join(nvt_dir, 'nvt.top'), '-o', os.path.join(nvt_dir, 'nvt'), '-maxwarn', '4'],
            ['gmx', 'mdrun', '-v', '-deffnm', os.path.join(nvt_dir, 'nvt'), '-pme', 'gpu', '-update', 'gpu', '-bonded', 'gpu', '-nb', 'gpu', '-ntmpi', '1', '-ntomp', '16']
        ]),
        ("Running second energy minimization and NPT equilibration", [
            ['gmx', 'grompp', '-f', os.path.join(min_dir, 'min2.mdp'), '-c', os.path.join(nvt_dir, 'nvt.gro'), '-p', os.path.join(min_dir, 'min.top'), '-o', os.path.join(min_dir, 'min'), '-maxwarn', '4'],
            ['gmx', 'mdrun', '-v', '-deffnm', os.path.join(min_dir, 'min')],
            ['cp', os.path.join(base_dir, 'npt.top'), os.path.join(npt_dir, 'npt.top')],
            ['cp', os.path.join(base_dir, 'npt.mdp'), os.path.join(npt_dir, 'npt.mdp')],
            ['cp', os.path.join(min_dir, 'min.gro'), os.path.join(npt_dir, 'npt.gro')],
            ['gmx', 'grompp', '-f', os.path.join(npt_dir, 'npt.mdp'), '-c', os.path.join(npt_dir, 'npt.gro'), '-p', os.path.join(npt_dir, 'npt.top'), '-o', os.path.join(npt_dir, 'npt'), '-maxwarn', '4'],
            ['gmx', 'mdrun', '-v', '-deffnm', os.path.join(npt_dir, 'npt'), '-pme', 'gpu', '-update', 'gpu', '-bonded', 'gpu', '-nb', 'gpu', '-ntmpi', '1', '-ntomp', '16']
        ]),
        ("Finalizing MD simulation", [
            ['gmx', 'trjconv', '-f', os.path.join(npt_dir, 'npt.trr'), '-s', os.path.join(npt_dir, 'npt.tpr'), '-o', os.path.join(npt_dir, 'npt.gro'), '-pbc', 'mol', '-dump', '1000', '-input', '0'],
            ['cp', os.path.join(npt_dir, 'npt.gro'), os.path.join(base_dir, 'npt.gro')],
            ['cp', os.path.join(topology_dir, 'nacl.itp'), os.path.join(base_dir, 'nacl.itp')],
            ['cp', '-r', os.path.join(topology_dir, 'pv*.itp'), base_dir]
        ])
    ]

    for step_desc, commands in subprocess_steps:
        t = threading.Thread(target=run_spinner, args=(step_desc,))
        t.start()
        for command in commands:
            if 'gmx trjconv' in command[0]:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=False)
                output, error = process.communicate(input=trjconv_input)
            else:
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t.join()
            
    itp_files = glob.glob(os.path.join(topology_dir, 'pv*.itp'))
    for itp_file in itp_files:
        shutil.copy(itp_file, base_dir)

# Modify charges in the ITP file of the reacted PVC residue
def modify_itp_charges(itp_filepath, double_bonded_C_atoms, OH_bonded_to_C_atoms, hydrogen_bonded_to_OH_bonded_to_C_atoms, u):
    modified_lines = []
    atoms_section = False
    total_charge = 0
    hydrogen_indices = []

    hydrogen_itp_numbers = {}
    carbon_itp_numbers = {}
    chlorine_itp_numbers = {}
    double_bonded_C_itp_numbers = {}

    double_bonded_C_names = {u.atoms[i].name for i in double_bonded_C_atoms}

    with open(itp_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if '[ atoms ]' in line:
            atoms_section = True
            modified_lines.append(line)
        elif '[' in line and ']' in line:
            atoms_section = False
            modified_lines.append(line)
        elif atoms_section and line.strip() == '':
            atoms_section = False
            modified_lines.append(line)
        elif not atoms_section:
            modified_lines.append(line)        
        elif atoms_section and line.strip() and not line.startswith(';'):
            parts = line.split()
            atom_id = int(parts[0])
            atom_name = parts[4]
            charge = calculate_charge(atom_id, atom_name, u, double_bonded_C_itp_numbers, hydrogen_itp_numbers, carbon_itp_numbers, chlorine_itp_numbers, OH_bonded_to_C_atoms, hydrogen_bonded_to_OH_bonded_to_C_atoms)
            parts[6] = str(charge)
            total_charge += charge
            if atom_name.startswith('H'):
                hydrogen_indices.append((atom_id, len(modified_lines)))
            if atom_name in double_bonded_C_names:
                double_bonded_C_itp_numbers[atom_name] = atom_id
            elif atom_name.startswith('H'):
                hydrogen_itp_numbers[atom_name] = atom_id
            elif atom_name.startswith('C') and not atom_name.startswith('Cl'):
                carbon_itp_numbers[atom_name] = atom_id
            elif atom_name.startswith('Cl'):
                chlorine_itp_numbers[atom_name] = atom_id
            modified_lines.append(' '.join(parts) + '\n')

    apply_charge_correction(modified_lines, total_charge, hydrogen_indices)
    with open(itp_filepath, 'w') as f:
        f.writelines(modified_lines)

# Calculate charge for atoms
def calculate_charge(atom_id, atom_name, u, double_bonded_C_itp_numbers, hydrogen_itp_numbers, carbon_itp_numbers, chlorine_itp_numbers, OH_bonded_to_C_atoms, hydrogen_bonded_to_OH_bonded_to_C_atoms):
    if atom_name in double_bonded_C_itp_numbers.keys():
        return -0.186  
    elif atom_name in hydrogen_itp_numbers.keys():
        return calculate_hydrogen_charge(atom_id, u, hydrogen_itp_numbers, hydrogen_bonded_to_OH_bonded_to_C_atoms)
    elif atom_name in carbon_itp_numbers.keys():
        return calculate_carbon_charge(atom_id, u, carbon_itp_numbers, double_bonded_C_itp_numbers)
    elif atom_name in chlorine_itp_numbers.keys():
        return -0.283
    else:
        return 0.0

# Assign charges on H atoms
def calculate_hydrogen_charge(atom_id, u, hydrogen_itp_numbers, hydrogen_bonded_to_OH_bonded_to_C_atoms):
    if atom_id in hydrogen_itp_numbers.values():
        return 0.12
    else:
        return 0.12

# Assign charges on C atoms
def calculate_carbon_charge(atom_id, u, carbon_itp_numbers, double_bonded_C_itp_numbers):
    if atom_id in double_bonded_C_itp_numbers.values():
        return -0.186  
    else:
        atom = u.atoms[atom_id - 1]
        bonded_hydrogens = [a for a in atom.bonded_atoms if a.name.startswith('H')]
        if len(bonded_hydrogens) == 3:
            return -0.253
        elif len(bonded_hydrogens) == 2:
            return 0.085
        else:
            return -0.135

# Distribute residual charges on H atoms
def apply_charge_correction(modified_lines, total_charge, hydrogen_indices):
    if abs(total_charge) > 0.001:
        charge_correction = total_charge / len(hydrogen_indices)
        for h_id, line_number in hydrogen_indices:
            parts = modified_lines[line_number].split()
            corrected_charge = float(parts[6]) - charge_correction
            parts[6] = str(corrected_charge)
            modified_lines[line_number] = ' '.join(parts) + '\n'

# Display on terminal
def display_initial_stats(u, base_dir):
    terminal_width = shutil.get_terminal_size().columns
    title = pyfiglet.figlet_format("PolySimKMC", font="small")
    centered_title = '\n'.join([line.center(terminal_width) for line in title.split('\n')])
    print(centered_title)
    subtext = "A Composite Kinetic Monte Carlo (KMC) - Molecular Dynamics (MD) Model for Analyzing PVC Dehydrochlorination Process (v1.0)"
    print(subtext.center(terminal_width))
    total_atoms = len(u.atoms)
    total_reactions = 12
    pvc_residues = u.select_atoms("resname PV*").residues
    pvc_chains = len(pvc_residues)
    total_chlorine_atoms = len(u.select_atoms("name Cl*"))
    print(f"Total No. of Reactions: {total_reactions}".center(terminal_width))
    print(f"Total No. of Atoms: {total_atoms}".center(terminal_width))
    print(f"No. of PVC Chains: {pvc_chains}".center(terminal_width))
    print(f"Total No. of Chlorine Atoms: {total_chlorine_atoms}".center(terminal_width))
    print(f"Total of KMC-MD Steps: {max_iterations}".center(terminal_width))

# Initiate parameters: initial_NAS - initial number of NaOH molecules, rate_thresold - Rate threshold to stop KMC-MD loop, 
# initial_PVC_counts - initial number of PVC chains (PVC - PVC40, PVD - PVC60, PVE - PVC100)    
initial_NAS = 567
total_time = 0
iteration = 0
max_iterations = 500
rate_threshold = 1e-6
total_reacted_OH = 0
run_number = 1
initial_PVC_counts = {'PVC':3, 'PVD': 6, 'PVE': 1}

output_file_path = "output.txt"
base_dir = '/path/to/your/base/directory'
u = mda.Universe('npt.gro')
display_initial_stats(u, base_dir)

with open(output_file_path, 'w') as file:
    file.write("Time(min), % DHC, Selected reaction number, Total rate(s-1), Number of valid rates of sites, Reacted PVC residue name\n")

while iteration < max_iterations:
    # Initialize lists for rates and event information
    rates = []
    event_info = []

    # Load the universe and select atoms
    u = mda.Universe('npt.gro')
    H_atoms = u.select_atoms("name H* and resname PV*")
    OH_atoms = u.select_atoms("name Oh and resname OH")
    PVC_atoms = u.select_atoms("resname PV*")
    base_dir = '/path/to/your/base/directory'
    itp_files = ['na.itp', 'o.itp', 'thf.itp']

    # Add PVC itp files from base directory        
    pv_itp_files = glob.glob(os.path.join(base_dir, 'pv*.itp'))
    for pv_file in pv_itp_files:
        pv_filename = os.path.basename(pv_file)
        itp_files.append(pv_filename)

    # Store bond info using itp files
    create_bonds(u, itp_files)

    print(f"Starting KMC step {iteration + 1}:")
    start_time = time.time()
    pbar = tqdm(total=len(H_atoms) * 12, desc="Checking valid sites", leave=False)
    
    valid_reactions_sites = {i: [] for i in range(1, 13)} 

    # Iterate over each H atom to check for valid reaction sites
    for idx, H_atom in enumerate(H_atoms):
        for reaction_number in range(1, 13):
            is_valid = False
            if reaction_number == 10:
                is_valid = is_valid_for_reaction10(H_atom, OH_bonded_to_C_atoms)
            else:
                is_valid = eval(f'is_valid_for_reaction{reaction_number}')(H_atom)
    
            if is_valid:
                valid_reactions_sites[reaction_number].append(idx)

            pbar.update(1)
    pbar.close()
    
    # Iterate over valid sites to calculate rates and store event info
    for reaction_number, valid_sites in valid_reactions_sites.items():
        for site_idx in valid_sites:
            H_atom = H_atoms[site_idx]
            dist = distance_array(H_atom.position, OH_atoms.positions)
            rate = calculate_rate(dist, reaction_number)
            valid_indices = np.where(rate > rate_threshold)
            rates.extend(rate[valid_indices])
            event_info.extend([(site_idx, reaction_number) for _ in valid_indices[0]])

    # Cummulative rate calculation
    cumulative_rates = np.cumsum(rates)
    len_rate = len(rates)
    if cumulative_rates.size == 0 or cumulative_rates[-1] < rate_threshold:
        print("No more possible reactions!")
        break
    else:

        # Selecting event based on Metropolis algorithm
        total_rate = cumulative_rates[-1]
        rand_u = np.random.random()
        selected_event_idx = min(np.searchsorted(cumulative_rates, rand_u * total_rate), len(event_info) - 1)
        selected_H_idx, selected_reaction = event_info[selected_event_idx]
    
        selected_H = H_atoms[selected_H_idx]
        if eval(f'is_valid_for_reaction{selected_reaction}')(selected_H):
            eval(f'reaction{selected_reaction}')(selected_H, u)
            H_states[selected_H_idx] = selected_reaction

        elapsed_time = time.time() - start_time
        print(f"Selected site: {selected_H.name} (Residue: {selected_H.resname}, Index: {selected_H_idx})")
        print(f"Time elapsed for this KMC step: {elapsed_time:.2f} seconds")
        
        # Update system clock
        delta_t = np.log(1 / rand_u) / total_rate
        total_time += delta_t

        print(f"KMC Step: {iteration + 1} / {max_iterations}, Time: {total_time}, Selected Reaction ID: {selected_reaction}")

    def is_dynamic_pvc_residue(resname):
        return bool(re.match(r'^PV\d+$', resname))

    if selected_reaction and selected_H:
        reaction_residue_name = selected_H.resname
        reacted_residue_names.add(reaction_residue_name)

    if not is_dynamic_pvc_residue(reaction_residue_name):
        if reaction_residue_name in initial_PVC_counts:
            initial_PVC_counts[reaction_residue_name] -= 1

    for new_residue_name in new_residue_names:
        if new_residue_name not in initial_PVC_counts:
            initial_PVC_counts[new_residue_name] = 1
        else:
            initial_PVC_counts[new_residue_name] += 1
        
    reaction_residue_name = selected_H.resname
    reaction_residue_number = selected_H.resid
    original_residue_name = selected_H.resname
    reacted_residue_names.add(reaction_residue_name)
    print("Updating new structure...")
    write_gro_file(u, f"output_{iteration}.gro", reaction_residue_name, reaction_residue_number, reacted_OH_atoms, hydrogen_bonded_to_OH_atoms, selected_H_atoms, target_Na_atoms, reacted_Cl_atoms, OH_bonded_to_C_atoms, hydrogen_bonded_to_OH_bonded_to_C_atoms)
    total_reacted_OH += len(reacted_OH_atoms)
    print("Structure updated.")
    
    # Calculate % Dehydrochlorination as (No.  of reacted Chlorine atoms)*100/(Total No.  of Chlorine atoms)
    percentage_dehydrochlorination = (iteration + 1) * 100 / 505
    time_in_minutes = total_time / 60

    with open(output_file_path, 'a') as file:
        file.write(f"{time_in_minutes} {percentage_dehydrochlorination} {selected_reaction} {total_rate} {len_rate} {reaction_residue_name}\n")

    iteration += 1

    # Run MD simulation after every KMC step
    if iteration % 1 == 0:
        run_amber_and_process_files(initial_NAS, total_reacted_OH, run_number, reacted_residue_names, new_residue_names)
        run_number += 1
        print(f"MD simulation for step {run_number - 1} completed.")
        u = mda.Universe('npt.gro')
        
        base_dir = '/data3/victor/kmc/kmc/expand/present'

        itp_files = ['na.itp', 'o.itp', 'nacl.itp', 'thf.itp']

        pv_itp_files = glob.glob(os.path.join(base_dir, 'pv*.itp'))
        for pv_file in pv_itp_files:
            pv_filename = os.path.basename(pv_file)
            itp_files.append(pv_filename)

        create_bonds(u, itp_files)

        PVC_types = ['PVC', 'PVD','PVE']
        H_atoms = u.select_atoms("name H* and resname PV*")
        PVC_atoms = u.select_atoms("resname PV*")

        double_bonded_C_atoms = set()
        for atom in PVC_atoms:
            if atom.name.startswith('C') and not any(a.name.startswith('Cl') for a in atom.bonded_atoms):
                bonded_hydrogens = [a for a in atom.bonded_atoms if a.name.startswith('H')]
                bonded_carbons = [a for a in atom.bonded_atoms if a.name.startswith('C') and not a.name.startswith('Cl')]

                if len(bonded_hydrogens) == 1:
                    double_bonded_C_atoms.add(atom.index)
                if len(bonded_hydrogens) == 2 and len(bonded_carbons) == 1:
                    bonded_carbon = bonded_carbons[0]
                    bonded_hydrogens_to_bonded_carbon = [a for a in bonded_carbon.bonded_atoms if a.name.startswith('H')]

                    if len(bonded_hydrogens_to_bonded_carbon) == 1:
                        double_bonded_C_atoms.add(atom.index)
        
        reacted_OH_atoms = set()
        reacted_Cl_atoms = set()
        hydrogen_bonded_to_OH_atoms = set()
        selected_H_atoms = set()
        target_Na_atoms = set()
        OH_bonded_to_C_atoms = set()
        hydrogen_bonded_to_OH_bonded_to_C_atoms = set()
        reacted_residue_names = set()
        new_residue_names = set()
        reacted_terminal_carbons = set()
        continue

print("KMC-MD simulation has been completed.")
