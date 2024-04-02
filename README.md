# KMC-MD
An integrated Off-Lattice Kinetic Monte Carlo (KMC)-Molecular Dynamics (MD) Framework for modeling PVC dehydrochlorination process

# Please note: 
The code was run on a GeForce RTX 3090 using a 16-core GPU with the following software installed
- GROMACS v. 2022.4 (https://www.gromacs.org/)
- Python3 3.10.12 (https://www.python.org/downloads/release/python-31012/)
- MDAnalysis package (https://www.mdanalysis.org/)

Please check lines 46-64 of KMC-MD/kmc-md/main.py file for the necessary Python packages to be installed.

# Instructions for Running KMC-MD Simulation:
This script was created for the execution of Kinetic Monte Carlo (KMC) coupled with Molecular Dynamics (MD) simulations, for studying the dehydrochlorination (DHC) process of PVC. 

Key Configuration Parameters:
1. PVC chains are differentiated by chain length using the following residue names:
   - PVA for N=5, PVB for N=20, PVC for N=40, PVD for N=60, PVE for N=100, PVF for N=120.
   Additional residue names include NAS (sodium), OH (hydroxide), NAC (sodium chloride), and SOL (water).

2. Simulation Temperature and Initial Structure file: 
  - Set the simulation temperature (in Kelvin) at line 66 and the initial structure (npt.gro) at line 174.

3. Reaction Rate Catalog: 
  - Edit reaction rates between lines 149-161.

4. Cutoff Radius for Rate Calculation: 
  - Adjust the cutoff radius (in Angstroms) for rate calculations at line 166.

5. Forcefield Atom Types: 
  - Modify the [atomtypes] section of the forcefield.itp file at line 1236 as required.

6. Initial NaOH Molecules: 
  - Edit the initial number of NaOH molecules at line 1518.

7. Iteration settings: 
   - Set the maximum number of iterations at line 1521.
   - Define the rate threshold (in /s) below which the simulation halts at line 1522.

8. Initial number of PVC Chains: 
   - Edit the initial count of PVC chains at line 1525.

9. Simulation Directory: 
  - Assign the base directory for simulation output at lines 1528 and 1545.

10. % DHC Calculation: 
   - Adjust the calculation method for % DHC at line 1644.

11. MD Execution: 
    - Adjust the commands for the MD simulation between lines 1354 and 1383
    - Set the frequency of MD stage execution relative to KMC stages at line 1653. 
      e.g. 'if iteration % 1 == 0' signifies MD stage execution after every KMC stage.

Ensure that all modifications align with the specific requirements of your simulation.
