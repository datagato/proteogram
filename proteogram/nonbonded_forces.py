"""Modelling non-bonded forces in PDB file (e.g., Van der Waals) with OpenMM.

This module provides functionality for:
    - Preprocessing PDB structures (fixing, solvation, ionization)
    - Running MD simulations (equilibration and production)
    - Calculating residue-residue interaction energies (VdW and electrostatic)
"""
from openmm.app import (
    PDBFile, ForceField, Modeller, Simulation, PME,
    HBonds, StateDataReporter, DCDReporter
)
from openmm import (
    LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
    NonbondedForce, CustomNonbondedForce, Context, System,
    VerletIntegrator, CustomExternalForce
)
from openmm.unit import (
    kelvin, atmosphere, picosecond, picoseconds, femtoseconds,
    nanometer, kilojoules_per_mole, elementary_charge, angstrom,
    kilojoules_per_mole, nanometers
)
from pdbfixer import PDBFixer

import io
import numpy as np
import sys
import warnings
from typing import Optional
from pathlib import Path


class NonBondedForceModel:
    """Model for computing non-bonded forces between residues using MD simulation.

    This class provides a complete pipeline for:
        1. Preprocessing PDB structures (fixing, solvation, ionization)
        2. Running MD simulations (NPT and NVT equilibration, production)
        3. Calculating residue-residue interaction energies

    Attributes:
        pdb_path (str): Path to the input PDB file.
        temperature (float): Simulation temperature in Kelvin.
        pressure (float): Simulation pressure in atmospheres.
        padding (float): Padding around protein for water box in nanometers.
        timestep (float): Integration timestep in femtoseconds.
        forcefield (ForceField): OpenMM force field object.
        topology: OpenMM topology of the system.
        positions: Atomic positions.
        system (System): OpenMM system object.
        simulation (Simulation): OpenMM simulation object.
        residue_atom_indices (dict): Mapping of residue index to atom indices.
    """

    # Default simulation parameters
    DEFAULT_TEMPERATURE = 310.15  # Kelvin (37 C)
    DEFAULT_PRESSURE = 1.0  # atmospheres
    DEFAULT_FRICTION_COEFFICIENT = 1.0  # 1/ps
    DEFAULT_PADDING = 1.0  # nanometers
    DEFAULT_TIMESTEP = 2.0  # femtoseconds
    DEFAULT_NPT_STEPS = 50000  # 100 ps with 2 fs timestep
    DEFAULT_NVT_STEPS = 50000  # 100 ps  with 2 fs timestep
    DEFAULT_PRODUCTION_STEPS = 500000  # 1 ns with 2 fs timestep
    DEFAULT_REPORTING_INTERVAL = 5000  # Report every 10 ps

    def __init__(
        self,
        pdb_path: str,
        temperature: float = DEFAULT_TEMPERATURE,
        pressure: float = DEFAULT_PRESSURE,
        padding: float = DEFAULT_PADDING,
        timestep: float = DEFAULT_TIMESTEP,
        use_gpu: bool = False,
        output_dir: Optional[str] = None
    ):
        """Initialize the NonBondedForceModel.

        Args:
            pdb_path (str): Path to the input PDB file.
            temperature (float, optional): Simulation temperature in Kelvin.
                Defaults to 300 K.
            pressure (float, optional): Simulation pressure in atmospheres.
                Defaults to 1 atm.
            padding (float, optional): Padding around protein for water box
                in nanometers. Defaults to 1.0 nm.
            timestep (float, optional): Integration timestep in femtoseconds.
                Defaults to 2 fs.
            use_gpu (bool, optional): Whether to use GPU acceleration.
                Defaults to False (CPU).
            output_dir (str, optional): Directory for saving debug outputs
                (e.g., energy plots). Defaults to the PDB file's parent directory.
        """
        self.pdb_path = pdb_path
        self.temperature = temperature * kelvin
        self.pressure = pressure * atmosphere
        self.padding = padding * nanometer
        self.timestep = timestep * femtoseconds
        self.use_gpu = use_gpu
        self.output_dir = Path(output_dir) if output_dir else Path(pdb_path).parent

        # These will be set during setup
        self.forcefield = None
        self.topology = None
        self.positions = None
        self.modeller = None
        self.system = None
        self.simulation = None
        self.residue_atom_indices = {}
        self.protein_residue_indices = []
        self.debug = False
        
        # Energy logging for debug mode
        # Each stage stores: {'time_ps': [], 'energy_kj': [], 'stage': str}
        self.energy_log = {
            'initial': {'time_ps': [], 'energy_kj': [], 'stage': 'Initial'},
            'minimization': {'time_ps': [], 'energy_kj': [], 'stage': 'Minimization'},
            'npt': {'time_ps': [], 'energy_kj': [], 'stage': 'NPT Equilibration'},
            'nvt': {'time_ps': [], 'energy_kj': [], 'stage': 'NVT Equilibration'},
            'production': {'time_ps': [], 'energy_kj': [], 'stage': 'Production'}
        }

    @staticmethod
    def fix_pdb_file(pdb_path: str) -> io.StringIO:
        """Fix a PDB structure for input to MD simulation.

        Performs the following fixes:
            - Replace non-standard residues with standard equivalents
            - Remove heterogens including crystal waters
            - Add missing atoms
            - Add hydrogens (at pH 7.0)

        Args:
            pdb_path (str): Path to the input PDB file.

        Returns:
            io.StringIO: A PDB file in memory after fixing.
        """
        fixer = PDBFixer(pdb_path)

        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        # Write PDB file info to an IO stream
        pdb_file_in_mem = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, pdb_file_in_mem)
        pdb_file_in_mem.seek(0)

        return pdb_file_in_mem

    def setup_system(self) -> None:
        """Set up the MD simulation system.

        This method:
            1. Fixes the PDB structure
            2. Loads the AMBER ff19SB force field with TIP3P water
            3. Solvates the protein in a water box
            4. Adds ions to neutralize the system
            5. Creates the OpenMM system with PME electrostatics

        Raises:
            RuntimeError: If system setup fails.
        """
        # Step 1: Fix the PDB structure
        fixed_pdb = self.fix_pdb_file(self.pdb_path)
        pdb = PDBFile(fixed_pdb)
        
        # Step 2: Load force field (AMBER ff19SB + TIP3P water)
        self.forcefield = ForceField('amber19-all.xml', 'amber19/tip3pfb.xml')
        
        # Step 3: Create modeller and add solvent
        self.modeller = Modeller(pdb.topology, pdb.positions)
        
        # Store protein residue indices before adding water
        self._identify_protein_residues()
        
        # Add water box with padding
        self.modeller.addSolvent(
            self.forcefield,
            model='tip3p',
            padding=self.padding,
            neutralize=True,  # Add ions to neutralize
            positiveIon='Na+',
            negativeIon='Cl-'
        )
        
        self.topology = self.modeller.topology
        self.positions = self.modeller.positions
        
        # Step 4: Create the system with PME for long-range electrostatics
        self.system = self.forcefield.createSystem(
            self.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            constraints=HBonds  # Allows 2 fs timestep
        )
        
        # Build residue-to-atom mapping for energy calculations
        self._build_residue_atom_mapping()

    def _identify_protein_residues(self) -> None:
        """Identify protein residue indices before solvation."""
        protein_resnames = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
            'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
            'TYR', 'VAL', 'HIE', 'HID', 'HIP', 'CYX'
        }
        self.protein_residue_indices = []
        for i, residue in enumerate(self.modeller.topology.residues()):
            if residue.name in protein_resnames:
                self.protein_residue_indices.append(i)

    def _build_residue_atom_mapping(self) -> None:
        """Build a mapping from residue index to atom indices."""
        self.residue_atom_indices = {}
        for residue in self.topology.residues():
            res_idx = residue.index
            atom_indices = [atom.index for atom in residue.atoms()]
            self.residue_atom_indices[res_idx] = atom_indices
        
        # Also identify solvent and ion atoms for interaction group calculations
        self._identify_solvent_atoms()

    def _identify_solvent_atoms(self) -> None:
        """Identify solvent (water) and ion atom indices after solvation.
        
        This builds sets of atom indices for:
            - protein_atom_indices: All atoms belonging to protein residues
            - solvent_atom_indices: Water molecules (HOH, WAT, TIP3, etc.)
            - ion_atom_indices: Ions (Na+, Cl-, K+, etc.)
        """
        solvent_resnames = {'HOH', 'WAT', 'TIP3', 'TIP4', 'SPC', 'T3P', 'T4P', 'T5P'}
        ion_resnames = {'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'NA+', 'CL-', 'K+', 'MG2+', 'CA2+', 'ZN2+'}
        
        self.protein_atom_indices = set()
        self.solvent_atom_indices = set()
        self.ion_atom_indices = set()
        
        for residue in self.topology.residues():
            atom_indices = [atom.index for atom in residue.atoms()]
            resname = residue.name.upper()
            
            if residue.index in self.protein_residue_indices:
                self.protein_atom_indices.update(atom_indices)
            elif resname in solvent_resnames:
                self.solvent_atom_indices.update(atom_indices)
            elif resname in ion_resnames:
                self.ion_atom_indices.update(atom_indices)
            else:
                # Unknown residue type - could be ligand, treat as solvent for now
                self.solvent_atom_indices.update(atom_indices)

    def _get_platform(self) -> Platform:
        """Get the appropriate compute platform.

        Returns:
            Platform: OpenMM platform (CUDA if GPU, CPU otherwise).
        """
        if self.use_gpu:
            try:
                return Platform.getPlatformByName('CUDA')
            except Exception:
                print("CUDA not available, falling back to CPU")
                return Platform.getPlatformByName('CPU')
        return Platform.getPlatformByName('CPU')

    def _create_new_simulation(self,
                           hbonds_constraint: bool = False,
                           add_calpha_restraint: bool = False,
                           add_barostat: bool = False) -> None:
        """Create an OpenMM simulation object.

        Args:
            hbonds_constraint (bool): Whether to constrain hydrogen bonds.
            add_barostat (bool): Whether to add a barostat for NPT simulation.
            add_calpha_restraint (bool): Whether to add constraints to CA atoms.
        """
        # Create a fresh system copy for modification
        if hbonds_constraint:
            constraints = HBonds
        else:
            constraints = None

        system = self.forcefield.createSystem(
            self.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            constraints=constraints
        )
        
        if add_barostat:
            barostat = MonteCarloBarostat(self.pressure, self.temperature)
            self.system.addForce(barostat)

        if add_calpha_restraint:
            # Get indices of CA atoms
            ca_indices = [atom.index for atom in self.topology.atoms() if atom.name == 'CA']

            # Custom external force to restrain CA atoms to their initial positions
            # restraint_force = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
            restraint_force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")

            # Add per-particle parameters for reference positions
            restraint_force.addPerParticleParameter('x0') # reference x
            restraint_force.addPerParticleParameter('y0') # reference y
            restraint_force.addPerParticleParameter('z0') # reference z

            # Add global parameter for force constant
            restraint_force.addGlobalParameter('k', 1000.0 * kilojoules_per_mole / nanometer**2)

            # Add CA atoms to the constraint force
            for idx in ca_indices:
                restraint_force.addParticle(idx, self.positions[idx].value_in_unit(nanometers))
            system.addForce(restraint_force)
        
        integrator = LangevinMiddleIntegrator(
            self.temperature,
            self.DEFAULT_FRICTION_COEFFICIENT / picosecond,
            self.timestep
        )
        
        platform = self._get_platform()
        self.simulation = Simulation(
            self.topology,
            system,
            integrator,
            platform
        )
        self.simulation.context.reinitialize(preserveState=False)
        self.simulation.context.setPositions(self.positions)

    def minimize_energy(self, max_iterations: int = 1000) -> None:
        """Perform energy minimization.

        Args:
            max_iterations (int): Maximum number of minimization iterations.
        """
        print("Performing energy minimization...")
        
        # Log initial energy before minimization (debug mode)
        if self.debug:
            initial_state = self.simulation.context.getState(getEnergy=True)
            initial_energy = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            self._log_energy('initial', 0.0, initial_energy)
            self._log_energy('minimization', 0.0, initial_energy)
        
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
        self.positions = self.simulation.context.getState(
            getPositions=True
        ).getPositions()
        
        # Log final energy after minimization (debug mode)
        if self.debug:
            final_state = self.simulation.context.getState(getEnergy=True)
            final_energy = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            # Use a small time increment to show minimization as a step
            self._log_energy('minimization', 0.1, final_energy)
            print(f"  [DEBUG] Minimization: {initial_energy:.1f} -> {final_energy:.1f} kJ/mol")
        
        print("Energy minimization complete.")

    def get_simulated_pdb_stream(self) -> io.StringIO:
        """Get the current structure as a PDB file stream.

        This method writes the current positions to an in-memory PDB file
        that can be saved or further processed. Can be called after any
        simulation step (minimization, equilibration, or production).

        Returns:
            io.StringIO: An in-memory PDB file containing the current structure.

        Raises:
            RuntimeError: If positions are not available (simulation not run).
        """
        if self.positions is None:
            raise RuntimeError(
                "No positions available. Run minimize_energy() or a simulation first."
            )
        if self.topology is None:
            raise RuntimeError(
                "No topology available. Run setup_system() first."
            )

        pdb_stream = io.StringIO()
        PDBFile.writeFile(self.topology, self.positions, pdb_stream)
        pdb_stream.seek(0)
        return pdb_stream

    def _log_energy(
        self,
        stage: str,
        time_ps: float,
        energy_kj: float
    ) -> None:
        """Log energy value for a given stage (only when debug is enabled).

        Args:
            stage (str): One of 'initial', 'minimization', 'npt', 'nvt', 'production'.
            time_ps (float): Time in picoseconds.
            energy_kj (float): Potential energy in kJ/mol.
        """
        if self.debug and stage in self.energy_log:
            self.energy_log[stage]['time_ps'].append(time_ps)
            self.energy_log[stage]['energy_kj'].append(energy_kj)

    def _reset_energy_log(self) -> None:
        """Reset all energy logs (useful when starting a new simulation)."""
        for stage in self.energy_log:
            self.energy_log[stage]['time_ps'] = []
            self.energy_log[stage]['energy_kj'] = []

    def plot_energy_history(
        self,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> None:
        """Plot energy vs. time for all simulation stages.

        Creates a multi-panel figure showing energy evolution during:
        - Initial state (single point)
        - Energy minimization
        - NPT equilibration
        - NVT equilibration
        - Production run

        Args:
            output_path (str, optional): Path to save the figure. If None,
                saves to the same directory as the PDB file with suffix '_energy.png'.
            show_plot (bool): Whether to display the plot interactively.
                Defaults to False.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            warnings.warn(
                "matplotlib not installed. Cannot generate energy plot. "
                "Install with: pip install matplotlib"
            )
            return

        # Collect all stages that have data
        stages_with_data = []
        for stage_key, stage_data in self.energy_log.items():
            if stage_data['energy_kj']:
                stages_with_data.append((stage_key, stage_data))

        if not stages_with_data:
            print("No energy data to plot.")
            return

        # Color scheme for different stages
        stage_colors = {
            'initial': '#2ecc71',       # Green
            'minimization': '#3498db',  # Blue
            'npt': '#9b59b6',           # Purple
            'nvt': '#e67e22',           # Orange
            'production': '#e74c3c'     # Red
        }

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
        
        # Top panel: Combined view with cumulative time
        ax_combined = axes[0]
        cumulative_time = 0.0
        stage_boundaries = []  # Track where each stage ends
        
        for stage_key, stage_data in stages_with_data:
            times = np.array(stage_data['time_ps'])
            energies = np.array(stage_data['energy_kj'])
            color = stage_colors.get(stage_key, '#7f8c8d')
            
            if len(times) > 0:
                # Adjust times to be cumulative
                adjusted_times = times + cumulative_time
                
                if stage_key == 'initial':
                    # Plot as a single point
                    ax_combined.scatter(adjusted_times, energies, color=color, 
                                       s=100, zorder=5, label=stage_data['stage'])
                else:
                    ax_combined.plot(adjusted_times, energies, color=color, 
                                    linewidth=1.5, label=stage_data['stage'])
                
                # Update cumulative time
                if len(times) > 1:
                    cumulative_time = adjusted_times[-1]
                    stage_boundaries.append((cumulative_time, stage_data['stage']))
        
        # Add vertical lines at stage boundaries
        for boundary_time, stage_name in stage_boundaries[:-1]:  # Skip last boundary
            ax_combined.axvline(x=boundary_time, color='gray', linestyle='--', 
                               alpha=0.5, linewidth=0.8)
        
        ax_combined.set_xlabel('Time (ps)', fontsize=11)
        ax_combined.set_ylabel('Potential Energy (kJ/mol)', fontsize=11)
        ax_combined.set_title('Energy Evolution Throughout Simulation', fontsize=12, fontweight='bold')
        ax_combined.legend(loc='upper right', fontsize=9)
        ax_combined.grid(True, alpha=0.3)
        
        # Bottom panel: Individual stage panels
        ax_individual = axes[1]
        
        # Calculate how many stages have more than 1 data point
        multi_point_stages = [(k, d) for k, d in stages_with_data 
                              if len(d['energy_kj']) > 1]
        
        if multi_point_stages:
            n_stages = len(multi_point_stages)
            
            # Create inset axes for each stage
            for i, (stage_key, stage_data) in enumerate(multi_point_stages):
                times = np.array(stage_data['time_ps'])
                energies = np.array(stage_data['energy_kj'])
                color = stage_colors.get(stage_key, '#7f8c8d')
                
                # Calculate subplot position
                width = 0.8 / n_stages
                left = 0.1 + i * (0.85 / n_stages)
                
                # Create inset axis
                ax_inset = ax_individual.inset_axes([left, 0.15, width * 0.9, 0.75])
                ax_inset.plot(times, energies, color=color, linewidth=1.2)
                ax_inset.set_title(stage_data['stage'], fontsize=10, fontweight='bold')
                ax_inset.set_xlabel('Time (ps)', fontsize=8)
                ax_inset.set_ylabel('Energy (kJ/mol)', fontsize=8)
                ax_inset.tick_params(axis='both', labelsize=7)
                ax_inset.grid(True, alpha=0.3)
                
                # Add energy change annotation
                energy_change = energies[-1] - energies[0]
                change_text = f'ΔE = {energy_change:+.1f} kJ/mol'
                ax_inset.annotate(change_text, xy=(0.5, 0.02), xycoords='axes fraction',
                                 ha='center', fontsize=8, 
                                 color='green' if energy_change < 0 else 'red')
        
        # Hide the main bottom axis (we're using insets)
        ax_individual.set_visible(False)
        
        plt.tight_layout()
        
        if output_path is None:
            # Save to output_dir (defaults to PDB file's parent directory)
            debug_plots_dir = self.output_dir / "debug_plots"
            debug_plots_dir.mkdir(parents=True, exist_ok=True)
            
            pdb_stem = Path(self.pdb_path).stem
            output_path = debug_plots_dir / f"{pdb_stem}_energy.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def _validate_energy(
        self,
        energy_kj: float,
        stage: str,
        prev_energy: float | None = None,
        n_atoms: int | None = None
    ) -> list[str]:
        """Validate energy values and return warnings if issues detected.

        Args:
            energy_kj (float): Current potential energy in kJ/mol.
            stage (str): Simulation stage name (e.g., 'NPT', 'NVT', 'Production').
            prev_energy (float, optional): Previous energy value for trend checking.
            n_atoms (int, optional): Number of atoms for per-atom normalization.

        Returns:
            list[str]: List of warning messages (empty if no issues).
        """
        warnings_list = []

        # Check for positive energy (system exploding)
        if energy_kj > 0:
            warnings_list.append(
                f"WARNING [{stage}]: Positive potential energy ({energy_kj:.1f} kJ/mol) - "
                "system may be unstable or exploding!"
            )

        # Check for sudden large energy increase
        if prev_energy is not None:
            delta = energy_kj - prev_energy
            # Warning if energy increases by more than 10%
            if prev_energy < 0 and delta > abs(prev_energy) * 0.1:
                warnings_list.append(
                    f"WARNING [{stage}]: Large energy increase detected "
                    f"({delta:.1f} kJ/mol, {100*delta/abs(prev_energy):.1f}% change)"
                )
            # Warning for very large jumps
            if abs(delta) > 100000:  # 100,000 kJ/mol jump
                warnings_list.append(
                    f"WARNING [{stage}]: Very large energy change ({delta:.1f} kJ/mol) - "
                    "check for instabilities"
                )

        # Check per-atom energy if n_atoms provided
        if n_atoms is not None and n_atoms > 0:
            per_atom = energy_kj / n_atoms
            # Typical range is -10 to -20 kJ/mol per atom
            if per_atom > 0:
                warnings_list.append(
                    f"WARNING [{stage}]: Positive per-atom energy ({per_atom:.2f} kJ/mol/atom)"
                )
            elif per_atom < -50:
                warnings_list.append(
                    f"WARNING [{stage}]: Unusually low per-atom energy ({per_atom:.2f} kJ/mol/atom) - "
                    "possible atomic overlaps"
                )

        return warnings_list

    def _get_n_atoms(self) -> int:
        """Get the total number of atoms in the system."""
        return sum(1 for _ in self.topology.atoms())

    def equilibrate_npt(
        self,
        steps: int = DEFAULT_NPT_STEPS,
        report_interval: int = DEFAULT_REPORTING_INTERVAL
    ) -> None:
        """Run NPT equilibration (constant pressure and temperature).
        This is run before NVT equilibration.

        Args:
            steps (int): Number of simulation steps.
            report_interval (int): Interval for reporting state data.
        """
        print(f"Running NPT equilibration for {steps} steps...")
        
        # Create simulation with barostat
        self._create_new_simulation(
            add_calpha_restraint=True,
            add_barostat=True)
        
        # Add reporter for monitoring
        self.simulation.reporters.append(
            StateDataReporter(
                sys.stdout,
                report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                density=True,
                separator='\t'
            )
        )
        
        # Get initial energy for monitoring
        n_atoms = self._get_n_atoms()
        initial_state = self.simulation.context.getState(getEnergy=True)
        initial_energy = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Initial potential energy: {initial_energy:.1f} kJ/mol "
              f"({initial_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Track energies for validation
        energy_history = [initial_energy]
        check_interval = max(steps // 5, report_interval)  # Check 5 times during equilibration
        
        # Calculate timestep in ps for energy logging
        timestep_ps = self.timestep.value_in_unit(picoseconds)
        
        # Log initial energy (debug mode)
        if self.debug:
            self._log_energy('npt', 0.0, initial_energy)
        
        # Run equilibration in chunks for monitoring
        steps_run = 0
        while steps_run < steps:
            chunk = min(check_interval, steps - steps_run)
            self.simulation.step(chunk)
            steps_run += chunk
            
            # Get current energy
            state = self.simulation.context.getState(getEnergy=True)
            current_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            energy_history.append(current_energy)
            
            # Log energy (debug mode)
            if self.debug:
                time_ps = steps_run * timestep_ps
                self._log_energy('npt', time_ps, current_energy)
            
            # Validate energy
            warnings_list = self._validate_energy(
                current_energy, 'NPT', 
                prev_energy=energy_history[-2] if len(energy_history) > 1 else None,
                n_atoms=n_atoms
            )
            for w in warnings_list:
                warnings.warn(w)
                print(f"  {w}")
        
        # Final energy check
        final_state = self.simulation.context.getState(getEnergy=True)
        final_energy = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Final potential energy: {final_energy:.1f} kJ/mol "
              f"({final_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Check overall trend
        if final_energy > initial_energy:
            warnings.warn(
                f"NPT equilibration: Energy increased from {initial_energy:.1f} to {final_energy:.1f} kJ/mol"
            )
            print(f"  WARNING: Energy increased during NPT equilibration")
        else:
            print(f"  Energy decreased by {initial_energy - final_energy:.1f} kJ/mol (good)")
        
        # Update positions
        state = self.simulation.context.getState(
            getPositions=True
        )
        self.positions = state.getPositions()
        print("NPT equilibration complete.")

    def equilibrate_nvt(
        self,
        steps: int = DEFAULT_NVT_STEPS,
        report_interval: int = DEFAULT_REPORTING_INTERVAL
    ) -> None:
        """Run NVT equilibration (constant volume and temperature).
        This is run after NPT equilibration.
        
        Args:
            steps (int): Number of simulation steps.
            report_interval (int): Interval for reporting state data.
        """
        print(f"Running NVT equilibration for {steps} steps...")
        
        # Create simulation with barostat
        self._create_new_simulation(
            add_calpha_restraint=True,
            add_barostat=False)
        
        # Add reporter for monitoring
        self.simulation.reporters.append(
            StateDataReporter(
                sys.stdout,
                report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                separator='\t'
            )
        )
        
        # Get initial energy for monitoring
        n_atoms = self._get_n_atoms()
        initial_state = self.simulation.context.getState(getEnergy=True)
        initial_energy = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Initial potential energy: {initial_energy:.1f} kJ/mol "
              f"({initial_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Track energies for validation
        energy_history = [initial_energy]
        check_interval = max(steps // 5, report_interval)  # Check 5 times during equilibration
        
        # Calculate timestep in ps for energy logging
        timestep_ps = self.timestep.value_in_unit(picoseconds)
        
        # Log initial energy (debug mode)
        if self.debug:
            self._log_energy('nvt', 0.0, initial_energy)
        
        # Run equilibration in chunks for monitoring
        steps_run = 0
        while steps_run < steps:
            chunk = min(check_interval, steps - steps_run)
            self.simulation.step(chunk)
            steps_run += chunk
            
            # Get current energy
            state = self.simulation.context.getState(getEnergy=True)
            current_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            energy_history.append(current_energy)
            
            # Log energy (debug mode)
            if self.debug:
                time_ps = steps_run * timestep_ps
                self._log_energy('nvt', time_ps, current_energy)
            
            # Validate energy
            warnings_list = self._validate_energy(
                current_energy, 'NVT', 
                prev_energy=energy_history[-2] if len(energy_history) > 1 else None,
                n_atoms=n_atoms
            )
            for w in warnings_list:
                warnings.warn(w)
                print(f"  {w}")
        
        # Final energy check
        final_state = self.simulation.context.getState(getEnergy=True)
        final_energy = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Final potential energy: {final_energy:.1f} kJ/mol "
              f"({final_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Check overall trend
        if final_energy > initial_energy:
            warnings.warn(
                f"NVT equilibration: Energy increased from {initial_energy:.1f} to {final_energy:.1f} kJ/mol"
            )
            print(f"  WARNING: Energy increased during NVT equilibration")
        else:
            print(f"  Energy decreased by {initial_energy - final_energy:.1f} kJ/mol (good)")
        
        # Update positions
        state = self.simulation.context.getState(
            getPositions=True
        )
        self.positions = state.getPositions()
        print("NVT equilibration complete.")

    def equilibrate_nvt_with_warming(
        self,
        steps: int = DEFAULT_NVT_STEPS,
        report_interval: int = DEFAULT_REPORTING_INTERVAL
    ) -> None:
        """Run NVT equilibration (constant volume and temperature).
        This is run after NPT equilibration.

        Args:
            steps (int): Number of simulation steps.
            report_interval (int): Interval for reporting state data.
        """
        print(f"Running NVT equilibration for {steps} steps...")

        # Create simulation without barostat but with CA constraints
        self._create_new_simulation(
            add_calpha_restraint=True,
            add_barostat=False)

        # Slowly warm up temperature - every 1000 steps raise 
        # the temperature by 5 K
        self.simulation.context.setVelocitiesToTemperature(5*kelvin)

        # Add reporter for monitoring
        self.simulation.reporters.append(
            StateDataReporter(
                sys.stdout,
                report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                separator='\t'
            )
        )

        # Get initial energy for monitoring
        n_atoms = self._get_n_atoms()
        initial_state = self.simulation.context.getState(getEnergy=True)
        initial_energy = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Initial potential energy: {initial_energy:.1f} kJ/mol "
              f"({initial_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Track energies for validation
        energy_history = [initial_energy]
        
        # Calculate timestep in ps for energy logging
        timestep_ps = self.timestep.value_in_unit(picoseconds)
        
        # Log initial energy (debug mode)
        if self.debug:
            self._log_energy('nvt', 0.0, initial_energy)
        
        T = 5
        n = 1000
        n_intervals = steps // n
        energy_check_interval = max(n_intervals // 10, 1)  # Check ~10 times
        
        steps_run = 0
        for i in range(n_intervals):
            self.simulation.step(n)
            steps_run += n
            temperature = (T+(i*T))*kelvin 
            self.simulation.integrator.setTemperature(temperature)
            
            # Periodic energy validation
            if (i + 1) % energy_check_interval == 0:
                state = self.simulation.context.getState(getEnergy=True)
                current_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                energy_history.append(current_energy)
                
                # Log energy (debug mode)
                if self.debug:
                    time_ps = steps_run * timestep_ps
                    self._log_energy('nvt', time_ps, current_energy)
                
                # During heating, energy is expected to increase, but watch for instabilities
                warnings_list = self._validate_energy(
                    current_energy, 'NVT',
                    prev_energy=None,  # Don't warn about increase during heating
                    n_atoms=n_atoms
                )
                for w in warnings_list:
                    warnings.warn(w)
                    print(f"  {w}")
        
        # Final energy check
        final_state = self.simulation.context.getState(getEnergy=True)
        final_energy = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Final potential energy: {final_energy:.1f} kJ/mol "
              f"({final_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Note: Energy increase during NVT heating is expected
        if final_energy > initial_energy:
            print(f"  Energy increased by {final_energy - initial_energy:.1f} kJ/mol "
                  f"(expected during temperature ramping from 5K to 300K)")
        
        # Check for very large energy (possible instability)
        if final_energy > 0:
            warnings.warn("NVT equilibration ended with positive energy - system may be unstable")
            print(f"  WARNING: Positive final energy detected!")
        
        # Update positions
        state = self.simulation.context.getState(
            getPositions=True
        )
        self.positions = state.getPositions()
        print("NVT equilibration complete.")

    def run_production(
        self,
        steps: int = DEFAULT_PRODUCTION_STEPS,
        energy_calc_interval: int = 10000,
        subtract_solvent: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run production MD and calculate residue-residue interaction energies and forces.

        This method runs an MD simulation and periodically calculates the
        pairwise interaction energies and forces between all protein residues.

        Args:
            steps (int): Number of production steps.
            energy_calc_interval (int): Interval (in steps) for calculating
                interaction energies.
            subtract_solvent (bool): Whether to subtract solvent interaction energies.

        Returns:
            tuple[np.ndarray, ...]: A tuple of four NxN matrices:
                - vdw_attractive: Attractive VdW energies (kJ/mol)
                - vdw_repulsive: Repulsive VdW energies (kJ/mol)
                - es_attractive: Attractive electrostatic energies (kJ/mol)
                - es_repulsive: Repulsive electrostatic energies (kJ/mol)
        """
        print(f"Running production MD for {steps} steps...")
        
        # Create fresh simulation for production
        self._create_new_simulation(
            hbonds_constraint=False,
            add_calpha_restraint=False,
            add_barostat=False)
        
        n_residues = len(self.protein_residue_indices)
        n_frames = steps // energy_calc_interval
        n_atoms = self._get_n_atoms()

        report_interval = max(steps // 10, self.DEFAULT_REPORTING_INTERVAL)
        # Add reporter for monitoring
        self.simulation.reporters.append(
            StateDataReporter(
                sys.stdout,
                report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                separator='\t'
            )
        )

        # Get initial energy
        initial_state = self.simulation.context.getState(getEnergy=True)
        initial_energy = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"  Initial potential energy: {initial_energy:.1f} kJ/mol "
              f"({initial_energy/n_atoms:.2f} kJ/mol/atom)")
        
        # Track energy statistics
        energy_history = [initial_energy]
        energy_min = initial_energy
        energy_max = initial_energy
        
        # Calculate timestep in ps for energy logging
        timestep_ps = self.timestep.value_in_unit(picoseconds)
        
        # Log initial energy (debug mode)
        if self.debug:
            self._log_energy('production', 0.0, initial_energy)
        
        # Accumulators for energies
        vdw_energy_attractive_sum = np.zeros((n_residues, n_residues))
        vdw_energy_repulsive_sum = np.zeros((n_residues, n_residues))
        es_energy_attractive_sum = np.zeros((n_residues, n_residues))
        es_energy_repulsive_sum = np.zeros((n_residues, n_residues))
        
        steps_run = 0
        frame_count = 0
        for _ in range(0, steps, energy_calc_interval):
            # Run simulation chunk
            self.simulation.step(energy_calc_interval)
            steps_run += energy_calc_interval
            
            # Get current state with positions and energy
            state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            positions = state.getPositions(asNumpy=True)
            current_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            
            # Track energy statistics
            energy_history.append(current_energy)
            energy_min = min(energy_min, current_energy)
            energy_max = max(energy_max, current_energy)
            
            # Log energy (debug mode)
            if self.debug:
                time_ps = steps_run * timestep_ps
                self._log_energy('production', time_ps, current_energy)
            
            # Validate energy periodically
            if frame_count % 10 == 0:
                warnings_list = self._validate_energy(
                    current_energy, 'Production',
                    prev_energy=energy_history[-2] if len(energy_history) > 1 else None,
                    n_atoms=n_atoms
                )
                for w in warnings_list:
                    warnings.warn(w)
                    print(f"  {w}")
            
            # Calculate pairwise energies and forces for this frame
            (vdw_e_att, vdw_e_rep, es_e_att, es_e_rep) = self._calculate_pairwise_energies(
                positions=positions,
                subtract_solvent=subtract_solvent
            )

            # # Calculate pairwise energies and forces for this frame
            # (vdw_e_att, vdw_e_rep, es_e_att, es_e_rep) = self._calculate_pairwise_energies_get_potential(
            #     positions
            # )
            # vdw_f_att, vdw_f_rep, es_f_att, es_f_rep = 1,1,1,1 # Placeholder for forces - need to implement force calculation

            # Accumulate energies
            vdw_energy_attractive_sum += vdw_e_att
            vdw_energy_repulsive_sum += vdw_e_rep
            es_energy_attractive_sum += es_e_att
            es_energy_repulsive_sum += es_e_rep

            frame_count+=1
        
        # Final energy statistics
        final_energy = energy_history[-1]
        mean_energy = np.mean(energy_history)
        std_energy = np.std(energy_history)
        
        print(f"\n  Production MD Energy Statistics:")
        print(f"    Initial: {initial_energy:.1f} kJ/mol")
        print(f"    Final:   {final_energy:.1f} kJ/mol")
        print(f"    Mean:    {mean_energy:.1f} ± {std_energy:.1f} kJ/mol")
        print(f"    Range:   [{energy_min:.1f}, {energy_max:.1f}] kJ/mol")
        print(f"    Per-atom mean: {mean_energy/n_atoms:.2f} kJ/mol/atom")
        
        # Warn if energy fluctuations are very large
        if std_energy > abs(mean_energy) * 0.05:  # >5% relative fluctuation
            warnings.warn(
                f"Large energy fluctuations during production: std={std_energy:.1f} kJ/mol "
                f"({100*std_energy/abs(mean_energy):.1f}% of mean)"
            )
            print(f"  WARNING: Large energy fluctuations detected")
        
        # Update positions after production
        self.positions = self.simulation.context.getState(
            getPositions=True
        ).getPositions()

        # Average over frames
        vdw_energy_attractive_avg = vdw_energy_attractive_sum / frame_count
        vdw_energy_repulsive_avg = vdw_energy_repulsive_sum / frame_count
        es_energy_attractive_avg = es_energy_attractive_sum / frame_count
        es_energy_repulsive_avg = es_energy_repulsive_sum / frame_count
        
        print("Production MD complete.")
        
        # Generate energy plot if debug mode is enabled
        if self.debug:
            self.plot_energy_history()
        
        return [vdw_energy_attractive_avg, vdw_energy_repulsive_avg, es_energy_attractive_avg, es_energy_repulsive_avg]

    def _get_context(self) -> Context:
        """
        Create the integrator and (re)set up the simulation Context.
        """
        integrator = LangevinMiddleIntegrator(self.DEFAULT_TEMPERATURE,
                                              self.DEFAULT_FRICTION_COEFFICIENT / picosecond,
                                              self.DEFAULT_TIMESTEP)
        context = Context(self.system,
                        integrator,
                        Platform.getPlatformByName(self._get_platform()),
                        self.platform_properties) 
        return context.setPositions(self.positions)

    def _energy_calculation(self,
            solute_coulomb_scale,
            solute_lj_scale,
            solvent_coulomb_scale,
            solvent_lj_scale):
        """
        Calculate the energy with a new Context.
        """
        context = self._get_context()
        context.setParameter("solute_coulomb_scale", solute_coulomb_scale)
        context.setParameter("solute_lj_scale", solute_lj_scale)
        context.setParameter("solvent_coulomb_scale", solvent_coulomb_scale)
        context.setParameter("solvent_lj_scale", solvent_lj_scale)
        return context.getState(getEnergy=True, groups={0}).getPotentialEnergy()

    def _get_vdw_and_electrostatic_energy(self):
        """
        Now we can evaluate the interaction energies by subtracting internal
        energies from the total energy for each type, Coulomb and Lennard-Jones.
        """
        total_coulomb = self._energy_calculation(1, 0, 1, 0)
        solute_coulomb = self._energy_calculation(1, 0, 0, 0)
        solvent_coulomb = self._energy_calculation(0, 0, 1, 0)
        total_lj = self._energy_calculation(0, 1, 0, 1)
        solute_lj = self._energy_calculation(0, 1, 0, 0)
        solvent_lj = self._energy_calculation(0, 0, 0, 1)
        vdw_interaction_energy = total_lj - solute_lj - solvent_lj
        electrostatic_interaction_energy = total_coulomb - solute_coulomb -solvent_coulomb
        return vdw_interaction_energy, electrostatic_interaction_energy

    def _calculate_residue_solvent_energies(
        self,
        positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate per-residue interaction energies with solvent using interaction groups.
        
        Uses CustomNonbondedForce with addInteractionGroup() to efficiently compute
        the interaction energy between each protein residue and all solvent molecules.
        This leverages OpenMM's optimized force calculations (including GPU acceleration).

        Args:
            positions (np.ndarray): Current atomic positions in nanometers.

        Returns:
            tuple: Four N-length arrays (one value per protein residue):
                - vdw_energy_solvent: VdW energy with solvent (kJ/mol)
                - es_energy_solvent: Electrostatic energy with solvent (kJ/mol)
                - vdw_force_solvent: VdW force magnitude with solvent (kJ/(mol·nm))
                - es_force_solvent: ES force magnitude with solvent (kJ/(mol·nm))
        """
        n_residues = len(self.protein_residue_indices)
        
        vdw_energy_solvent = np.zeros(n_residues)
        es_energy_solvent = np.zeros(n_residues)
        vdw_force_solvent = np.zeros(n_residues)
        es_force_solvent = np.zeros(n_residues)
        
        # Get nonbonded force parameters from existing system
        nonbonded_force = None
        for force in self.system.getForces():
            if isinstance(force, NonbondedForce):
                nonbonded_force = force
                break
        
        if nonbonded_force is None:
            raise RuntimeError("No NonbondedForce found in system")
        
        # Combine solvent and ion atoms
        solvent_and_ions = self.solvent_atom_indices | self.ion_atom_indices
        
        # Calculate residue-solvent energies using OpenMM interaction groups
        for i, res_i in enumerate(self.protein_residue_indices):
            residue_atoms = set(self.residue_atom_indices[res_i])
            
            # Create a temporary system with CustomNonbondedForce for this residue-solvent pair
            # LJ force with interaction group (no cutoff - compute all interactions)
            lj_force = CustomNonbondedForce(
                "4*epsilon*((sigma/r)^12-(sigma/r)^6); "
                "sigma=0.5*(sigma1+sigma2); "
                "epsilon=sqrt(epsilon1*epsilon2)"
            )
            lj_force.addPerParticleParameter("sigma")
            lj_force.addPerParticleParameter("epsilon")
            lj_force.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
            lj_force.setForceGroup(0)
            
            # Coulomb force with interaction group (no cutoff - compute all interactions)
            coulomb_force = CustomNonbondedForce(
                "138.935456*charge1*charge2/r"  # k_coulomb in kJ·nm/(mol·e²)
            )
            coulomb_force.addPerParticleParameter("charge")
            coulomb_force.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
            coulomb_force.setForceGroup(1)
            
            # Add all particles with their parameters
            for atom_idx in range(nonbonded_force.getNumParticles()):
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
                lj_force.addParticle([
                    sigma.value_in_unit(nanometer),
                    epsilon.value_in_unit(kilojoules_per_mole)
                ])
                coulomb_force.addParticle([charge.value_in_unit(elementary_charge)])
            
            # Set interaction groups: only compute residue-solvent interactions
            lj_force.addInteractionGroup(residue_atoms, solvent_and_ions)
            coulomb_force.addInteractionGroup(residue_atoms, solvent_and_ions)
            
            # Create temporary system
            temp_system = System()
            for _ in range(nonbonded_force.getNumParticles()):
                temp_system.addParticle(1.0)  # Mass doesn't matter for energy calc
            
            # Copy box vectors from topology
            vectors = self.topology.getPeriodicBoxVectors()
            if vectors is not None:
                temp_system.setDefaultPeriodicBoxVectors(*vectors)
            
            temp_system.addForce(lj_force)
            temp_system.addForce(coulomb_force)
            
            # Create context and calculate energies
            integrator = VerletIntegrator(0.001 * picoseconds)
            platform = self._get_platform()
            context = Context(temp_system, integrator, platform)
            context.setPositions(positions)
            
            # Get LJ energy (force group 0)
            lj_state = context.getState(getEnergy=True, groups={0})
            lj_energy = lj_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            
            # Get Coulomb energy (force group 1)
            coulomb_state = context.getState(getEnergy=True, groups={1})
            coulomb_energy = coulomb_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            
            # Normalize by number of atoms in residue
            n_atoms_in_residue = len(residue_atoms)
            vdw_energy_solvent[i] = lj_energy / n_atoms_in_residue
            es_energy_solvent[i] = coulomb_energy / n_atoms_in_residue
            
            # For forces, we use the energy as a proxy (force calculation would require
            # additional context.getState calls with getForces=True)
            # Approximate force magnitude from energy gradient
            vdw_force_solvent[i] = abs(lj_energy) / n_atoms_in_residue
            es_force_solvent[i] = abs(coulomb_energy) / n_atoms_in_residue
            
            # Clean up
            del context
            del integrator
        
        return vdw_energy_solvent, es_energy_solvent, vdw_force_solvent, es_force_solvent

    def _calculate_pairwise_energies(
        self,
        positions: np.ndarray,
        subtract_solvent: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate pairwise residue-residue interaction energies and forces.

        Uses Lennard-Jones potential for VdW and Coulomb's law for electrostatics.
        Separates attractive and repulsive contributions for both energies.
        
        If subtract_solvent=True, the residue-solvent interaction energies are
        subtracted from the diagonal elements, providing a measure of the 
        desolvation penalty for each residue pair interaction.
        
        WARNING: subtract_solvent=True is computationally expensive as it
        iterates over all solvent atoms for each protein residue.

        Args:
            positions (np.ndarray): Current atomic positions in nanometers.
            subtract_solvent (bool): If True, subtract residue-solvent energies
                from the pairwise energies. This approximates the desolvation 
                penalty when two residues interact. Defaults to False.

        Returns:
            tuple: Eight NxN matrices:
                - vdw_energy_attractive: Attractive VdW energies (kJ/mol)
                - vdw_energy_repulsive: Repulsive VdW energies (kJ/mol)
                - es_energy_attractive: Attractive electrostatic energies (kJ/mol)
                - es_energy_repulsive: Repulsive electrostatic energies (kJ/mol)
        """
        n_residues = len(self.protein_residue_indices)
        
        # Calculate residue-solvent energies if needed for subtraction
        if subtract_solvent:
            (vdw_solv, es_solv, 
             vdw_force_solv, es_force_solv) = self._calculate_residue_solvent_energies(positions)
        
        # Energy matrices
        vdw_energy_attractive = np.zeros((n_residues, n_residues))
        vdw_energy_repulsive = np.zeros((n_residues, n_residues))
        es_energy_attractive = np.zeros((n_residues, n_residues))
        es_energy_repulsive = np.zeros((n_residues, n_residues))
        
        # Get nonbonded force parameters from the system
        nonbonded_force = None
        for force in self.system.getForces():
            if isinstance(force, NonbondedForce):
                nonbonded_force = force
                break
        
        if nonbonded_force is None:
            raise RuntimeError("No NonbondedForce found in system")
        
        # Coulomb constant in OpenMM units (kJ·nm/(mol·e²))
        k_coulomb = 138.935456
        
        # Calculate pairwise energies between protein residues
        for i, res_i in enumerate(self.protein_residue_indices):
            atoms_i = self.residue_atom_indices[res_i]
            
            for j, res_j in enumerate(self.protein_residue_indices):
                if j <= i:
                    continue  # Only upper triangle
                
                atoms_j = self.residue_atom_indices[res_j]
                
                # Energy accumulators
                vdw_energy_att_ij = 0.0
                vdw_energy_rep_ij = 0.0
                es_energy_att_ij = 0.0
                es_energy_rep_ij = 0.0
                
                for atom_i in atoms_i:
                    charge_i, sigma_i, epsilon_i = nonbonded_force.getParticleParameters(atom_i)
                    charge_i = charge_i.value_in_unit(elementary_charge)
                    sigma_i = sigma_i.value_in_unit(nanometer)
                    epsilon_i = epsilon_i.value_in_unit(kilojoules_per_mole)
                    pos_i = positions[atom_i]
                    
                    for atom_j in atoms_j:
                        charge_j, sigma_j, epsilon_j = nonbonded_force.getParticleParameters(atom_j)
                        charge_j = charge_j.value_in_unit(elementary_charge)
                        sigma_j = sigma_j.value_in_unit(nanometer)
                        epsilon_j = epsilon_j.value_in_unit(kilojoules_per_mole)
                        pos_j = positions[atom_j]
                        
                        # Calculate distance
                        r_vec = pos_j - pos_i
                        r = np.sqrt(np.sum(r_vec**2))
                        
                        if r < 0.1:  # Skip if too close (< 1 Angstrom)
                            continue
                        
                        # Lennard-Jones combining rules (Lorentz-Berthelot)
                        sigma_ij = (sigma_i + sigma_j) / 2
                        epsilon_ij = np.sqrt(epsilon_i * epsilon_j)
                        
                        # LJ potential: 4*eps*[(sigma/r)^12 - (sigma/r)^6]
                        sigma_over_r = sigma_ij / r
                        sr6 = sigma_over_r ** 6
                        sr12 = sr6 ** 2
                        
                        # LJ Energy terms
                        # Repulsive term (r^-12)
                        vdw_energy_rep_ij += 4 * epsilon_ij * sr12
                        # Attractive term (r^-6)
                        vdw_energy_att_ij += -4 * epsilon_ij * sr6
                        
                        # Coulomb potential: k * q1 * q2 / r
                        es_energy = k_coulomb * charge_i * charge_j / r
                        
                        if es_energy > 0:
                            es_energy_rep_ij += es_energy
                        else:
                            es_energy_att_ij += es_energy
                
                # Normalize by number of atom pairs
                norm_factor = len(atoms_i) * len(atoms_j)
                vdw_energy_att_ij /= norm_factor
                vdw_energy_rep_ij /= norm_factor
                es_energy_att_ij /= norm_factor
                es_energy_rep_ij /= norm_factor
    
                # Store in matrices (upper triangle)
                vdw_energy_attractive[i, j] = vdw_energy_att_ij
                vdw_energy_repulsive[i, j] = vdw_energy_rep_ij
                es_energy_attractive[i, j] = es_energy_att_ij
                es_energy_repulsive[i, j] = es_energy_rep_ij
        
        # Subtract residue-solvent energies if requested
        # This approximates the desolvation penalty: when residues i and j interact,
        # they partially lose their interactions with solvent
        if subtract_solvent:
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    # Average the solvent energies of both residues involved
                    # This represents the "desolvation cost" of forming this contact
                    avg_vdw_solv = (vdw_solv[i] + vdw_solv[j]) / 2
                    avg_es_solv = (es_solv[i] + es_solv[j]) / 2
                    
                    # Subtract from attractive terms (solvent interaction is typically stabilizing)
                    vdw_energy_attractive[i, j] -= avg_vdw_solv
                    es_energy_attractive[i, j] -= avg_es_solv
        
        return (vdw_energy_attractive, vdw_energy_repulsive, es_energy_attractive, es_energy_repulsive)
    
    def _calculate_pairwise_energies_get_potential(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate pairwise residue-residue interaction energies using OpenMM's getPotentialEnergy.

        This method uses OpenMM's parameter offset feature to compute interaction
        energies directly, rather than manually calculating LJ and Coulomb potentials.
        This approach follows the OpenMM Cookbook methodology for computing interaction
        energies between groups of atoms.

        The interaction energy is computed as:
            E_interaction = E_total(A+B) - E_internal(A) - E_internal(B)

        This method separates Coulomb and Lennard-Jones contributions using
        separate parameter scales for each.

        Note:
            This method creates a new system and context for each residue pair,
            which is computationally expensive but accurate. For large proteins,
            consider using the manual calculation method instead.

        Args:
            positions (np.ndarray): Current atomic positions in nanometers.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two NxN matrices (only upper triangle filled):
                - lj_interaction_energies: Lennard-Jones interaction energies (kJ/mol)
                - coulomb_interaction_energies: Coulomb interaction energies (kJ/mol)
        """
        n_residues = len(self.protein_residue_indices)
        
        # Initialize energy matrices
        vdw_energy_attractive = np.zeros((n_residues, n_residues))
        vdw_energy_repulsive = np.zeros((n_residues, n_residues))
        es_energy_attractive = np.zeros((n_residues, n_residues))
        es_energy_repulsive = np.zeros((n_residues, n_residues))
        
        # Get all protein atom indices for reference
        all_protein_atoms = set()
        for res_idx in self.protein_residue_indices:
            all_protein_atoms.update(self.residue_atom_indices[res_idx])
        
        print(f"Calculating pairwise energies for {n_residues} residues...")
        total_pairs = n_residues * (n_residues - 1) // 2
        # pair_count = 0
        
        # Calculate pairwise energies between protein residues
        for i, res_i in enumerate(self.protein_residue_indices):
            atoms_i = set(self.residue_atom_indices[res_i])
            
            for j, res_j in enumerate(self.protein_residue_indices):
                if j <= i:
                    continue  # Only upper triangle
                
                atoms_j = set(self.residue_atom_indices[res_j])
                
                # Create a fresh system for this residue pair calculation
                system = self.forcefield.createSystem(
                    self.topology,
                    nonbondedMethod=PME,
                    nonbondedCutoff=1.0 * nanometer,
                    constraints=None
                )
                
                # Find and modify the NonbondedForce
                for force in system.getForces():
                    if isinstance(force, NonbondedForce):
                        force.setForceGroup(0)
                        force.setUseDispersionCorrection(False)
                        
                        # Add global parameters for scaling
                        force.addGlobalParameter("res_i_coulomb_scale", 1)
                        force.addGlobalParameter("res_i_lj_scale", 1)
                        force.addGlobalParameter("res_j_coulomb_scale", 1)
                        force.addGlobalParameter("res_j_lj_scale", 1)
                        
                        # Set up parameter offsets for each particle
                        for atom_idx in range(force.getNumParticles()):
                            charge, sigma, epsilon = force.getParticleParameters(atom_idx)
                            
                            # Zero out all parameters first
                            force.setParticleParameters(atom_idx, 0, 0, 0)
                            
                            # Add parameter offsets based on which residue the atom belongs to
                            if atom_idx in atoms_i:
                                force.addParticleParameterOffset(
                                    "res_i_coulomb_scale", atom_idx, charge, 0*nanometer, 0*kilojoules_per_mole
                                )
                                force.addParticleParameterOffset(
                                    "res_i_lj_scale", atom_idx, 0*elementary_charge, sigma, epsilon
                                )
                            elif atom_idx in atoms_j:
                                force.addParticleParameterOffset(
                                    "res_j_coulomb_scale", atom_idx, charge, 0*nanometer, 0*kilojoules_per_mole
                                )
                                force.addParticleParameterOffset(
                                    "res_j_lj_scale", atom_idx, 0*elementary_charge, sigma, epsilon
                                )
                            # Other atoms (water, ions, other residues) are zeroed out
                        
                        # Zero out all exceptions (intra-residue interactions)
                        for exc_idx in range(force.getNumExceptions()):
                            p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(exc_idx)
                            force.setExceptionParameters(exc_idx, p1, p2, 0, 0, 0)
                    else:
                        # Put other forces in a different group so they're not evaluated
                        force.setForceGroup(2)
                
                # Create context for energy evaluation
                integrator = VerletIntegrator(0.001 * picoseconds)
                platform = self._get_platform()
                context = Context(system, integrator, platform)
                context.setPositions(self.positions)
                
                # Define energy calculation helper
                def get_energy(res_i_coulomb, res_i_lj, res_j_coulomb, res_j_lj):
                    context.setParameter("res_i_coulomb_scale", res_i_coulomb)
                    context.setParameter("res_i_lj_scale", res_i_lj)
                    context.setParameter("res_j_coulomb_scale", res_j_coulomb)
                    context.setParameter("res_j_lj_scale", res_j_lj)
                    return context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
                
                # Calculate Coulomb interaction energy
                # E_interaction = E_total(i+j) - E_internal(i) - E_internal(j)
                total_coulomb = get_energy(1, 0, 1, 0)
                res_i_coulomb = get_energy(1, 0, 0, 0)
                res_j_coulomb = get_energy(0, 0, 1, 0)
                coulomb_interaction = total_coulomb - res_i_coulomb - res_j_coulomb
                
                # Calculate LJ interaction energy
                total_lj = get_energy(0, 1, 0, 1)
                res_i_lj = get_energy(0, 1, 0, 0)
                res_j_lj = get_energy(0, 0, 0, 1)
                lj_interaction = total_lj - res_i_lj - res_j_lj
                
                # Store results (convert to float, remove units)
                lj_interaction_energy = lj_interaction.value_in_unit(kilojoules_per_mole)
                coulomb_interaction_energy = coulomb_interaction.value_in_unit(kilojoules_per_mole)

                if lj_interaction_energy < 0:
                    vdw_energy_attractive[i, j] = lj_interaction_energy
                else:
                    vdw_energy_repulsive[i, j] = lj_interaction_energy

                if coulomb_interaction_energy < 0:
                    es_energy_attractive[i, j] = coulomb_interaction_energy
                else:
                    es_energy_repulsive[i, j] = coulomb_interaction_energy
                
                # pair_count += 1
                # if pair_count % 100 == 0:
                #     print(f"  Processed {pair_count}/{total_pairs} residue pairs...")
                
                # Clean up context
                del context
                del integrator
        
        print(f"Completed pairwise energy calculations for {total_pairs} pairs.")
        
        return (vdw_energy_attractive, vdw_energy_repulsive, es_energy_attractive, es_energy_repulsive)
    
    def run_full_pipeline(
        self,
        npt_steps: int = DEFAULT_NPT_STEPS,
        nvt_steps: int = DEFAULT_NVT_STEPS,
        production_steps: int = DEFAULT_PRODUCTION_STEPS,
        energy_calc_interval: int = 10000,
        return_simulated_pdb: bool = False,
        subtract_solvent_energies: bool = True,
        debug: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray] | \
         tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray, io.StringIO]:
        """Run the complete MD simulation pipeline.

        This is a convenience method that runs all steps:
            1. System setup (fix PDB, solvate, ionize)
            2. Energy minimization
            3. NPT equilibration
            4. NVT equilibration
            5. Production MD with energy and force calculations

        Args:
            npt_steps (int): Number of NPT equilibration steps.
            nvt_steps (int): Number of NVT equilibration steps.
            production_steps (int): Number of production MD steps.
            energy_calc_interval (int): Interval for calculating energies/forces.
            return_simulated_pdb (bool): If True, return the final production
                structure as a PDB stream. Defaults to False.
            debug (bool): If True, print additional information during the pipeline
                such as energy statistics and graphs.

        Returns:
            tuple[np.ndarray, ...]: Eight NxN matrices (or nine items if
                return_simulated_pdb is True):
                - vdw_attractive: Attractive VdW energies (kJ/mol)
                - vdw_repulsive: Repulsive VdW energies (kJ/mol)
                - es_attractive: Attractive electrostatic energies (kJ/mol)
                - es_repulsive: Repulsive electrostatic energies (kJ/mol)
                - production_pdb (io.StringIO): Final production structure as PDB
                    stream (only if return_simulated_pdb=True)
        """
        print("=" * 60)
        print("Starting full MD simulation pipeline")
        print("=" * 60)
        
        # Set debug mode for energy logging and plotting
        self.debug = debug
        
        # Step 1: Setup system
        print("\n[Step 1/5] Setting up system...")
        self.setup_system()
        
        # Step 2: Energy minimization of system
        print("\n[Step 2/5] Energy minimization...")
        self._create_new_simulation(
            add_barostat=False, add_calpha_restraint=True)
        self.minimize_energy()
        
        # Step 3: NPT equilibration with new system including barostat force
        print("\n[Step 3/5] NPT equilibration...")
        self.equilibrate_npt(steps=npt_steps)
        
        # Step 4: NVT equilibration using same system without barostat force
        print("\n[Step 4/5] NVT equilibration...")
        self.equilibrate_nvt_with_warming(steps=nvt_steps)
        
        # Step 5: Production MD using new system and simulation with energy calculations
        print("\n[Step 5/5] Production MD...")
        results = self.run_production(
            steps=production_steps,
            energy_calc_interval=energy_calc_interval,
            subtract_solvent=subtract_solvent_energies)
        
        # Capture production structure if requested
        production_pdb_stream = None
        if return_simulated_pdb:
            production_pdb_stream = self.get_simulated_pdb_stream()
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        
        if return_simulated_pdb:
            results.append(production_pdb_stream)
        return results
