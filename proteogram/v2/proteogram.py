"""Proteogram V2 module for protein structure analysis and visualization.

This module defines the ProteogramV2 class, which provides methods for generating proteogram maps from PDB files. The proteogram maps include distance, hydrophobicity,
Van der Waals, and electrostatic interaction maps. The class also integrates with the NonBondedForceModel module to perform molecular dynamics simulations for calculating the non-bonded interaction energies."""

import numpy as np
import warnings
import gc

from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
from Bio.PDB.Polypeptide import PPBuilder

from ..common.constants import HYDROPHOBICITY_LIST, RESIDUE_LIST
from .nonbonded_forces import NonBondedForceModel


# Ignore PDB construction warnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

class ProteogramV2:
    """Proteogram V2 class for generating protein structure maps.

    This class provides methods for calculating distance maps, hydrophobicity maps,
    and other structural features from PDB files.

    Attributes:
        pdb_path (str): Path to the PDB file.
        structure: Parsed PDB structure.
        model: First model from the PDB structure.
        chain: Selected chain from the model.
        allowed_amino_acids (dict): Mapping of residue names to single-letter codes.
        sequence (str): Amino acid sequence of the chain.
        calpha_atom_distance_cutoff (float): Cα distance cutoff for hydrophobicity map in Angstroms.
        sequence_len_lower_cutoff (int): Minimum sequence length for valid chains.
        sequence_len_upper_cutoff (float): Maximum sequence length for valid chains.
    """

    def __init__(self,
                 pdb_path,
                 output_dir,
                 chain_id,
                 calpha_atom_distance_cutoff=10,
                 sequence_len_lower_cutoff=20,
                 sequence_len_upper_cutoff=1e9,
                 use_gpu=False):
        """Initialize the ProteogramV2 instance.

        Args:
            pdb_path (str): Path to the PDB file.
            chain_id (str): Chain identifier to extract from the PDB file.
            calpha_atom_distance_cutoff (float, optional): Cα distance cutoff
                for hydrophobicity map in Angstroms. Defaults to 10.
            sequence_len_lower_cutoff (int, optional): Minimum sequence length
                for valid chains. Defaults to 20.
            sequence_len_upper_cutoff (float, optional): Maximum sequence length
                for valid chains. Defaults to 1e9.
            use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.

        Raises:
            KeyError: If the specified chain_id is not found in the PDB file.
        """
        self.pdb_path = pdb_path
        self.output_dir = output_dir
        parser = PDBParser()
        self.structure = parser.get_structure("protein_id", self.pdb_path)
        self.model = self.structure[0]
        try:
            self.chain = self.model[chain_id]
        except KeyError:
            raise KeyError(f"Chain ID {chain_id} not found in PDB file.")
        self.allowed_amino_acids = {b: a for a, b in RESIDUE_LIST}
        self.sequence = ''.join(
            [self.allowed_amino_acids[res.resname] for res in self.chain])
        self.calpha_atom_distance_cutoff = calpha_atom_distance_cutoff
        self.sequence_len_lower_cutoff = sequence_len_lower_cutoff
        self.sequence_len_upper_cutoff = sequence_len_upper_cutoff
        self.use_gpu = use_gpu
    
    def is_valid_chain(self):
        """Check if the chain meets the sequence length criteria.

        Returns:
            bool: True if the chain length is within the specified cutoffs,
                False otherwise.
        """
        seq_len = len(self.sequence)
        return (self.sequence_len_lower_cutoff <= seq_len <= self.sequence_len_upper_cutoff)
    
    def calculate_proteogram(self,
                             return_simulated_pdb: bool = False,
                             debug: bool = False,
                             subtract_solvent_energies: bool = True,
                             memory_efficient: bool = False):
        """Calculate the proteogram maps.

        Computes distance, hydrophobicity, Van der Waals, and electrostatic maps
        for the protein structure.

        Args:
            return_simulated_pdb (bool): If True, also return the final
                production structure from the MD simulation as a PDB stream.
                Defaults to False.
            debug (bool): If True, print debug information during calculations.
                Defaults to False.
            subtract_solvent_energies (bool): If True, subtract solvent-only 
                energies from the protein+solvent energies to isolate the protein 
                contributions. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray | None: The stacked proteogram array if
                    successful, None otherwise.
                - dict | None: Error dictionary if any errors occurred,
                    None otherwise.
                - io.StringIO | None: Production PDB structure stream
                    (only if return_simulated_pdb=True).
        """
        # Initialize the model
        model = NonBondedForceModel(
            pdb_path=self.pdb_path,
            output_dir=self.output_dir,
            temperature=310.15, # Kelvin (37 C)
            timestep=2.0, # Femtoseconds
            use_gpu=self.use_gpu, # Set True for GPU acceleration
            memory_efficient=memory_efficient
        )

        energy_calc_interval = 10000  # Default: Calculate energies every 20 ps
        if memory_efficient:
            energy_calc_interval = 50000  # Calculate energies every 100 ps

        # Run the full pipeline (recommended)
        pipeline_result = model.run_full_pipeline(
            npt_steps=50000,      # 100 ps NPT equilibration
            nvt_steps=50000,      # 100 ps NVT equilibration
            production_steps=500000,  # 1 ns production
            energy_calc_interval=energy_calc_interval,
            return_simulated_pdb=return_simulated_pdb,
            debug=debug,
            subtract_solvent_energies=subtract_solvent_energies # Subtract solvent-only energies
        )

        # Explicit clean-up of OpenMM resources after pipeline completion
        model.cleanup_all_resources(final_run=True)
        model._clear_cuda_cache()
        del model
        
        # Unpack results based on whether simulated PDB was requested
        if return_simulated_pdb:
            vdw_e_att, vdw_e_rep, es_e_att, es_e_rep, \
                disto_map, simulated_pdb = pipeline_result
        else:
            vdw_e_att, vdw_e_rep, es_e_att, es_e_rep, disto_map = pipeline_result
            simulated_pdb = None

        # Hydrophobicity map depends on the MD-derived distance matrix
        hydro_map = self.calc_hydrophobicity_map(self.sequence, disto_map)
        
        # Normalize all maps to [0-255]
        norm_disto_map, disto_err = self.normalize_map(disto_map)
        norm_hydro_map, hydro_err = self.normalize_map(hydro_map)
        norm_vdw_att_map, vdw_att_err = self.normalize_map(vdw_e_att)
        norm_vdw_rep_map, vdw_rep_err = self.normalize_map(vdw_e_rep)
        norm_es_att_map, es_att_err = self.normalize_map(es_e_att)
        norm_es_rep_map, es_rep_err = self.normalize_map(es_e_rep)
        
        # Clear the original energy maps to save memory
        del disto_map, hydro_map, vdw_e_att, vdw_e_rep, es_e_att, es_e_rep
        del pipeline_result
        gc.collect()  # Force garbage collection after deleting large arrays
        
        # Check for normalization errors
        errors = {
            'distance': disto_err,
            'hydrophobicity': hydro_err,
            'vdw_attractive': vdw_att_err,
            'electronic_repulsive': vdw_rep_err,
            'electrostatic_attractive': es_att_err,
            'electrostatic_repulsive': es_rep_err
        }
        # Filter to only include actual errors
        errors = {k: v for k, v in errors.items() if v}
                
        # Create upper and lower triangle
        try:
            final_upper = np.dstack(
                [norm_vdw_att_map, norm_vdw_rep_map, norm_disto_map])
            final_lower = np.rot90(np.dstack(
                [norm_es_att_map, norm_es_rep_map, norm_hydro_map]),
                2
            )
            final_data = final_upper + final_lower
            # Clear intermediate arrays
            del norm_disto_map, norm_hydro_map, norm_vdw_att_map, norm_vdw_rep_map
            del norm_es_att_map, norm_es_rep_map, final_upper, final_lower
            gc.collect()  # Force garbage collection after large array operations
            if return_simulated_pdb:
                return final_data, None, simulated_pdb
            return final_data, None
        except Exception as e:
            gc.collect()  # Force garbage collection even on error
            if return_simulated_pdb:
                return None, {'Error stacking maps': str(e)}, simulated_pdb
            return None, {'Error stacking maps': str(e)}

    @staticmethod
    def normalize_map(arr):
        """Normalize any numpy array to [0-255] using Min-Max linear scaling.

        Args:
            arr (numpy.ndarray): Input array to normalize.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Normalized array with values in range [0, 255].
                - str: Error message if normalization failed, empty string otherwise.
        """
        err = ''
        try:
            arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        except Exception as e:
            err = f'Problem normalizing map: {e}'
        return arr, err
        
    def calc_dist_matrix(self):
        """Calculate the C-alpha distance matrix for the chain.

        Computes pairwise distances between all C-alpha atoms in the chain.
        Only the upper triangle of the matrix is populated; the lower triangle
        contains zeros.

        Returns:
            numpy.ndarray: A symmetric matrix of shape (n_residues, n_residues)
                containing C-alpha distances in Angstroms.
        """
        ca_atoms = [res["CA"] for res in self.chain if "CA" in res]
        n_residues = len(ca_atoms)
        # Initialize a results matrix with zeros
        distogram = np.zeros((n_residues, n_residues), dtype=np.float64)
        
        # Assign upper triangle the c-alpha distances (lower triangle remains all 0)
        for i in range(n_residues):
            for j in range(i + 1, n_residues): # Only iterate over unique pairs (upper triangle)
                # Use the distance operator overload for Atom objects
                distance = ca_atoms[i] - ca_atoms[j]
                distogram[i, j] = distance
                
        return distogram

    def calc_hydrophobicity_map(self, sequence, disto_map):
        """Calculate the hydrophobicity difference map.

        Computes the absolute difference in hydrophobicity values between
        residue pairs that are within the atom distance cutoff.

        Args:
            sequence (str): Amino acid sequence of the protein.
            disto_map (numpy.ndarray): Distance matrix from calc_dist_matrix.

        Returns:
            numpy.ndarray: A matrix of shape (len(sequence), len(sequence))
                containing hydrophobicity delta values for residue pairs
                within the distance cutoff.
        """
        hydro_map = np.zeros((len(sequence), len(sequence)))

        for row in range(len(sequence)):
            for col in range(row+1, len(sequence)):
                # If residues less than cutoff num of Angstroms
                if disto_map[row,col] < self.calpha_atom_distance_cutoff:
                    try:
                        row_val = np.abs(HYDROPHOBICITY_LIST[sequence[row]])
                        col_val = np.abs(HYDROPHOBICITY_LIST[sequence[col]])
                        delta = np.abs(row_val - col_val)
                    # Throw exception when can't retrieve hydrophobicity
                    except:
                        delta = 0
                    hydro_map[row,col] = delta
                    # If on the diag, set to 0
                    if row == col:
                        hydro_map[row,col] = 0

        return hydro_map

    def set_sequence(self):
        """Set the protein sequence from the PDB structure.

        Uses Biopython's PPBuilder to extract the amino acid sequence
        from the structure and stores it in self.sequence.
        """
        ppb = PPBuilder()
        seq = ''
        for pp in ppb.build_peptides(self.structure):
            seq += pp.get_sequence()
        self.sequence = seq
        