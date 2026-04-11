import numpy as np
import warnings

import pyrotein as pr
from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
from Bio.PDB.Polypeptide import PPBuilder

from ..common.constants import (
    HYDROPHOBICITY_LIST_BINARY,
    HYDROPHOBICITY_LIST,
    CHARGE_LIST
)

# Ignore PDB construction warnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

class Proteogram:

    def __init__(self,
                 pdb_path,
                 atom_distance_cutoff=15,
                 hydrophobicity_delta_cutoff=20,
                 sequence_len_lower_cutoff=20,
                 sequence_len_upper_cutoff=1e9):
        self.pdb_path = pdb_path
        self.atom_distance_cutoff = atom_distance_cutoff
        self.hydrophobicity_delta_cutoff = hydrophobicity_delta_cutoff
        self.sequence_len_lower_cutoff = sequence_len_lower_cutoff
        self.sequence_len_upper_cutoff = sequence_len_upper_cutoff

    def calc_distogram(self, sequence, chain):
        nterm = 1
        cterm = len(sequence)
        
        # Define atoms and chain used for distance matrix analysis
        backbone = ["CA"]

        # Read coordinates from a PDB file
        atoms_pdb = pr.atom.read(self.pdb_path)
        
        # Create a lookup table for this pdb
        atom_dict = pr.atom.create_lookup_table(atoms_pdb)
        
        # Obtain the chain to process
        chain_dict = atom_dict[chain]
        
        # Obtain coordinates
        xyzs = pr.atom.extract_xyz_by_atom(backbone, chain_dict, nterm, cterm)
        
        # Calculate distance matrix
        dmat = pr.distance.calc_dmat(xyzs, xyzs)
        tri_lower_diag = np.tril(dmat, k=0)
        
        # Assign upper triangle same values in symmetry
        tri_upper_diag = np.rot90(tri_lower_diag, 2)
        tri_diag = tri_lower_diag + tri_upper_diag

        tri_diag = np.nan_to_num(tri_diag)

        return tri_diag

    def calc_simple_polarity_map(self, sequence, disto_map):
        polarity_map = np.zeros((len(sequence), len(sequence)))

        for row in range(len(sequence)):
            for col in range(len(sequence)):
                # If less residues less than cutoff num of Angstroms
                if disto_map[row,col] < self.atom_distance_cutoff:
                    row_val = HYDROPHOBICITY_LIST_BINARY[sequence[row]]
                    col_val = HYDROPHOBICITY_LIST_BINARY[sequence[col]]
                    if row_val == col_val:
                        polarity_map[row,col] = 1
                        polarity_map[col,row] = 1
                    # If on the diag, set to 0
                    if row == col:
                        polarity_map[row,col] = 0
        return polarity_map

    def calc_hydrophobicity_map(self, sequence, disto_map):
        hydro_map = np.zeros((len(sequence), len(sequence)))

        for row in range(len(sequence)):
            for col in range(len(sequence)):
                # If less residues less than cutoff num of Angstroms
                if disto_map[row,col] < self.atom_distance_cutoff:
                    row_val = np.abs(HYDROPHOBICITY_LIST[sequence[row]])
                    col_val = np.abs(HYDROPHOBICITY_LIST[sequence[col]])
                    delta = np.abs(row_val - col_val)
                    hydro_map[row,col] = delta
                    hydro_map[col,row] = delta
                    # If on the diag, set to 0
                    if row == col:
                        hydro_map[row,col] = 0
        return hydro_map

    def calc_charge_map(self, sequence, disto_map):
        charge_map = np.zeros((len(sequence), len(sequence)))

        for row in range(len(sequence)):
            for col in range(len(sequence)):
                # If less residues less than cutoff num of Angstroms
                if disto_map[row,col] < self.atom_distance_cutoff:
                    row_val = CHARGE_LIST[sequence[row]]
                    col_val = CHARGE_LIST[sequence[col]]
                    if row_val == -1 and col_val == 1:
                        charge_map[row,col] =  1
                        charge_map[col,row] =  1
                    if row_val == 1 and col_val == -1:
                        charge_map[row,col] =  1
                        charge_map[col,row] =  1
                    if row == col:
                        charge_map[row,col] = 0
        return charge_map

    def stack_data(self, distance_map, hydro_map, charge_map):
        # normalize maps
        if np.max(distance_map) != np.min(distance_map):
            distance_map = distance_map / (np.max(distance_map) - np.min(distance_map))
        if np.max(hydro_map) != np.min(hydro_map):
            hydro_map = hydro_map / (np.max(hydro_map) - np.min(hydro_map))
        final_data = np.dstack([distance_map, hydro_map, charge_map])
        return final_data

    def get_sequence(self):
        """Get a protein sequence from a PDB file"""
        p = PDBParser(PERMISSIVE=0)
        structure = p.get_structure('xyz', self.pdb_path)
        ppb = PPBuilder()
        seq = ''
        for pp in ppb.build_peptides(structure):
            seq += pp.get_sequence()
        return seq
