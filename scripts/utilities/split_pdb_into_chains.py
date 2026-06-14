"""
Split downloaded PDB files into per-chain monomer files.
Outputs {PDBID}_{CHAIN}.pdb files to pdb_monomers_dir.

The chain ID encoded in the filename flows downstream to:
  - create_pdb_annotation_file.py  (chain-specific GO lookup)
  - create_v2_proteograms.py       (chain_id = bname[5] still works since
                                    positions 0-3 = PDB ID, 4 = '_', 5 = chain)
"""
import glob
import os
import warnings

from Bio.PDB import PDBIO, MMCIFParser, PDBParser, Select
from Bio.PDB.PDBParser import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa

from proteogram.common import read_yaml


warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class ProteinChainSelect(Select):
    """Keep only standard amino acid residues from one chain."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        return is_aa(residue, standard=True)


def is_protein_chain(chain):
    return any(is_aa(res, standard=True) for res in chain)


def pdb_id_from_filename(basename):
    """Handle RCSB naming conventions: pdb5wsu.ent → 5WSU, 5wsu.pdb → 5WSU."""
    noext = basename.rsplit('.', 1)[0].lower()
    if noext.startswith('pdb') and len(noext) == 7:
        return noext[3:].upper()
    return noext[:4].upper()


if __name__ == '__main__':
    config = read_yaml('config.yml')
    pdb_download_dir = config['pdb_download_dir']
    pdb_monomers_dir = config['pdb_monomers_dir']

    os.makedirs(pdb_monomers_dir, exist_ok=True)

    pdb_parser = PDBParser(PERMISSIVE=1)
    cif_parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    pdb_files = []
    for ext in ('*.pdb', '*.ent', '*.cif', '*.mmcif'):
        pdb_files.extend(
            glob.glob(os.path.join(pdb_download_dir, '**', ext), recursive=True))

    print(f'Found {len(pdb_files)} structure files to split.')
    written, skipped = 0, 0

    for pdb_file in pdb_files:
        basename = os.path.basename(pdb_file)
        ext = basename.rsplit('.', 1)[-1].lower()
        pdb_id = pdb_id_from_filename(basename)

        try:
            parser = cif_parser if ext in ('cif', 'mmcif') else pdb_parser
            structure = parser.get_structure(pdb_id, pdb_file)
        except Exception as e:
            print(f'Failed to parse {pdb_file}: {e}')
            skipped += 1
            continue

        model = next(iter(structure))
        io.set_structure(structure)

        for chain in model:
            chain_id = chain.get_id().strip()
            if not chain_id or not is_protein_chain(chain):
                continue
            out_file = os.path.join(pdb_monomers_dir, f'{pdb_id}_{chain_id}.pdb')
            if os.path.exists(out_file):
                continue
            try:
                io.save(out_file, ProteinChainSelect(chain_id))
                written += 1
            except Exception as e:
                print(f'Failed to write {out_file}: {e}')
                skipped += 1

    print(f'Written {written} monomer files, skipped {skipped}.')
