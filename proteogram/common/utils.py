from .constants import RESIDUE_LIST, UNKNOWN_RESIDUE
import glob
import yaml
import os

from Bio import PDB


PARSER = PDB.PDBParser(PERMISSIVE=1)

# Proteogram image formats accepted on input. png is create_v2_proteograms.py's
# default and what the per-channel ablation sets are written in, since JPEG
# chroma subsampling erases much of the difference between channels. jpg is
# still accepted so datasets generated before the switch keep working.
PROTEOGRAM_IMAGE_EXTENSIONS = ('.png', '.jpg')


def find_proteogram_images(directory):
    """Return a sorted list of proteogram image paths directly under directory.

    Accepts both .png and .jpg so datasets generated before png became the
    default are still consumed by the same training and similarity scripts.
    """
    return sorted(
        path
        for ext in PROTEOGRAM_IMAGE_EXTENSIONS
        for path in glob.glob(os.path.join(directory, f'*{ext}'))
    )

def get_3letter_res_name(res_code):
    lookup = dict(RESIDUE_LIST)
    return lookup.get(res_code, UNKNOWN_RESIDUE[1])

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
def split_by_chain_and_save(pdb_file, chain_id, scope_sid,  scope_pdbs_dir):
    """
    Parse the single PDB file downloaded with biopython into it's chains and
    save with the scope sid name in the scope_pdbs_dir
    """
    structure = PARSER.get_structure(pdb_file, pdb_file)
    writer = PDB.PDBIO()
    filename = os.path.join(scope_pdbs_dir, f'{scope_sid}.ent')
    writer.set_structure(structure)
    writer.save(filename, ChainSelect(chain_id))

class ChainSelect(PDB.Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0