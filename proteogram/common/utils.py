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


def find_proteogram_images(directory, recursive=False):
    """Return a sorted list of proteogram image paths under directory.

    Accepts both .png and .jpg so datasets generated before png became the
    default are still consumed by the same training, similarity and
    corpus-embedding scripts.

    Extensions are matched case-insensitively. Globbing a wildcard pattern is
    case-sensitive even where the filesystem is not (os.path.normcase is a
    no-op on posix), so a dataset written as .PNG would otherwise be visible
    to some callers and invisible to others. Matching on the lowercased
    extension instead keeps every caller agreeing, and cannot double-count a
    file the way globbing each case variant separately could.

    Args:
        directory: Directory to search.
        recursive: If True, descend into subdirectories. Corpora are often
            laid out in per-class subfolders; training and similarity read a
            single flat directory, so this defaults to False.

    Returns:
        Sorted list of paths. Empty if directory holds no proteogram images.
    """
    pattern = os.path.join(directory, '**', '*') if recursive \
        else os.path.join(directory, '*')
    return sorted(
        path
        for path in glob.glob(pattern, recursive=recursive)
        if os.path.splitext(path)[1].lower() in PROTEOGRAM_IMAGE_EXTENSIONS
        and os.path.isfile(path)
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