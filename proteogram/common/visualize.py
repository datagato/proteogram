import nglview as nv
from Bio.PDB import PDBParser
from Bio import SeqIO
from ase import Atom, Atoms
import matplotlib.pyplot as plt

from .constants import BACKBONE_POSITIONS
from .utils import get_3letter_res_name


def plot_maps(data, sequence, color_bar=True, filename=None):
    """
    Plot a NxN matrix of values where each value is a numerical
    value associated with a residue pair such as a distogram.
    """
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(7, 7)
    # fig.set_dpi(300)
    # img = ax.imshow(data)

    # X-tick labels (sequence)
    ax.set_xticks(range(len(sequence)))
    ax.tick_params()
    x_label_list=list(sequence)
    ax.set_xticklabels(x_label_list)

    # X-tick labels (sequence)
    ax.set_yticks(range(len(sequence)))
    ax.tick_params()
    ax.set_yticklabels(x_label_list, rotation=90)

    # Colorbar
    if color_bar == True:
        cbar = fig.colorbar(img, ax=ax)

    if filename:
        # Visualize a few distance matrix
        fl_dmat = os.path.join(filename)
        #img = data.astype('uint8')
        img = data.astype(float)
        plt.imsave(fl_dmat, img)

def draw_atoms_ngl(pdb_filename):
    """
    PDB protein structure visualization of backbone atoms using
    NGLView.
    
    Arguments
    ---------
    pdb_filename : str
        PDB protein structure file

    Returns
    -------
    view : nglview.widget.NGLWidget
    """
    pdb_parser = PDBParser()
    pdb_structure = pdb_parser.get_structure("pdb_struct", pdb_filename)
    query_seqres = SeqIO.parse(pdb_filename, 'pdb-seqres')
    
    pdb_sequence = []
    for chain in query_seqres:
        pdb_sequence.extend(chain.seq)
    
    pdb_atoms = pdb_structure.get_atoms()
    
    formula = ""
    positions = []
    for i, r in enumerate(pdb_sequence):
        res_name = get_3letter_res_name(r)
        for meta, at in zip(BACKBONE_POSITIONS[res_name], pdb_atoms):
            symbol = meta[0] if meta[0] != "CA" else "C"
            positions.append(at.coord)
            formula += symbol
    view = nv.show_ase(Atoms(formula, positions=positions))
    return view