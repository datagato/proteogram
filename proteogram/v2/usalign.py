"""Structure helpers for checking similarity maps against structural alignment.

Proteogram row k is the k-th residue of the domain's first chain that is a
standard (or mapped modified) amino acid with a C-alpha atom, the same rule
ProteogramV2 uses to build the image. ``proteogram_residues`` applies that rule
to a SCOPe .ent file, and ``write_indexed_pdb`` writes exactly those residues
renumbered 1..L (modified residues renamed to their standard parent, all as
ATOM records). USalign run on these files therefore reports residue k as
proteogram row k-1, with no residue-numbering bookkeeping.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from dataclasses import dataclass, field

import numpy as np
from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning

from ..common.constants import RESIDUE_LIST, MODIFIED_RESIDUES_TO_STANDARD

warnings.filterwarnings('ignore', category=PDBConstructionWarning)

_THREE_TO_ONE = {three: one for one, three in RESIDUE_LIST}
_ALLOWED = dict(_THREE_TO_ONE)
_ALLOWED.update({mod: _THREE_TO_ONE[std] for mod, std in MODIFIED_RESIDUES_TO_STANDARD.items()
                 if std in _THREE_TO_ONE})


@dataclass
class ProteogramResidues:
    path: str
    residues: list            # Bio.PDB residues in proteogram order
    sequence: str
    ca: np.ndarray            # (L, 3) C-alpha coordinates

    def __len__(self):
        return len(self.residues)

    def resid_labels(self) -> list[str]:
        """Original chain residue numbers (with insertion codes), for reporting."""
        return [f'{r.id[1]}{r.id[2].strip()}' for r in self.residues]


def proteogram_residues(structure_path: str) -> ProteogramResidues:
    """Residues of the first chain in proteogram row order (see module docstring)."""
    s = PDBParser(QUIET=True).get_structure('x', structure_path)
    chain = next(s[0].get_chains())
    res = [r for r in chain if r.resname in _ALLOWED and 'CA' in r]
    seq = ''.join(_ALLOWED[r.resname] for r in res)
    ca = np.array([r['CA'].get_coord() for r in res], dtype=float)
    return ProteogramResidues(structure_path, res, seq, ca)


def write_indexed_pdb(pres: ProteogramResidues, path: str, bfactors=None,
                      rotation=None, chain_id: str = 'A') -> None:
    """Write the proteogram residues renumbered 1..L as ATOM records.

    bfactors: optional (L,) values for the B-factor column (e.g. scores).
    rotation: optional (t, U) from USalign; coordinates become t + U @ x.
    """
    lines, serial = [], 1
    for k, r in enumerate(pres.residues):
        name = MODIFIED_RESIDUES_TO_STANDARD.get(r.resname, r.resname)
        b = 0.0 if bfactors is None else float(bfactors[k])
        for atom in r:
            if atom.element == 'H':
                continue
            x = atom.get_coord().astype(float)
            if rotation is not None:
                t, U = rotation
                x = t + U @ x
            el = (atom.element or atom.get_id()[0]).strip()
            lines.append(f'ATOM  {serial:5d} {atom.get_fullname():4s} {name:>3s} {chain_id}'
                         f'{k + 1:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{1.0:6.2f}{b:6.2f}'
                         f'          {el:>2s}')
            serial += 1
    lines += ['TER', 'END']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def contact_map(ca: np.ndarray, cutoff: float = 8.0, min_sep: int = 3) -> np.ndarray:
    """Boolean (L, L) C-alpha contacts, i < j only, |i - j| >= min_sep."""
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    L = len(ca)
    i, j = np.triu_indices(L, min_sep)
    out = np.zeros((L, L), dtype=bool)
    out[i, j] = d[i, j] < cutoff
    return out


# ---------------------------------------------------------------------------
# USalign
# ---------------------------------------------------------------------------

@dataclass
class Alignment:
    tm1: float                  # TM-score normalised by structure 1 (query)
    tm2: float                  # TM-score normalised by structure 2 (target)
    rmsd: float
    aligned_length: int
    seq_id: float
    partner: np.ndarray         # (L1,) target row aligned to each query row, -1 if none
    close: np.ndarray           # (L1,) bool, aligned pair within 5 A (":" in USalign)
    rotation: tuple | None = None   # (t (3,), U (3, 3)) taking structure 1 onto structure 2
    raw: str = field(default='', repr=False)

    def pairs(self, close_only: bool = True) -> np.ndarray:
        """(n, 2) array of aligned (query row, target row) pairs."""
        keep = (self.partner >= 0) & (self.close if close_only else True)
        idx = np.nonzero(keep)[0]
        return np.stack([idx, self.partner[idx]], axis=1)


def parse_usalign(text: str, L1: int | None = None, L2: int | None = None) -> Alignment:
    """Parse USalign/TM-align default text output (alignment block included)."""
    def num(pattern, cast=float, default=float('nan')):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    tm = re.findall(r'TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_([12])', text)
    tms = {k: float(v) for v, k in tm}
    rmsd = num(r'RMSD=\s*([\d.]+)')
    alen = num(r'Aligned length=\s*(\d+)', int, -1)
    sid = num(r'Seq_ID=n_identical/n_aligned=\s*([\d.]+)')

    lines = text.splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.startswith('(":" denotes')), None)
    if start is None or start + 3 >= len(lines):
        raise ValueError('No alignment block in USalign output.')
    s1, mk, s2 = lines[start + 1], lines[start + 2], lines[start + 3]
    n = max(len(s1), len(s2))
    s1, mk, s2 = s1.ljust(n, '-'), mk.ljust(n), s2.ljust(n, '-')
    n1 = sum(c != '-' for c in s1)
    n2 = sum(c != '-' for c in s2)
    if L1 is not None and n1 != L1:
        raise ValueError(f'USalign read {n1} residues of structure 1, expected {L1}.')
    if L2 is not None and n2 != L2:
        raise ValueError(f'USalign read {n2} residues of structure 2, expected {L2}.')
    partner = -np.ones(n1, dtype=int)
    close = np.zeros(n1, dtype=bool)
    i = j = 0
    for a, m, b in zip(s1, mk, s2):
        if a != '-' and b != '-':
            partner[i] = j
            close[i] = m == ':'
        i += a != '-'
        j += b != '-'
    return Alignment(tms.get('1', float('nan')), tms.get('2', float('nan')), rmsd, alen, sid,
                     partner, close, raw=text)


def parse_matrix(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the -m rotation matrix file: rows 'm t[m] u[m][0] u[m][1] u[m][2]'."""
    rows = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) == 5 and parts[0] in ('0', '1', '2'):
            try:
                rows[int(parts[0])] = [float(x) for x in parts[1:]]
            except ValueError:
                continue
    if len(rows) != 3:
        raise ValueError('Could not read the 3x4 rotation matrix from USalign -m output.')
    t = np.array([rows[m][0] for m in range(3)])
    U = np.array([rows[m][1:] for m in range(3)])
    return t, U


def run_usalign(exe: str, pdb1: str, pdb2: str, workdir: str, L1: int | None = None,
                L2: int | None = None, timeout: int = 300) -> Alignment:
    """Align pdb1 (query) onto pdb2 (target) with USalign; keep output and matrix in workdir."""
    os.makedirs(workdir, exist_ok=True)
    mat = os.path.join(workdir, 'usalign_matrix.txt')
    proc = subprocess.run([exe, pdb1, pdb2, '-m', mat], capture_output=True, text=True,
                          timeout=timeout)
    with open(os.path.join(workdir, 'usalign.out'), 'w') as f:
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f'USalign failed ({proc.returncode}): {proc.stderr.strip()[:300]}')
    aln = parse_usalign(proc.stdout, L1, L2)
    if os.path.exists(mat):
        with open(mat) as f:
            aln.rotation = parse_matrix(f.read())
    return aln
