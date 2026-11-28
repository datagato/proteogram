"""Tests for the exact similarity split, its mapping back to residues, and USalign parsing."""
import numpy as np
import torch

from proteogram.v2 import similarity_maps as sm
from proteogram.v2 import usalign as ua


def test_decomposition_sums_to_cosine():
    torch.manual_seed(0)
    fq, ft = torch.relu(torch.randn(512, 16, 16)), torch.relu(torch.randn(512, 16, 16))
    mu = torch.relu(torch.randn(512))
    d = sm.decompose(fq, ft, center=mu)
    cos = torch.nn.functional.cosine_similarity(fq.mean((1, 2)), ft.mean((1, 2)), dim=0).item()
    assert abs(d.pair.sum() - d.cos) < 1e-12        # exact in float64
    assert abs(d.cos - cos) < 1e-6                   # reference computed in float32
    assert np.allclose(d.query_cells.ravel(), d.pair.sum(1))
    assert np.allclose(d.target_cells.ravel(), d.pair.sum(0))
    q, t = fq.mean((1, 2)).double(), ft.mean((1, 2)).double()
    assert abs(d.query_centered.sum() - d.centered_total) < 1e-9
    assert abs(d.centered_total - float(q @ (t - mu.double()) / (q.norm() * t.norm()))) < 1e-6
    assert abs((d.lift * np.outer(d.query_cells.ravel(), d.target_cells.ravel()) / d.cos).sum()
               - d.pair.sum()) < 1e-9


def test_cells_to_native_preserves_mass_resize_and_pad():
    cells = np.random.default_rng(0).random((16, 16))
    nat = sm.cells_to_native(cells, sm.Geometry(L=137, grid=500, resize=True))
    assert nat.shape == (137, 137)
    assert abs(nat.sum() - cells.sum()) < 1e-9
    # pad mode: a 100-residue image centred in a 200 grid keeps only the mass over the image
    g = sm.Geometry(L=100, grid=200, resize=False, offset=50)
    nat = sm.cells_to_native(np.ones((7, 7)), g)
    assert nat.shape == (100, 100)
    assert 0.2 < nat.sum() / 49 < 0.3       # ~ (100/200)^2 of the mass


def test_lower_triangle_is_rotated():
    L = 10
    nat = np.zeros((L, L))
    nat[7, 2] = 1.0                        # lower pixel (7, 2) -> residues (2, 7)
    res = sm.native_to_residues(nat)
    assert res.lower[2, 7] == 1.0 and res.upper.sum() == 0
    assert res.score_lower[2] == 0.5 and res.score_lower[7] == 0.5
    nat = np.zeros((L, L))
    nat[1, 4] = 2.0                        # upper pixel (1, 4) -> residues (1, 4)
    res = sm.native_to_residues(nat)
    assert res.upper[1, 4] == 2.0 and res.score_upper[1] == 1.0
    nat = np.random.default_rng(1).random((L, L))
    res = sm.native_to_residues(nat)
    assert abs(res.score.sum() - nat.sum()) < 1e-9


def test_cell_regions_halves():
    g = sm.Geometry(L=160, grid=500, resize=True)
    regs = sm.cell_regions((16, 16), g)
    up, lo, dg = regs[0 * 16 + 15], regs[15 * 16 + 0], regs[5 * 16 + 5]
    assert (up.half, lo.half, dg.half) == ('upper', 'lower', 'diagonal')
    assert up.residues[:10].all() and up.residues[150:].all() and not up.residues[20:140].any()
    # lower cell rows 150-160, cols 0-10 -> residues 0-10 and 150-160 after the flip
    assert lo.residues[:10].all() and lo.residues[150:].all()


USALIGN_OUT = """
Name of Structure_1: q.pdb:A (to be superimposed onto Structure_2)
Name of Structure_2: t.pdb:A
Length of Structure_1: 6 residues
Length of Structure_2: 5 residues

Aligned length=    4, RMSD=   1.23, Seq_ID=n_identical/n_aligned= 0.500
TM-score= 0.61000 (normalized by length of Structure_1: L=6, d0=0.50)
TM-score= 0.70000 (normalized by length of Structure_2: L=5, d0=0.50)
(You should use TM-score normalized by length of the reference structure)

(":" denotes residue pairs of d <  5.0 Angstrom, "." denotes other aligned residues)
MKV-LAG
 ::  .:
-KVAL-G
"""


def test_parse_usalign():
    a = ua.parse_usalign(USALIGN_OUT, 6, 5)
    assert (a.tm1, a.tm2, a.rmsd, a.aligned_length) == (0.61, 0.70, 1.23, 4)
    assert a.partner.tolist() == [-1, 0, 1, 3, -1, 4]
    assert a.close.tolist() == [False, True, True, False, False, True]
    assert a.pairs().tolist() == [[1, 0], [2, 1], [5, 4]]


def test_parse_matrix():
    text = """------ The rotation matrix to rotate Structure_1 to Structure_2 ------
m               t[m]        u[m][0]        u[m][1]        u[m][2]
0       1.0000000000   1.0000000000   0.0000000000   0.0000000000
1       2.0000000000   0.0000000000   1.0000000000   0.0000000000
2       3.0000000000   0.0000000000   0.0000000000   1.0000000000
"""
    t, U = ua.parse_matrix(text)
    assert t.tolist() == [1, 2, 3] and np.allclose(U, np.eye(3))


def test_mask_channels_halves():
    nat = np.full((6, 6, 3), 200, dtype=np.uint8)
    out = sm.mask_channels(nat, [False, True, True, True, True, False])   # drop VdW att + hydrophobicity
    assert (out[0, 3, 0] == 0) and (out[2, 2, 0] == 0)    # upper R (diagonal counts as upper)
    assert out[3, 0, 0] == 200                             # lower R kept (elec attractive)
    assert out[4, 1, 2] == 0 and out[1, 4, 2] == 200       # lower B dropped, upper B kept
    assert (out[..., 1] == 200).all()


def test_channel_split_is_additive():
    torch.manual_seed(0)
    import torchvision.models as tv
    model = tv.resnet18(weights=None).eval()
    rng = np.random.default_rng(0)
    nat_q = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    nat_t = rng.integers(0, 255, (44, 44, 3), dtype=np.uint8)
    with torch.no_grad():
        fq = sm.layer4_features(model, sm.preprocess_array(nat_q, 64, True))[0]
        ft = sm.layer4_features(model, sm.preprocess_array(nat_t, 64, True))[0]
    d = sm.decompose(fq, ft)
    for side, nat, partner in (('query', nat_q, ft), ('target', nat_t, fq)):
        cs = sm.channel_split(model, nat, partner, side, 64, True)
        assert np.abs(cs.full - d.pair).max() < 1e-6                  # all channels = the usual split
        assert np.abs(cs.phi.sum(0) - (cs.full - cs.empty)).max() < 1e-12
        assert abs(cs.totals.sum() - (cs.values[-1] - cs.values[0])) < 1e-9


def test_channels_present_validates_names():
    assert sm.channels_present(None) == (True,) * 6
    assert sm.channels_present('distance,hydrophobicity') == (True, True, False, True, True, False)
    try:
        sm.channels_present(['hydrophobicty'])
    except ValueError as e:
        assert 'hydrophobicty' in str(e)
    else:
        raise AssertionError('typo not caught')
