"""Tests for mixed .png/.jpg proteogram datasets.

png is create_v2_proteograms.py's default output format. jpg is still accepted
everywhere so datasets, label tables and corpus embeddings built before the
switch keep working, including when the two formats are mixed: a jpg-era label
table has to join against png results, and a jpg-keyed embeddings pickle (what
the current release ships) has to stay usable alongside png proteograms.

These tests pin that compatibility down. They cover the path/label plumbing
only — none of them runs a model, an MD simulation or a real embedding.

Run with: uv run --extra test pytest tests/test_mixed_image_formats.py
"""
import os

import pandas as pd
import pytest

from proteogram.common import find_proteogram_images
from proteogram.common.utils import PROTEOGRAM_IMAGE_EXTENSIONS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def mixed_dir(tmp_path):
    """A proteogram directory holding both formats plus files to ignore."""
    for name in ('d1abca_.png', 'd2bcda_.jpg', 'd3cdea_.png', 'd4defa_.jpg'):
        (tmp_path / name).write_bytes(b'')
    # Non-proteogram files that must not be picked up.
    (tmp_path / 'notes.txt').write_text('not an image')
    (tmp_path / 'labels.tsv').write_text('pdb_id_chain\n')
    return tmp_path


@pytest.fixture
def jpg_era_label_df():
    """A label table written before the png switch: proteogram_file is .jpg."""
    return pd.DataFrame({
        'pdb_id_chain': ['d1abca_', 'd2bcda_', 'd3cdea_'],
        'proteogram_file': ['d1abca_.jpg', 'd2bcda_.jpg', 'd3cdea_.jpg'],
        'class': ['a', 'a', 'b'],
        'fold': ['a.1', 'a.1', 'b.1'],
        'superfamily': ['a.1.1', 'a.1.2', 'b.1.1'],
        'family': ['a.1.1.1', 'a.1.2.1', 'b.1.1.1'],
    })


# --------------------------------------------------------------------------
# find_proteogram_images: the shared reader used by training and similarity
# --------------------------------------------------------------------------

def test_find_proteogram_images_returns_both_formats(mixed_dir):
    found = [os.path.basename(p) for p in find_proteogram_images(mixed_dir)]
    assert found == ['d1abca_.png', 'd2bcda_.jpg', 'd3cdea_.png', 'd4defa_.jpg']


def test_find_proteogram_images_ignores_non_images(mixed_dir):
    found = [os.path.basename(p) for p in find_proteogram_images(mixed_dir)]
    assert 'notes.txt' not in found
    assert 'labels.tsv' not in found


def test_find_proteogram_images_is_sorted_not_grouped_by_extension(mixed_dir):
    """Ordering is by name, so a mixed dataset keeps a stable, format-blind
    order. Grouping by extension would silently reorder a dataset the moment a
    png file was added next to jpgs."""
    found = [os.path.basename(p) for p in find_proteogram_images(mixed_dir)]
    assert found == sorted(found)


def test_find_proteogram_images_is_flat_by_default(mixed_dir):
    """Training and similarity read one flat directory, so recursion is opt-in."""
    nested = mixed_dir / 'class_a'
    nested.mkdir()
    (nested / 'd5efga_.png').write_bytes(b'')
    found = [os.path.basename(p) for p in find_proteogram_images(mixed_dir)]
    assert 'd5efga_.png' not in found


def test_find_proteogram_images_recursive_descends(mixed_dir):
    """What create_corpus_embeddings.py asks for: corpora are often laid out
    in per-class subfolders."""
    nested = mixed_dir / 'class_a'
    nested.mkdir()
    (nested / 'd5efga_.png').write_bytes(b'')
    (nested / 'd6fgha_.jpg').write_bytes(b'')
    found = [os.path.basename(p)
             for p in find_proteogram_images(mixed_dir, recursive=True)]
    assert 'd5efga_.png' in found and 'd6fgha_.jpg' in found
    assert 'd1abca_.png' in found  # top-level files still included


def test_find_proteogram_images_empty_dir(tmp_path):
    assert find_proteogram_images(tmp_path) == []


# --------------------------------------------------------------------------
# Case handling: every caller must agree on which files exist
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['d7ghia_.PNG', 'd8hija_.JPG', 'd9ijka_.Png'])
def test_uppercase_extensions_are_found(tmp_path, name):
    """glob wildcards are case-sensitive even on a case-insensitive filesystem
    (os.path.normcase is a no-op on posix), so matching had to move off the
    glob pattern and onto the lowercased extension. Without that, a .PNG
    dataset is visible to the corpus embedder but invisible to training.
    """
    (tmp_path / name).write_bytes(b'')
    found = [os.path.basename(p) for p in find_proteogram_images(tmp_path)]
    assert found == [name]


def test_jpeg_is_not_accepted_by_the_shared_finder(tmp_path):
    """Documents a known asymmetry, not an endorsement of it.

    PROTEOGRAM_IMAGE_EXTENSIONS is ('.png', '.jpg'), while
    Img2Vec.validate_source also accepts '.jpeg'. create_v2_proteograms.py
    only ever writes .png or .jpg, so this bites only a hand-converted
    dataset. If '.jpeg' is ever added to the tuple, this test should flip.
    """
    (tmp_path / 'e1jkla_.jpeg').write_bytes(b'')
    assert find_proteogram_images(tmp_path) == []


def test_mixed_case_dataset_is_found_exactly_once(tmp_path):
    """Regression guard for the previous per-case-variant globbing: on a
    case-insensitive filesystem, globbing both *.png and *.PNG can return the
    same file twice, double-counting it in the corpus and in progress output."""
    for name in ('a.png', 'b.PNG', 'c.jpg', 'd.JPG'):
        (tmp_path / name).write_bytes(b'')
    found = [os.path.basename(p) for p in find_proteogram_images(tmp_path)]
    assert len(found) == len(set(found)) == 4


def test_directories_named_like_images_are_excluded(tmp_path):
    """A per-class subfolder named e.g. 'a.png' must not enter the file list."""
    (tmp_path / 'decoy.png').mkdir()
    (tmp_path / 'real.png').write_bytes(b'')
    found = [os.path.basename(p)
             for p in find_proteogram_images(tmp_path, recursive=True)]
    assert found == ['real.png']


def test_corpus_embedder_and_training_see_the_same_files(tmp_path):
    """The point of sharing one implementation: the recursive corpus walk and
    the flat training walk must agree on a flat directory, whatever the case
    of the extensions."""
    for name in ('a.png', 'b.JPG', 'c.jpg', 'd.PNG'):
        (tmp_path / name).write_bytes(b'')
    assert (find_proteogram_images(tmp_path)
            == find_proteogram_images(tmp_path, recursive=True))


def test_png_is_listed_before_jpg_in_accepted_extensions():
    """png leads the tuple because it is the current default output format."""
    assert PROTEOGRAM_IMAGE_EXTENSIONS[0] == '.png'
    assert '.jpg' in PROTEOGRAM_IMAGE_EXTENSIONS


# --------------------------------------------------------------------------
# evaluate_methods_v2: label/result joins across extensions
# --------------------------------------------------------------------------

def test_image_stem_strips_either_extension():
    from evaluate_methods_v2 import image_stem

    assert image_stem('d1abca_.png') == 'd1abca_'
    assert image_stem('d1abca_.jpg') == 'd1abca_'
    assert image_stem('/corpus/nested/d1abca_.png') == 'd1abca_'
    assert image_stem('d1abca_') == 'd1abca_'


def test_jpg_label_table_joins_against_png_results(jpg_era_label_df):
    """The core compatibility claim: a label table written when proteograms
    were .jpg still resolves results that name .png files."""
    from evaluate_methods_v2 import image_stem, lookup

    label_df = jpg_era_label_df.copy()
    label_df['proteogram_stem'] = label_df['proteogram_file'].apply(image_stem)

    png_result = image_stem('d1abca_.png')
    assert lookup(label_df, 'proteogram_stem', png_result, 'family') == 'a.1.1.1'
    assert lookup(label_df, 'proteogram_stem', png_result, 'class') == 'a'


def test_join_is_symmetric_across_formats(jpg_era_label_df):
    """A png-era label table must equally resolve a jpg-named result, so a
    corpus half-regenerated in either direction evaluates identically."""
    from evaluate_methods_v2 import image_stem, lookup

    png_label_df = jpg_era_label_df.copy()
    png_label_df['proteogram_file'] = png_label_df['proteogram_file'].str.replace(
        '.jpg', '.png', regex=False)
    png_label_df['proteogram_stem'] = png_label_df['proteogram_file'].apply(image_stem)

    assert lookup(png_label_df, 'proteogram_stem',
                  image_stem('d1abca_.jpg'), 'family') == 'a.1.1.1'


def test_self_hit_detection_survives_a_format_mismatch(jpg_era_label_df):
    """evaluate_methods_v2 drops the self-hit by comparing stems. If a query
    named .png were compared raw against a .jpg corpus key, the self-hit would
    leak into the results and inflate every metric."""
    from evaluate_methods_v2 import image_stem

    query = image_stem('d1abca_.png')
    corpus_self_hit = image_stem('d1abca_.jpg')
    assert query == corpus_self_hit


def test_unknown_stem_returns_none_rather_than_raising(jpg_era_label_df):
    from evaluate_methods_v2 import image_stem, lookup

    label_df = jpg_era_label_df.copy()
    label_df['proteogram_stem'] = label_df['proteogram_file'].apply(image_stem)
    assert lookup(label_df, 'proteogram_stem', 'dZZZZZ_', 'family') is None


# --------------------------------------------------------------------------
# query_similar_proteins: single-query label lookup
# --------------------------------------------------------------------------

def test_lookup_label_row_matches_png_query_against_jpg_table(jpg_era_label_df):
    from query_similar_proteins import lookup_label_row

    row = lookup_label_row(jpg_era_label_df, '/results/d2bcda_.png')
    assert row is not None
    assert row['superfamily'] == 'a.1.2'


def test_lookup_label_row_falls_back_to_pdb_id_chain():
    """A table with no proteogram_file column still matches on the bare ID."""
    from query_similar_proteins import lookup_label_row

    label_df = pd.DataFrame({
        'pdb_id_chain': ['d1abca_'],
        'class': ['a'], 'fold': ['a.1'],
        'superfamily': ['a.1.1'], 'family': ['a.1.1.1'],
    })
    row = lookup_label_row(label_df, 'd1abca_.png')
    assert row is not None and row['class'] == 'a'


def test_lookup_label_row_returns_none_when_absent(jpg_era_label_df):
    from query_similar_proteins import lookup_label_row

    assert lookup_label_row(jpg_era_label_df, 'dZZZZZ_.png') is None


# --------------------------------------------------------------------------
# Img2Vec source resolution
# --------------------------------------------------------------------------

def test_img2vec_validate_source_accepts_a_mixed_directory(mixed_dir):
    """validate_source does not touch self, so it can be exercised without
    loading a model checkpoint."""
    from proteogram.v2 import Img2Vec

    found = [os.path.basename(p)
             for p in Img2Vec.validate_source(None, str(mixed_dir))]
    assert sorted(found) == ['d1abca_.png', 'd2bcda_.jpg',
                             'd3cdea_.png', 'd4defa_.jpg']


def test_img2vec_validate_source_finds_nested_images(mixed_dir):
    """Img2Vec always recurses, matching find_proteogram_images(recursive=True)
    — corpora are often laid out in per-class subdirectories."""
    from proteogram.v2 import Img2Vec

    nested = mixed_dir / 'class_a'
    nested.mkdir()
    (nested / 'd5efga_.jpg').write_bytes(b'')
    found = [os.path.basename(p)
             for p in Img2Vec.validate_source(None, str(mixed_dir))]
    assert 'd5efga_.jpg' in found


# --------------------------------------------------------------------------
# Ablation sets: PNG only, by design
# --------------------------------------------------------------------------

def test_ablation_accepts_png_and_skips_jpg(mixed_dir):
    """The ablation study measures between-channel differences that JPEG
    chroma subsampling destroys, so it reads png only."""
    from create_ablation_proteograms import get_image_paths

    found = [p.name for p in get_image_paths(mixed_dir)]
    assert found == ['d1abca_.png', 'd3cdea_.png']


def test_ablation_rejects_a_jpg_only_dataset(tmp_path):
    """A jpg-only input must fail loudly rather than produce ablations whose
    channel differences have already been smeared together by JPEG."""
    from create_ablation_proteograms import get_image_paths

    (tmp_path / 'd1abca_.jpg').write_bytes(b'')
    with pytest.raises(SystemExit) as excinfo:
        get_image_paths(tmp_path)
    assert 'JPEG' in str(excinfo.value)
