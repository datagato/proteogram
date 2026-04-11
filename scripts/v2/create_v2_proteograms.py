from proteogram.v2 import ProteogramV2
import gc
import glob
import os
from time import time
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend to avoid memory buildup
import matplotlib.pyplot as plt
import torch
import warnings
import argparse
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBConstructionWarning
from proteogram.common import read_yaml
import psutil
import tracemalloc
try:
    import objgraph
except ImportError:
    objgraph = None
# Ignore PDB construction warnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create Proteograms (v2).")
    parser.add_argument("--max_workers", "-w",
                        type=int,
                        default=None,
                        help="Max number of workers for multiprocessing \
                            (default: 0, all available nodes).")
    parser.add_argument('--overwrite',
                        action='store_true',
                        help="Recreate / overwrite Proteograms.")    
    parser.add_argument('--verbose',
                        action='store_true',
                        help="Verbose output.")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Enable debug mode with additional logging and plots.")
    parser.add_argument('--memory-efficient',
                        action='store_true',
                        help="Use memory-efficient solvent subtraction (slower).")
    parser.add_argument('--save_simulated_pdb',
                        action='store_true',
                        help="Save the final simulated PDB structure.")
    args = parser.parse_args()

    start = time()

    # Tell glibc to route large allocations (>= 128 KB) through mmap rather than
    # the sbrk heap.  mmap-allocated pages are returned to the OS via munmap as
    # soon as they are freed, bypassing the per-arena free-list fragmentation that
    # otherwise causes RSS to grow by ~2 GB per protein.  M_TRIM_THRESHOLD=0
    # additionally tells glibc to trim the heap break after every free that leaves
    # space at the top of the main arena.
    # These are no-ops on macOS/Windows; set them once before any OpenMM objects
    # are created so all subsequent large C++ allocations take the mmap path.
    try:
        import ctypes as _ctypes
        _libc = _ctypes.CDLL("libc.so.6")
        _M_TRIM_THRESHOLD = -1   # mallopt param: trim heap after free
        _M_MMAP_THRESHOLD = -3   # mallopt param: mmap allocations >= this size
        _libc.mallopt(_M_TRIM_THRESHOLD, 0)           # trim aggressively
        _libc.mallopt(_M_MMAP_THRESHOLD, 128 * 1024)  # mmap for allocations >= 128 KB
    except Exception:
        pass  # not Linux or libc unavailable — skip silently

    if args.overwrite:
        recreate = True
    else:
        recreate = False
    config = read_yaml('config.yml')
    limit_file = config['limit_file']
    structures_dir = config['scope_structures_dir']
    proteograms_output_dir = config['all_proteograms_dir']

    # Only create proteograms for these structures in the input limit file
    limit_to_these_structs = []
    if limit_file:
        with open(limit_file, 'r') as f:
            for line in f:
                limit_to_these_structs.append(line.strip())

    # If the output dir exists, don't recreate, otherwise make one
    if os.path.exists(proteograms_output_dir):
        print(f'Directory {proteograms_output_dir} exists, will use.')
    else:
        os.makedirs(proteograms_output_dir)

    # Create output directory for production PDB structures if requested
    production_pdb_output_dir = None
    if args.save_simulated_pdb:
        production_pdb_output_dir = os.path.join(
            proteograms_output_dir, 'production_structures')
        if os.path.exists(production_pdb_output_dir):
            print(f'Directory {production_pdb_output_dir} exists, will use.')
        else:
            os.makedirs(production_pdb_output_dir)
            print(f'Created directory {production_pdb_output_dir} for production structures.')

    pdb_files = glob.glob(os.path.join(structures_dir, '**', '*'),
                          recursive=True)
    existing_image_files = glob.glob(
            os.path.join(proteograms_output_dir, '*'))

    # Make a list of (pdb-file-name, image-file-name) tuples as input
    # to process pool
    file_list = []
    exts = ['ent', 'mmcif', 'cif', 'pdb']
    for pdb_file in pdb_files:
        bname =  os.path.basename(pdb_file)
        if len(bname.split('.')) == 1:
            # If there is no extension, add .ent as default
            bname += '.ent'
        
        # name without extensions
        bname_noext = bname
        if bname.split('.')[-1] in exts:
            for x in exts:
                bname_noext = bname_noext.replace(f'.{x}', '')
        
        # There are structure files we wish to limit to
        # Compare using basename without extension since limit file may not include extensions
        if limit_to_these_structs:
            if bname_noext not in limit_to_these_structs and bname not in limit_to_these_structs:
                continue
        
        chain_id = bname[1:5].upper()+':'+ bname[5].upper()
        
        image_file = os.path.join(proteograms_output_dir,
                f'{bname_noext}.jpg')
        if recreate == False:
            if image_file not in existing_image_files:
                file_list.append((pdb_file, image_file))
        else: # recreate
            file_list.append((pdb_file, image_file))

    print(f'Number of structure files = {len(file_list)}')

    # Check if CUDA-enabled GPU is available
    use_gpu = torch.cuda.is_available()

    # Insert regular loop over files using ProteogramV2 here
    problem_files = []
    problem_pdb_cnts = 0
    process = psutil.Process()

    for pdb_file, image_file in tqdm(file_list):
        if args.debug:
            mem_before = process.memory_info().rss / 1024 / 1024
            if torch.cuda.is_available():
                gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"GPU memory before: {gpu_mem_before:.1f} MB")
        try:
            print(f'Processing {pdb_file}...')
            bname =  os.path.basename(pdb_file)
            chain_id = bname[5].upper()
            # Create ProteogramV2 instance
            # Note: the cutoff values are in Angstroms for distance and
            # and are chosen to balance capturing meaningful interactions 
            # while managing computational cost. These can be adjusted along
            # with sequence length cutoffs based on the specific proteins 
            # being analyzed.
            proteogram = ProteogramV2(pdb_file,
                                      output_dir=proteograms_output_dir,
                                      chain_id=chain_id,
                                      calpha_atom_distance_cutoff=10,
                                      sequence_len_lower_cutoff=20,
                                      sequence_len_upper_cutoff=1000,
                                      use_gpu=use_gpu)
            
            # Skip chains that don't meet the sequence length cutoffs
            if not proteogram.is_valid_chain():
                print(f'Skipping {pdb_file}: sequence length {len(proteogram.sequence)} outside [{proteogram.sequence_len_lower_cutoff}, {proteogram.sequence_len_upper_cutoff}]')
                del proteogram
                continue

            # Calculate Proteogram with optional simulated PDB output
            if args.save_simulated_pdb:
                final_data, err, simulated_pdb_stream = proteogram.calculate_proteogram(
                    return_simulated_pdb=True,
                    subtract_solvent_energies=True,
                    debug=args.debug,
                    memory_efficient=args.memory_efficient)
            else:
                final_data, err = proteogram.calculate_proteogram(
                    subtract_solvent_energies=True,
                    debug=args.debug,
                    memory_efficient=args.memory_efficient)
                simulated_pdb_stream = None

            if err is not None and args.verbose:
                print(f'Error calculating Proteogram for {pdb_file}: {err}')
            
            # If Proteogram data is not None, save the image
            if final_data is not None:
                # Save image
                plt.imsave(image_file, final_data.astype('uint8'))
                plt.close('all')  # Clear matplotlib figures from memory
                plt.clf()  # Clear current figure
            
            # Save production simulation PDB structure if requested
            if simulated_pdb_stream is not None and production_pdb_output_dir is not None:
                # Extract base name without extension and add simulation suffix
                bname_base = os.path.splitext(bname)[0]
                production_pdb_file = os.path.join(
                    production_pdb_output_dir,
                    f'{bname_base}_production.pdb'
                )
                with open(production_pdb_file, 'w') as f:
                    f.write(simulated_pdb_stream.read())
                if args.verbose:
                    print(f'Saved production structure to {production_pdb_file}')
            
            # Clean up to free memory between proteins
            del proteogram
            if final_data is not None:
                del final_data
            if simulated_pdb_stream is not None:
                simulated_pdb_stream.close()
                del simulated_pdb_stream
            
            # Force aggressive garbage collection
            gc.collect()
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.debug:
                mem_after = process.memory_info().rss / 1024 / 1024
                print(f"\nMemory after cleanup: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
                if torch.cuda.is_available():
                    gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                    print(f"GPU memory after: {gpu_mem_after:.1f} MB (delta: {gpu_mem_after - gpu_mem_before:.1f} MB)")
                    
        except Exception as e:
            problem_files.append(pdb_file)
            problem_pdb_cnts += 1
            if args.verbose:
                print(f'Problem with file {pdb_file}: {e}') 

    if args.verbose:
        # Save problem pdbs to file
        with open(os.path.join(proteograms_output_dir, 'problem_structures.txt'), 'w') as f:
            for problem_file in problem_files:
                f.write(problem_file + '\n')

    print(f'Problems with {problem_pdb_cnts} structure files.')
    print(f'Computation took {time()-start} seconds')