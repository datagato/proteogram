"""
Download all PDB protein structures deposited before a given date from RCSB PDB.

Usage:
    python download_pdb_before_date.py --before 2020-01-01 --output-dir ./pdb_structures
    python download_pdb_before_date.py --before 2020-01-01 --output-dir ./pdb_structures --format cif --workers 16
"""
import argparse
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.{fmt}"
MAX_ROWS_PER_REQUEST = 10000


def fetch_pdb_ids_before_date(cutoff_date: str) -> list[str]:
    """Return PDB entry IDs for experimental protein-only structures deposited before cutoff_date."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.deposit_date",
                        "operator": "less",
                        "value": f"{cutoff_date}T00:00:00Z",
                        "negation": False,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "in",
                        "value": ["Protein (only)", "Protein/NA"],
                        "negation": False,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental",
                        "negation": False,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": MAX_ROWS_PER_REQUEST},
            "results_verbosity": "minimal",
        },
    }

    pdb_ids = []
    start = 0
    total = None

    print(f"Querying RCSB PDB for entries deposited before {cutoff_date} ...")
    with tqdm(unit=" entries", desc="Fetching IDs") as pbar:
        while True:
            query["request_options"]["paginate"]["start"] = start
            resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if total is None:
                total = data.get("total_count", 0)
                pbar.total = total

            batch = [hit["identifier"] for hit in data.get("result_set", [])]
            pdb_ids.extend(batch)
            pbar.update(len(batch))

            start += len(batch)
            if not batch or start >= total:
                break

    return pdb_ids


def download_structure(pdb_id: str, output_dir: Path, fmt: str, session: requests.Session) -> tuple[str, bool, str]:
    """Download a single structure file. Returns (pdb_id, success, error_msg)."""
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.lower(), fmt=fmt)
    dest = output_dir / f"{pdb_id.lower()}.{fmt}"

    if dest.exists():
        return pdb_id, True, "already exists"

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return pdb_id, False, "404 not found"
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return pdb_id, True, ""
    except Exception as e:
        return pdb_id, False, str(e)


def download_all(pdb_ids: list[str], output_dir: Path, fmt: str, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    with requests.Session() as session, \
         ThreadPoolExecutor(max_workers=workers) as executor, \
         tqdm(total=len(pdb_ids), unit=" files", desc="Downloading") as pbar:

        futures = {
            executor.submit(download_structure, pid, output_dir, fmt, session): pid
            for pid in pdb_ids
        }
        for future in as_completed(futures):
            pdb_id, success, msg = future.result()
            if not success and msg != "already exists":
                failed.append((pdb_id, msg))
            pbar.update(1)

    print(f"\nDownloaded {len(pdb_ids) - len(failed)} / {len(pdb_ids)} structures to {output_dir}")
    if failed:
        fail_log = output_dir / "failed_downloads.txt"
        fail_log.write_text("\n".join(f"{pid}\t{msg}" for pid, msg in failed))
        print(f"  {len(failed)} failures logged to {fail_log}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download PDB structures deposited before a given date.")
    parser.add_argument("--before", required=True, metavar="YYYY-MM-DD",
                        help="Download entries deposited strictly before this date.")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="Directory to save downloaded structure files.")
    parser.add_argument("--format", dest="fmt", choices=["pdb", "cif"], default="pdb",
                        help="File format to download (default: pdb).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download threads (default: 8).")
    parser.add_argument("--ids-file", metavar="FILE",
                        help="Optional: path to a file with one PDB ID per line. "
                             "Skips the RCSB query and downloads only these IDs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.ids_file:
        pdb_ids = Path(args.ids_file).read_text().split()
        print(f"Loaded {len(pdb_ids)} IDs from {args.ids_file}")
    else:
        pdb_ids = fetch_pdb_ids_before_date(args.before)
        print(f"Found {len(pdb_ids)} entries deposited before {args.before}")

        ids_cache = output_dir / f"pdb_ids_before_{args.before}.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        ids_cache.write_text("\n".join(pdb_ids))
        print(f"ID list saved to {ids_cache}")

    if not pdb_ids:
        print("No entries to download. Exiting.")
        sys.exit(0)

    download_all(pdb_ids, output_dir, args.fmt, args.workers)
