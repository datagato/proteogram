#!/bin/bash

# Define the directory you want to start from
start_dir="../data/pdbstyle-2.08" # Change this to your directory of PDB files

echo "Starting recursive loop using find in $start_dir"

# Use find to list files and loop through the output
# -type f ensures only regular files are returned
# -print0 and read -d '' safely handle filenames with spaces or special characters
find "$start_dir" -type f -print0 | while IFS= read -r -d '' file; do
  echo "Processing file: $file"
  pdb2fasta "$file" >> "pdbstyle-2.08.fasta"
done

