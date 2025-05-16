#!/usr/bin/env python3
import argparse
import sys
from tqdm import tqdm
import subprocess

# Argument parser
parser = argparse.ArgumentParser(description="Split input file into chunks by Reference, preserving Reference integrity.")
parser.add_argument("-i", "--input", required=True, help="Input file path")
parser.add_argument("-n", "--num_chunks", type=int, required=True, help="Number of output chunks")
args = parser.parse_args()

# Count total lines
with open(args.input, 'r') as f:
    total_lines = sum(1 for _ in f)

# Open input file
with open(args.input, 'r') as infile:
    header1 = infile.readline()
    header2 = infile.readline()

    lines_per_chunk = (total_lines - 2) // args.num_chunks

    # Prepare output files
    chunk_files = [open(f"{args.input}.chunk{i+1}.txt", "w") for i in range(args.num_chunks)]

    # Write headers
    for chunk_file in chunk_files:
        chunk_file.write(header1)
        chunk_file.write(header2)

    current_chunk_idx = 0
    current_lines_in_chunk = 0
    current_ref = None

    for line in tqdm(infile, total=total_lines-2, desc="Splitting file"):
        ref = line.split("\t")[0]

        if current_lines_in_chunk >= lines_per_chunk and ref != current_ref and current_chunk_idx < args.num_chunks - 1:
            current_chunk_idx += 1
            current_lines_in_chunk = 0

        chunk_files[current_chunk_idx].write(line)
        current_ref = ref
        current_lines_in_chunk += 1

# Close all output files
for chunk_file in chunk_files:
    chunk_file.close()

# Verification
print("\n[Verification]")
original_line_count = total_lines - 2
print(f"Original file line count: {original_line_count}")
chunk_total_lines = 0
chunk_refs = set()

original_refs = subprocess.check_output(f'tail -n +3 "{args.input}" | cut -f1 | sort | uniq', shell=True).decode().splitlines()
print(f"Original unique references: {len(original_refs)}")

for i in range(args.num_chunks):
    chunk_file_name = f"{args.input}.chunk{i+1}.txt"
    lines = int(subprocess.check_output(f'wc -l < "{chunk_file_name}"', shell=True).decode().strip()) - 2
    chunk_total_lines += lines
    print(f"Chunk {i+1} line count: {lines}")

    refs = subprocess.check_output(f'tail -n +3 "{chunk_file_name}" | cut -f1 | sort | uniq', shell=True).decode().splitlines()
    chunk_refs.update(refs)
    print(f"Chunk {i+1} unique references: {len(refs)}")

print(f"Chunk total unique references: {len(chunk_refs)}")

if chunk_total_lines == original_line_count and len(original_refs) == len(chunk_refs):
    print("Verification passed: Files split correctly.")
else:
    print("Verification failed: Discrepancy detected.")

