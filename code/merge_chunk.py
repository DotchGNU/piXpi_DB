#!/usr/bin/env python3
import argparse
import sys
import subprocess

# Argument parser
parser = argparse.ArgumentParser(description="Merge chunk files into a single file, ensuring only one header.")
parser.add_argument("-i", "--inputs", nargs='+', required=True, help="Input chunk file paths")
parser.add_argument("-o", "--output", required=True, help="Output merged file path")
args = parser.parse_args()

unique_refs_input = set()

with open(args.output, 'w') as outfile:
    header_written = False

    for idx, file in enumerate(args.inputs):
        with open(file, 'r') as infile:
            header = infile.readline()

            refs = subprocess.check_output(f'tail -n +2 "{file}" | cut -f1 | sort | uniq', shell=True).decode().splitlines()
            unique_refs_input.update(refs)

            if not header_written:
                outfile.write(header)
                header_written = True

            for line in infile:
                outfile.write(line)

# Verify merged output
unique_refs_output = subprocess.check_output(f'tail -n +2 "{args.output}" | cut -f1 | sort | uniq', shell=True).decode().splitlines()

print(f"Merged files into {args.output}")
print("\n[Verification]")
print(f"Unique references in input chunks: {len(unique_refs_input)}")
print(f"Unique references in merged output: {len(unique_refs_output)}")

if len(unique_refs_input) == len(unique_refs_output):
    print("Verification passed: References merged correctly.")
else:
    print("Verification failed: Discrepancy in references detected.")

