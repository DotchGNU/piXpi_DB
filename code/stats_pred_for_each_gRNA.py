import argparse
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from datetime import datetime
from tqdm import tqdm
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser(description="Streaming analysis by Reference (output to stdout).")
parser.add_argument("-i", "--input", required=True, help="Sorted input file by Reference")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"process_log_{timestamp}.txt"
sig_filename = f"significant_wXw_gRNAs_{timestamp}.txt"

with open(log_filename, "w") as log_file:
    log_file.write(f"{os.path.basename(__file__)}\n")  # First line = script name

# Globals
current_ref = None
chunk = []
error_count = 0
header_written = False
total_refs = 0
significant_refs = 0
sig_refs_list = []

# Format float for output
def format_number(x):
    if pd.isna(x):
        return "NA"
    elif abs(x) < 0.001:
        return f"{x:.2e}"
    else:
        return f"{x:.3f}"

# Avg substrate abundance
def avg_cleavage(sub_df, mm_level, col):
    target = sub_df[sub_df["Mismatch_Count"] == mm_level]
    return target[col].mean() if not target.empty else np.nan

# Count mismatches
def count_mismatches(sub_df, mm_level):
    return sub_df[sub_df["Mismatch_Count"] == mm_level].shape[0]

# Process each Reference group
def process_reference(ref, rows):
    global error_count, header_written, total_refs, significant_refs, sig_refs_list

    df = pd.DataFrame(rows)

    df["WT_log2_Substrate_Abundance"] = np.log2(df["WT_Substrate_Abundance"])
    df["wXw_log2_Substrate_Abundance"] = np.log2(df["wXw_Substrate_Abundance"])
    err = df[["WT_log2_Substrate_Abundance", "wXw_log2_Substrate_Abundance"]].isnull().any(axis=1)
    error_count += err.sum()

    try:
        stat, p = ttest_rel(df["WT_log2_Substrate_Abundance"], df["wXw_log2_Substrate_Abundance"], alternative="less")
    except Exception:
        p = np.nan

    total_refs += 1
    if p < 0.05:
        significant_refs += 1
        sig_refs_list.append(ref)

    result = {
        "Reference": ref,
        "Total_mismatch_count": df.shape[0],
        "1mm_count": count_mismatches(df, 1),
        "2mm_count": count_mismatches(df, 2),
        "3mm_count": count_mismatches(df, 3),
        "WT_total_avg_cleavage_rate": df["WT_Cleavage_Rate"].mean(),
        "wXw_total_avg_cleavage_rate": df["wXw_Cleavage_Rate"].mean(),
        "p-value": p,
        "WT_total_avg_substrate_abundance": df["WT_log2_Substrate_Abundance"].mean(),
        "wXw_total_avg_substrate_abundance": df["wXw_log2_Substrate_Abundance"].mean(),
        "WT_1mm_avg_Substrate_Abundance": avg_cleavage(df, 1, "WT_log2_Substrate_Abundance"),
        "wXw_1mm_avg_Substrate_Abundance": avg_cleavage(df, 1, "wXw_log2_Substrate_Abundance"),
        "WT_2mm_avg_Substrate_Abundance": avg_cleavage(df, 2, "WT_log2_Substrate_Abundance"),
        "wXw_2mm_avg_Substrate_Abundance": avg_cleavage(df, 2, "wXw_log2_Substrate_Abundance"),
        "WT_3mm_avg_Substrate_Abundance": avg_cleavage(df, 3, "WT_log2_Substrate_Abundance"),
        "wXw_3mm_avg_Substrate_Abundance": avg_cleavage(df, 3, "wXw_log2_Substrate_Abundance"),
    }

    if not header_written:
        print("\t".join(result.keys()))
        header_written = True

    print("\t".join(format_number(result[k]) if isinstance(result[k], float) else str(result[k])
                   for k in result))

    with open(log_filename, "a") as log_file:
        log_file.write(f"Processed Reference: {ref}\tRows: {df.shape[0]}\n")

# --- Main processing ---
with open(args.input, "r") as f:
    f.readline()  # Skip metadata line
    header = f.readline().strip().split("\t")

    total_lines = sum(1 for _ in open(args.input)) - 2
    f.seek(0)
    f.readline()
    f.readline()

    for line in tqdm(f, total=total_lines, desc="Processing", unit="line", file=sys.stderr):
        fields = line.strip().split("\t")
        row = dict(zip(header, fields))

        for col in ["Mismatch_Count", "WT_Substrate_Abundance", "wXw_Substrate_Abundance",
                    "WT_Cleavage_Rate", "wXw_Cleavage_Rate"]:
            try:
                row[col] = float(row[col])
            except ValueError:
                row[col] = np.nan

        ref = row["Reference"]

        if current_ref is None:
            current_ref = ref

        if ref != current_ref:
            process_reference(current_ref, chunk)
            current_ref = ref
            chunk = []

        chunk.append(row)

# Last group
if chunk:
    process_reference(current_ref, chunk)

# Save significant references
with open(sig_filename, "w") as f_sig:
    for ref in sig_refs_list:
        f_sig.write(f"{ref}\n")

# Final logs
print(f"Number of rows with NaN due to log2 transform: {error_count}", file=sys.stderr)
print(f"Processing log saved to: {log_filename}", file=sys.stderr)
print(f"Significant Reference list saved to: {sig_filename}", file=sys.stderr)
print(f"Total References processed: {total_refs}", file=sys.stderr)
print(f"References with p < 0.05: {significant_refs}", file=sys.stderr)
print(f"Processed at: {timestamp}", file=sys.stderr)
