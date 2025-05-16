import os
import torch
import joblib
import numpy as np
import argparse
from tqdm import tqdm
import sys

class MLPRegressor(torch.nn.Module):
    def __init__(self, seq_length, input_size, output_size):
        super(MLPRegressor, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(seq_length * input_size, 320), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(320, 80), torch.nn.ReLU())
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(80, 40), torch.nn.ReLU())
        self.out = torch.nn.Linear(40, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

# set parameter to convert substrate abundance to cleavage rate using polynomeal regression
POLY_FEATURES_PATH = 'saved_pkl/poly_features.pkl'
POLY_REG_MODEL_PATH = 'saved_pkl/poly_reg_model.pkl'
poly = joblib.load(POLY_FEATURES_PATH)
poly_reg = joblib.load(POLY_REG_MODEL_PATH)

base_vocab = ['A', 'T', 'G', 'C']
base_to_idx = {c: i for i, c in enumerate(base_vocab)}
transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

def substitution_encoding(ref, read, base_to_idx):
    encoding = []
    for r, s in zip(ref, read):
        flag = [0, 0]
        on_target_vec = [0, 0, 0, 0]
        off_target_vec = [0, 0, 0, 0]
        if r != s:
            if (r, s) in transitions:
                flag[0] = 1
            else:
                flag[1] = 1
        if r in base_to_idx:
            on_target_vec[base_to_idx[r]] = 1
        if s in base_to_idx:
            off_target_vec[base_to_idx[s]] = 1
        encoding.append(flag + on_target_vec + off_target_vec)
    return np.array(encoding)

def count_mismatch(read_pattern):
    return sum(1 for char in read_pattern if char != '.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-p", "--threads", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for GPU inference")
    parser.add_argument("-m", "--max_mismatch", type=int, default=3, help="Maximum mismatch allowed for processing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_file = 'saved_model/MLP_sequential.iter10.pth'
    mlp_model = MLPRegressor(seq_length=20, input_size=10, output_size=2)
    mlp_model.load_state_dict(torch.load(model_file, map_location=device))
    mlp_model.to(device)
    mlp_model.eval()

    error_log_filename = args.output_file[:-4] + "_error_log.txt" if args.output_file.endswith(".txt") else args.output_file + "_error_log.txt"

    with open(args.input_file, 'r') as fin:
        lines = fin.readlines()

    pbar = tqdm(total=len(lines), desc="Processing lines")

    with open(args.output_file, 'w') as fout, open(error_log_filename, 'w') as ferr:
        fout.write(f"Model file: {model_file}\n")
        fout.write("Reference\tRead_Pattern\tMismatch_Count\tWT_Substrate_Abundance\twXw_Substrate_Abundance\tWT_Cleavage_Rate\twXw_Cleavage_Rate\n")

        batch_data = []

        for line in lines:
            cols = line.strip().split("\t")
            if len(cols) < 4:
                ferr.write(line)
                continue
            reference_seq = cols[0]
            reads = cols[3].split('|')

            for read in reads:
                mismatch_count = count_mismatch(read)
                if mismatch_count <= args.max_mismatch:
                    batch_data.append((reference_seq, read, mismatch_count))

                if len(batch_data) >= args.batch_size:
                    inputs = [substitution_encoding(ref, [r if r != '.' else ref[i] for i, r in enumerate(read)], base_to_idx) for ref, read, _ in batch_data]
                    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        abundances = mlp_model(inputs_tensor).cpu().numpy()

                    for (ref, rp, mm_count), abun in zip(batch_data, abundances):
                        abundance_poly = poly.transform(abun.reshape(-1, 1)) # polynomeal regression
                        cleavage_rate = np.clip(poly_reg.predict(abundance_poly).flatten(), 0, 1)
                        fout.write(f"{ref}\t{rp}\t{mm_count}\t{abun[0]:.4f}\t{abun[1]:.4f}\t{cleavage_rate[0]:.4f}\t{cleavage_rate[1]:.4f}\n")
                    batch_data = []

            pbar.update(1)

        if batch_data:
            inputs = [substitution_encoding(ref, [r if r != '.' else ref[i] for i, r in enumerate(read)], base_to_idx) for ref, read, _ in batch_data]
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            with torch.no_grad():
                abundances = mlp_model(inputs_tensor).cpu().numpy()

            for (ref, rp, mm_count), abun in zip(batch_data, abundances):
                abundance_poly = poly.transform(abun.reshape(-1, 1))
                cleavage_rate = np.clip(poly_reg.predict(abundance_poly).flatten(), 0, 1)
                fout.write(f"{ref}\t{rp}\t{mm_count}\t{abun[0]:.4f}\t{abun[1]:.4f}\t{cleavage_rate[0]:.4f}\t{cleavage_rate[1]:.4f}\n")

    pbar.close()

if __name__ == "__main__":
    main()

