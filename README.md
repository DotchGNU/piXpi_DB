# README.md

## Overview

Predict substrate abundances for unmodified and ØXØ-modified gRNAs using a trained MLP regression model.

Two workflows are provided:

1. **Model Development Notebook** (`notebooks/predict_substrate_abundance_using_MLP.ipynb`)
2. **Genome-scale Inference Script** (`code/predict_substrate_abundance_of_human_genome.py`)

## Installation

### 1. Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate <env_name>
```

### 2. Pip Install

```bash
pip install -r requirements.txt
```

## Directory Structure

```
root/
├── notebooks/
│   └── predict_substrate_abundance_using_MLP.ipynb    # Model development notebook
├── code/
│   ├── predict_substrate_abundance_of_human_genome.py  # Genome-scale inference script
│   ├── separate_file_by_reference.py  			# (Helper) Separate result to chunks   
│   ├── stats_pred_for_each_gRNA.py  			# Calculate statistics of each gRNAs
│   └── merge_chunk.py                                  # (Helper) merge chunks 
├── training_dataset/					# Training dataset for developing DLL model 
├── predicting_dataset/                                 # Example for predicting dataset of human genome
├── saved_models/
│   └── MLP_sequential.iter10.pth                     	# MLP weights to predict human genome
├── saved_pkl/						# (optional) items to regress cleavage rate
├── environment.yml                                    	# Conda environment specification
├── requirements.txt                                   	# pip dependencies
├── README.md                                          	# This file
└── LICENSE                                            	# License
```

## Usage

### Model Development 

Launch and run the Jupyter notebook for data exploration, feature encoding, and MLP training:

```bash
jupyter notebook notebooks/predict_substrate_abundance_using_MLP.ipynb
```

### Genome-scale Prediction 

Apply the trained model to potential gRNAs of human genome
( link to access entire off-target datasets of human genome : [off-targets for putative gRNAs of human genome ( n= 747,063,488 ) ](http://clip.korea.ac.kr/piXpi_DB/source_data/crispr.parsed.TrackGreen.details.NGG.MmCode.txt) )
:

```bash
python code/predict_substrate_abundance_of_human_genome.py \
  -i predicting_dataset/crispr.parsed.TrackGreen.details.NGG.MmCode.example.txt \
  -o predicting_dataset/example_result.txt \
  --threads 4 --batch_size 32 --max_mismatch 3
```

---

# LICENSE

MIT License

Copyright (c) 2025 Geun-woo D. Kim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

