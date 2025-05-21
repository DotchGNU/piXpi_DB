# piXpi DB 

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


```bash
python code/predict_substrate_abundance_of_human_genome.py \
  -i predicting_dataset/crispr.parsed.TrackGreen.details.NGG.MmCode.example.txt \
  -o predicting_dataset/example_result.txt \
  --threads 4 --batch_size 32 --max_mismatch 3
```

---
# LICENSE & Contact

License:
This software is licensed under the MIT License. See the LICENSE file for details.

Contact:
For questions or bug reports, please refer to [gwpia0409@korea.ac.kr].
