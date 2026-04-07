# NMDetective-AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Code and data for predicting nonsense-mediated mRNA decay (NMD) efficiency from genomic sequence.
NMDetective-AI is a deep learning model that predicts NMD efficiency for premature termination codons (PTCs) using a finetuned [Orthrus](https://github.com/lucidrains/orthrus) sequence encoder trained on allele-specific expression data from TCGA somatic variants and deep mutational scanning (DMS) experiments.

## Project Organization

```
├── main.py                          <- CLI entry point (data preprocessing, training, manuscript figures)
├── models/
│   └── NMDetectiveAI.pt             <- Trained model weights
│
├── NMD/                             <- Main Python package
│   ├── config.py                    <- Paths, seeds, and global constants
│   ├── utils.py                     <- Shared utilities
│   ├── plots.py                     <- Plotting helpers
│   │
│   ├── data/                        <- Data preprocessing pipeline
│   │   ├── data.py                  <- PTC dataset processing (TCGA, GTEx)
│   │   ├── DMS.py                   <- DMS dataset processing (SP, LE, PE)
│   │   ├── preprocessing.py         <- Shared preprocessing functions
│   │   ├── transcripts.py           <- GenomeKit transcript utilities
│   │   ├── selection.py             <- gnomAD natural selection analysis
│   │   └── DatasetConfig.py         <- Dataset configuration dataclass
│   │
│   ├── modeling/                    <- Model architecture and training
│   │   ├── models/                  <- Model definitions (NMDetectiveAI, A, B variants)
│   │   ├── Trainer.py               <- Training loop
│   │   ├── TrainerConfig.py         <- Hyperparameters and training config
│   │   ├── SequenceDataset.py       <- PyTorch dataset for 6-track sequences
│   │   ├── predict.py               <- Inference / genome-wide prediction
│   │   ├── evaluation.py            <- Evaluation metrics
│   │   └── sweep.py                 <- Hyperparameter sweep
│   │
│   ├── analysis/                    <- Post-hoc analysis scripts
│   │   ├── dms_pca_analysis.py      <- Start-proximal PCA
│   │   ├── dms_sigmoid_fitting.py   <- Sigmoid curve fitting
│   │   ├── start_prox_clusters.py   <- Hierarchical clustering
│   │   ├── long_exon_pca_analysis.py <- Long exon PCA
│   │   └── analyze_long_exon_curves.py
│   │
│   └── manuscript/                  <- Figure-generating scripts (one per panel)
│       ├── manuscript_app.py        <- CLI to generate all figures
│       ├── output.py                <- Output path resolution
│       ├── NMDetectiveAI/           <- Fig 2: Model performance
│       ├── DMS/                     <- Fig 3: DMS overview
│       ├── PE/                      <- Fig 4: Penultimate exon / 50-nt rule
│       ├── LE/                      <- Fig 5: Long exon rule
│       ├── SPvar/                   <- Fig 6: Start-proximal variation
│       ├── SPreinit/                <- Fig 7: Translation reinitiation
│       ├── context/                 <- Fig 7: Sequence context effects
│       ├── selection/               <- Fig 8: Natural selection (gnomAD, TCGA)
│       └── supplementary/           <- Supplementary figures
│
├── data/
│   └── raw/
│       ├── DMS/                     <- DMS experimental data
│       └── PTC/                     <- TCGA/GTEx PTC variant data
│
├── manuscript/
│   ├── figures/                     <- Generated manuscript figures
│   └── supplementary/
│       ├── files/                   <- Supplementary data files
│       └── tables/                  <- Supplementary tables
│
├── environment.yml                  <- Conda environment
├── pyproject.toml                   <- Package metadata
└── LICENSE
```

## Getting Started

### Requirements

- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+
- Conda or mamba

> `environment.yml` in this repository is a historical full environment export.
> It is not recommended for reproducible setup on a fresh machine.

### Installation From Scratch (Tested)

The commands below were verified in a fresh environment on April 7, 2026.

```bash
# 0) Clone the repository and fetch Git LFS files (including model weights)
git clone <repository-url>
cd NMD
git lfs install
git lfs pull

# 1) Create a clean environment
conda create -y -n nmd python=3.10 pip
conda activate nmd

# 2) Install core dependencies used in this project
conda install -y -c conda-forge -c bioconda -c pytorch \
	numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
	tqdm loguru typer biopython openpyxl wandb \
	pytorch genomekit

# 3) Pin Hugging Face stack to versions compatible with Orthrus remote code
pip install -U "transformers==4.50.3" "tokenizers==0.21.1" "huggingface-hub==0.30.1"

# 4) Install pytorch, and packages for mamba. *Note the CUDA version (12.6 here) in the following command, this needs to match the local CUDA version.*
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
conda install conda-forge::causal-conv1d
conda install conda-forge::mamba-ssm

# 5) Install this package in editable mode
pip install -e .
```

### Verify The Installation

Run the following smoke tests:

```bash
python -c "import NMD; import torch, genome_kit, pandas, numpy, typer, transformers, wandb; print('imports_ok')"
python -c "import transformers, tokenizers, huggingface_hub; print(transformers.__version__, tokenizers.__version__, huggingface_hub.__version__)"
python main.py --help
python main.py predict --help
python main.py predict plot-transcript-predictions --help
```

You should see `imports_ok`, a compatible version triplet (`4.50.3 0.21.1 0.30.1`), and the Typer command trees.

The warning below is non-blocking and can be ignored for now:

`FutureWarning: You are using a Python version (3.10.x) which Google will stop supporting ...`

If you encounter:

`AttributeError: 'OrthrusPretrainedModel' object has no attribute 'all_tied_weights_keys'`

or

`ImportError: tokenizers>=0.22.0,<=0.23.0 is required ... but found tokenizers==0.21.1`

you likely have a mixed `transformers`/`tokenizers` install. Re-run a clean reinstall:

```bash
pip uninstall -y transformers tokenizers huggingface-hub
pip install --no-cache-dir "transformers==4.50.3" "tokenizers==0.21.1" "huggingface-hub==0.30.1"
```

### Data Preprocessing

Generate processed datasets from files in `data/raw/`, note however that the raw variant sets are currently not shared with this distribution, and are available upon request, or at a later stage:

```bash
# PTC datasets (somatic_TCGA, germline_TCGA, GTEx)
python main.py dataset all

# DMS datasets (SP, LE, PE) and combined table
python main.py dms all
```

Processed outputs are written to `data/processed/`.

### Training

```bash
# Train on TCGA/PTC data
python main.py train train-ptc

# Fine-tune on DMS SP (starting from pretrained model)
python main.py train train-dms-sp --pretrained-model-path models/NMDetectiveAI.pt

# Gene-level CV on DMS SP
python main.py train train-dms-sp-cv --pretrained-model-path models/NMDetectiveAI.pt
```

### Generating Manuscript Figures

```bash
# All figures
python main.py manuscript generate

# A specific figure
python main.py manuscript generate --fig Fig5

# A single panel
python main.py manuscript panel fig5d
```

## Transcript-Wide Prediction Profiles

### Single Transcript Profile (Plot + Optional CSV)

Generate a transcript profile by gene name:

```bash
python main.py predict plot-transcript-predictions \
	--gene-name TP53 \
	--model-path models/NMDetectiveAI.pt \
	--save-predictions
```

Or specify a transcript directly:

```bash
python main.py predict plot-transcript-predictions \
	--transcript-id ENST00000714408.1 \
	--model-path models/NMDetectiveAI.pt \
	--save-predictions
```

By default this writes:

- Plot PNG/PDF: `reports/figures/transcripts/<transcript_id>.png` and `.pdf`
- Predictions CSV (when `--save-predictions`): `reports/tables/transcripts/<transcript_id>_ptc_predictions.csv`

Single-transcript CSV columns:

- `gene_name`
- `transcript_id`
- `ptc_position` (1-based transcript nucleotide position of inserted stop codon)
- `prediction` (model score)

### Genome-Wide MANE-Style Profiles

Generate one prediction file per selected transcript across protein-coding genes:

```bash
python main.py predict generate-all-mane-predictions
```

Outputs are written to `reports/tables/GW_2/` as many files named:

`<GENE>_<TRANSCRIPT>_ptc_predictions.csv`

Genome-wide per-transcript CSV columns:

- `ptc_position`
- `prediction`
- `gene_name`
- `transcript_id`
- `stop_codon`
- `cds_length`
- `num_exons`
- `strand`

Example rows (`reports/tables/GW_2/TP53_ENST00000714408.1_ptc_predictions.csv`):

```text
ptc_position,prediction,gene_name,transcript_id,stop_codon,cds_length,num_exons,strand
143,-0.18883899,TP53,ENST00000714408.1,TAG,1236,11,-
146,-0.19247606,TP53,ENST00000714408.1,TAG,1236,11,-
149,-0.2078784,TP53,ENST00000714408.1,TAG,1236,11,-
```
