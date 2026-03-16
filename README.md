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

### Installation

```bash
conda env create -f environment.yml
conda activate NMD
```

### Data Preprocessing

Generate processed data files from the raw datasets:

```bash
python main.py data main
```

This runs the full preprocessing pipeline (see below) and writes output to `data/processed/`.

### Training

```bash
# Train on TCGA PTC data
python main.py train-ptc

# Fine-tune on DMS data
python main.py train-dms --pretrained-model-path models/NMDetectiveAI.pt

# Gene-level cross-validation on DMS
python main.py train-dms-gene-cv --pretrained-model-path models/NMDetectiveAI.pt
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
