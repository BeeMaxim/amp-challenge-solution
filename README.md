# AMP Challenge 2027 — Sequence VAE

A reproducible de novo antimicrobial peptide generator for the AMP Challenge 2027.

This repository implements a sequence variational autoencoder (VAE) trained on publicly available antimicrobial peptide sequences. Generated peptides are filtered according to the challenge constraints and ranked using APEX-pathogen predictions.

## Method

The generator uses a sequence VAE with recurrent encoder and decoder networks.

The model consists of:

* bidirectional GRU encoder
* 64-dimensional latent space
* GRU decoder with 256 hidden units
* 2 recurrent layers
* autoregressive sequence generation
* 20 standard amino acids plus BOS, EOS and PAD tokens

During generation, a latent vector is sampled from a standard normal distribution and transformed into the initial decoder hidden state. The peptide sequence is then generated autoregressively using multinomial sampling.

The default sampling temperature is `1.0`.

## Generation pipeline

```text
Training sequences
        │
        ▼
   Sequence VAE
        │
        ▼
 Random latent vectors
        │
        ▼
 Autoregressive decoding
        │
        ▼
 Generated peptide library
        │
        ▼
 Sequence validity filters
        │
        ▼
 50,000 unique sequences
        │
        ▼
 APEX-pathogen prediction
        │
        ▼
 Success Rate ranking
        │
        ▼
 Mean MIC tie-break
        │
        ▼
 Levenshtein reference filter
        │
        ▼
 Top 100 sequences
```

## Sequence generation constraints

Every generated sequence must satisfy the following conditions:

* length between 8 and 50 amino acids
* only the 20 standard amino acids
* no duplicate sequences in the library
* no exact match to a sequence in the antibacterial reference set

The reference set is stored in:

```text
data/antibacterial.fasta
```

## Top-100 selection

The generated library is scored using APEX-pathogen across the configured 11-strain panel.

For each candidate, the computational Success Rate is calculated as:

```text
Success Rate =
number of strains with predicted MIC <= 16 µM
-----------------------------------------------
                  11 strains
```

Candidates are sorted using the following criteria:

```text
1. Success Rate — descending
2. Mean predicted MIC — ascending
```

Therefore, all candidates with a higher Success Rate are ranked ahead of candidates with a lower Success Rate. Mean MIC is used only to break ties.

After ranking, candidates are checked against the antibacterial reference set using Levenshtein similarity.

A candidate is excluded if:

```text
Levenshtein.ratio(candidate, reference) > 0.80
```

for any reference sequence.

The first 100 remaining candidates are written to:

```text
generate/top.fasta
```

The computational ranking is a local pre-selection strategy. It is not intended to reproduce the official experimental scoring procedure of the challenge.

## Outputs

Running the generator produces:

```text
generate/
├── library.fasta
├── top.fasta
└── top_scores.csv
```

### `library.fasta`

The complete generated library containing 50,000 unique peptide sequences.

### `top.fasta`

The selected top-100 candidates after APEX ranking and challenge constraint filtering.

### `top_scores.csv`

Ranking information for the selected candidates:

```text
rank
sequence
success_rate
success_count
mean_mic
```

## Project structure

```text
amp-challenge-solution/
├── apex/
│   └── APEX-pathogen
├── checkpoint/
│   └── vae.pt
├── data/
│   ├── antibacterial.fasta
│   ├── README.md
│   └── training/
│       └── training.fasta
├── generate/
│   └── generated outputs
├── scripts/
│   └── verify_submission.py
├── src/
│   ├── __init__.py
│   ├── generate.py
│   ├── scoring.py
│   ├── train.py
│   ├── utils.py
│   └── vae.py
├── .gitattributes
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

## Training data

The model is trained on publicly available antimicrobial peptide sequences stored in:

```text
data/training/training.fasta
```

The training dataset was compiled from the following public AMP databases:

* DRAMP 3.0
* APD3
* DBAASP

Dataset provenance and preprocessing information are documented in:

```text
data/README.md
```

The antibacterial challenge reference set is kept separate from the training data and is used only for challenge-related filtering.

## Model checkpoint

The trained VAE checkpoint is stored at:

```text
checkpoint/vae.pt
```

The checkpoint is tracked with Git LFS.

## Reproducibility

The generation process uses a fixed random seed.

The default seed is:

```text
42
```

The same seed is applied to Python, NumPy and PyTorch random number generators.

The default generation command is:

```bash
uv run generate
```

The generator is designed to produce reproducible outputs under the same software and hardware environment.

## Installation

The project uses `uv`.

Create the environment and install dependencies with:

```bash
uv sync
```

The project requires:

```text
Python 3.10
```

The main dependencies include:

```text
PyTorch
NumPy
pandas
python-Levenshtein
```

## Running the generator

A small test run:

```bash
uv run generate \
    --n-sequences 1000 \
    --top-k 10
```

Full generation:

```bash
uv run generate
```

The default full run generates:

```text
50,000 library sequences
100 top candidates
```

## Command-line arguments

```text
--n-sequences
    Number of sequences to generate.
    Default: 50000

--top-k
    Number of sequences to retain in top.fasta.
    Default: 100

--batch-size
    Generation batch size.
    Default: 512

--temperature
    Sampling temperature.
    Default: 1.0

--seed
    Random see
```