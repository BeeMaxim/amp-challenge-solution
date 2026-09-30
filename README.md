# AMP Challenge 2027 — Sequence VAE

Sequence VAE generator for the AMP Challenge 2027.

The model is a GRU-based variational autoencoder trained on public antimicrobial peptide sequences. A latent vector is sampled and decoded autoregressively to generate peptide sequences.

Generated candidates are filtered by the challenge constraints and ranked using APEX-pathogen predictions.

## Generation

Install dependencies:

```bash
uv sync
```

Fetch Git LFS files:

```bash
git lfs install
git lfs pull
```

Run the generator:

```bash
uv run generate
```

This produces:

```text
generate/
├── library.fasta
└── top.fasta
```

`library.fasta` contains 50,000 generated sequences.

`top.fasta` contains the selected top-100 candidates.

The generator accepts optional arguments, all with defaults:

```bash
uv run generate \
    --n-sequences 50000 \
    --top-k 100 \
    --batch-size 512 \
    --temperature 1.0 \
    --seed 42
```

## Top-100 selection

Candidates are ranked using the 11-strain APEX panel.

Primary criterion:

```text
Success Rate = fraction of strains with predicted MIC <= 16 µM
```

Tie-breaker:

```text
lower mean predicted MIC
```

Candidates are then filtered against `data/antibacterial.fasta`.

A top-100 candidate is rejected when:

```text
Levenshtein.ratio(candidate, reference) > 0.80
```

## Sequence constraints

Generated sequences satisfy:

* 20 standard amino acids only
* length 8–50 residues
* unique sequences
* no exact match to `data/antibacterial.fasta`
* linear peptides with free termini

## Training data

Training sequences are provided in:

```text
data/training/training.fasta
```

Sources:

* DRAMP 3.0
* APD3
* DBAASP


## Project structure

```text
├── checkpoint/
├── data/
├── generate/
├── scripts/
├── src/
│   ├── generate.py
│   ├── scoring.py
│   ├── train.py
│   ├── utils.py
│   └── vae.py
├── pyproject.toml
└── uv.lock
```

## License

MIT
