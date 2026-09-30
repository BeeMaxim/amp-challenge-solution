from __future__ import annotations

import subprocess
from pathlib import Path
import pandas as pd


APEX_PANEL = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]


def apex_predictions(
    seqs: list[str],
    apex_dir: Path,
    workdir: Path,
    panel: list[str] = APEX_PANEL,
) -> pd.DataFrame:
    """Return full APEX MIC matrix: rows=sequences, cols=pathogens."""
    from src.utils import _write_fasta

    workdir.mkdir(parents=True, exist_ok=True)

    fasta = (workdir / "apex_candidates.fasta").resolve()
    out = (workdir / "apex_pred.csv").resolve()

    _write_fasta(seqs, fasta)

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "APEX_predict.py",
            "-i",
            str(fasta),
            "-o",
            str(out),
        ],
        cwd=str(apex_dir),
        check=True,
    )

    df = pd.read_csv(out, index_col=0)

    missing = [c for c in panel if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"APEX output missing panel columns: {missing}"
        )

    return df[panel]


def apex_mean_mic(
    seqs: list[str],
    apex_dir: Path,
    workdir: Path,
    panel: list[str] = APEX_PANEL,
) -> dict[str, float]:
    df = apex_predictions(
        seqs=seqs,
        apex_dir=apex_dir,
        workdir=workdir,
        panel=panel,
    )

    means = df.mean(axis=1)

    return {
        str(seq): float(score)
        for seq, score in means.items()
    }


def apex_success_rate(
    seqs: list[str],
    apex_dir: Path,
    workdir: Path,
    threshold: float = 16.0,
    panel: list[str] = APEX_PANEL,
) -> dict[str, float]:
    df = apex_predictions(
        seqs=seqs,
        apex_dir=apex_dir,
        workdir=workdir,
        panel=panel,
    )

    success_rate = (df <= threshold).mean(axis=1)

    return {
        str(seq): float(score)
        for seq, score in success_rate.items()
    }
