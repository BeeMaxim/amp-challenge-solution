import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def read_fasta(path: Path) -> list[str]:
    sequences = []
    current = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.upper())

    if current:
        sequences.append("".join(current))

    return sequences


def write_fasta(
    sequences: list[str],
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as f:
        for i, seq in enumerate(
            sequences,
            start=1,
        ):
            f.write(
                f">seq{i}\n{seq}\n"
            )


def _read_fasta_sequences(path: Path) -> list[str]:
    return read_fasta(path)


def _write_fasta(
    sequences: list[str],
    path: Path,
) -> None:
    write_fasta(sequences, path)
