from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

import pandas as pd
from Levenshtein import ratio

from src.scoring import apex_mean_mic, APEX_PANEL

from src.vae import AA, EOS, BOS, ID_TO_AA, SeqVAE, VAEConfig
from src.utils import set_seed, read_fasta, write_fasta


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "checkpoint" / "vae.pt"
REFERENCE_FASTA = ROOT / "data" / "antibacterial.fasta"

OUTPUT_DIR = ROOT / "generate"

MIN_LEN = 8
MAX_LEN = 50


@torch.no_grad()
def generate_batch(
    model: SeqVAE,
    batch_size: int,
    device: torch.device,
    temperature: float,
    min_len: int,
    max_len: int,
):
    z = torch.randn(
        batch_size,
        model.cfg.latent_dim,
        device=device,
    )

    h = model.z_to_hidden(z)
    h = torch.tanh(h)

    h = h.view(
        model.cfg.num_layers,
        batch_size,
        model.cfg.hidden_dim,
    ).contiguous()

    current = torch.full(
        (batch_size, 1),
        BOS,
        dtype=torch.long,
        device=device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    result = [[] for _ in range(batch_size)]

    for position in range(max_len):

        x = model.embedding(current[:, -1:])

        out, h = model.decoder(x, h)

        logits = model.output(out[:, -1, :])
        logits = logits / temperature

        # We don't allow EOS before min_len.
        if position < min_len:
            logits[:, EOS] = -float("inf")

        # At max length force EOS.
        if position == max_len - 1:
            next_token = torch.full(
                (batch_size,),
                EOS,
                dtype=torch.long,
                device=device,
            )
        else:
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            ).squeeze(1)

        current = torch.cat([current, next_token[:, None]], dim=1)

        for i, token in enumerate(next_token.tolist()):
            if finished[i]:
                continue

            if token == EOS:
                finished[i] = True
            elif token >= 3:
                result[i].append(ID_TO_AA[token])

        if finished.all():
            break

    return ["".join(seq) for seq in result]


def load_model(checkpoint: Path, device: torch.device):
    data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    cfg = VAEConfig(**data["config"])

    model = SeqVAE(cfg).to(device)

    model.load_state_dict(data["model"])

    model.eval()

    return model


def generate_library(
    model: SeqVAE,
    n_sequences: int,
    seed: int,
    batch_size: int,
    temperature: float,
    references: set[str],
    device: torch.device,
):
    set_seed(seed)

    collected = {}

    attempts = 0

    while len(collected) < n_sequences:

        attempts += 1

        if attempts > 1000:
            raise RuntimeError(
                "Could not collect enough unique "
                "sequences. Generator diversity is too low."
            )

        batch = generate_batch(
            model=model,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
            min_len=MIN_LEN,
            max_len=MAX_LEN,
        )

        for seq in batch:
            if not (MIN_LEN <= len(seq) <= MAX_LEN):
                continue

            if any(aa not in AA for aa in seq):
                continue

            if seq in references or seq in collected:
                continue

            collected[seq] = None

            if len(collected) >= n_sequences:
                break

        print(f"batch={attempts:04d} unique={len(collected)}/{n_sequences}")

    return list(collected), attempts * batch_size


POTENCY_THRESHOLD = 16.0
MAX_REFERENCE_SIMILARITY = 0.80


def score_top(
    seqs: list[str],
    references: set[str],
    apex_dir: Path,
    workdir: Path,
    top_k: int = 100,
):
    """
    Rank generated peptides by:
      1. APEX success rate across the 11-strain panel
         (fraction of strains with MIC <= 16 uM)
      2. Mean MIC across the same 11 strains (ascending)

    After ranking, candidates violating the challenge's
    Levenshtein similarity rule against the antibacterial
    reference set are excluded.

    Returns:
        top_sequences: list[str]
        ranking: pd.DataFrame
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    # ---------------------------------------------------------
    # 1. Hard sequence validity checks
    # ---------------------------------------------------------

    candidates = []
    seen = set()

    for seq in seqs:
        if seq in seen:
            continue

        seen.add(seq)

        # Standard amino acids only
        if not seq or any(aa not in AA for aa in seq):
            continue

        # Challenge length constraint
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue

        # Exact overlap is not allowed
        if seq in references:
            continue

        candidates.append(seq)

    if not candidates:
        raise RuntimeError("No valid candidates remain before APEX scoring.")

    print(
        f"Valid candidates for APEX: "
        f"{len(candidates)} / {len(seqs)}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 2. Run APEX
    # ---------------------------------------------------------

    apex_mean_mic(
        seqs=candidates,
        apex_dir=apex_dir,
        workdir=workdir,
    )

    apex_df = pd.read_csv(
        workdir / "apex_pred.csv",
        index_col=0,
    )

    missing = [col for col in APEX_PANEL if col not in apex_df.columns]
    if missing:
        raise RuntimeError(
            f"APEX output missing panel columns: {missing}"
        )

    # Keep exactly our candidate set and preserve candidate order
    try:
        apex_df = apex_df.loc[candidates, APEX_PANEL]
    except KeyError as exc:
        missing_candidates = [
            seq for seq in candidates
            if seq not in apex_df.index
        ]
        raise RuntimeError(
            f"APEX output is missing {len(missing_candidates)} "
            f"candidate sequence(s)."
        ) from exc

    # ---------------------------------------------------------
    # 3. Compute Success Rate + mean MIC
    # ---------------------------------------------------------

    ranking = pd.DataFrame(
        {
            "sequence": candidates,
            "success_rate": (
                apex_df.le(POTENCY_THRESHOLD)
                .mean(axis=1)
                .to_numpy()
            ),
            "success_count": (
                apex_df.le(POTENCY_THRESHOLD)
                .sum(axis=1)
                .to_numpy()
            ),
            "mean_mic": (
                apex_df.mean(axis=1)
                .to_numpy()
            ),
        }
    )

    # Highest success rate first.
    # For equal SR: lowest mean MIC first.
    # Stable sorting preserves generator order for complete ties.
    ranking = ranking.sort_values(
        by=["success_rate", "mean_mic"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Challenge Levenshtein filter
    # ---------------------------------------------------------

    selected = []
    rejected = []

    for row in ranking.itertuples(index=False):
        seq = row.sequence

        # Challenge verifier uses Levenshtein.ratio(seq, ref) > 0.8
        # as the rejection condition.
        too_similar = False

        for ref in references:
            if ratio(seq, ref) > MAX_REFERENCE_SIMILARITY:
                too_similar = True
                break

        if too_similar:
            rejected.append(seq)
            continue

        selected.append(seq)

        if len(selected) == top_k:
            break

    if len(selected) < top_k:
        raise RuntimeError(
            f"Could only select {len(selected)} valid top candidates "
            f"out of requested {top_k}."
        )

    # ---------------------------------------------------------
    # 5. Final ranking dataframe
    # ---------------------------------------------------------

    selected_set = set(selected)

    top_df = ranking[
        ranking["sequence"].isin(selected_set)
    ].copy()

    # Restore exact ranking order after filtering.
    order = {seq: i for i, seq in enumerate(selected)}
    top_df["_rank"] = top_df["sequence"].map(order)

    top_df = (
        top_df
        .sort_values("_rank")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))

    print()
    print("========== TOP SELECTION ==========")
    print(f"Candidates        : {len(candidates)}")
    print(f"Levenshtein reject: {len(rejected)}")
    print(f"Selected           : {len(top_df)}")
    print(
        f"Best SR            : "
        f"{top_df.success_rate.iloc[0]:.2%}"
    )
    print(
        f"Best mean MIC      : "
        f"{top_df.mean_mic.iloc[0]:.2f}"
    )
    print("===================================")

    return selected, top_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-sequences", type=int, default=50_000)

    parser.add_argument("--batch-size", type=int, default=512)

    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint", default=str(CHECKPOINT))

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    references = set(
        read_fasta(Path(args.checkpoint).parent.parent / "data" / "antibacterial.fasta")
    )

    model = load_model(Path(args.checkpoint), device)

    library, sampled = generate_library(
        model=model,
        n_sequences=args.n_sequences,
        seed=args.seed,
        batch_size=args.batch_size,
        temperature=args.temperature,
        references=references,
        device=device,
    )

    output = OUTPUT_DIR / "library.fasta"

    write_fasta(library, output)

    # Temporary development-only top 100.
    # This is NOT the final challenge ranking.
    # top_placeholder = OUTPUT_DIR / "top.fasta"


    top_sequences, top_df = score_top(
        seqs=library,
        references=references,
        apex_dir=ROOT / "apex",
        workdir=OUTPUT_DIR / "apex",
        top_k=100,
    )

    top_output = OUTPUT_DIR / "top.fasta"

    write_fasta(
        top_sequences,
        top_output,
    )

    top_df.to_csv(
        OUTPUT_DIR / "top_scores.csv",
        index=False,
    )

    print()
    print(f"Library : {output}")
    print(f"Top-100 : {top_output}")
    print(f"Scores  : {OUTPUT_DIR / 'top_scores.csv'}")

    # write_fasta(library[:100], top_placeholder)

    # -----------------------------
    # Statistics
    # -----------------------------

    n = len(library)

    unique = len(set(library))

    valid_alphabet = sum(all(aa in AA for aa in seq) for seq in library)

    valid_length = sum(MIN_LEN <= len(seq) <= MAX_LEN for seq in library)

    reference_overlap = sum(seq in references for seq in library)

    lengths = [len(seq) for seq in library]

    print()
    print("========== GENERATION STATS ==========")
    print(f"Generated          : {n}")
    print(f"Unique             : {unique}")
    print(
        f"Valid alphabet     : "
        f"{valid_alphabet}/{n} "
        f"({valid_alphabet / n:.2%})"
    )
    print(
        f"Valid length       : "
        f"{valid_length}/{n} "
        f"({valid_length / n:.2%})"
    )
    print(
        f"Reference overlap  : "
        f"{reference_overlap}"
    )
    print(
        f"Mean length        : "
        f"{np.mean(lengths):.2f}"
    )
    print(
        f"Min length         : "
        f"{min(lengths)}"
    )
    print(
        f"Max length         : "
        f"{max(lengths)}"
    )
    print(
        f"Samples drawn       : {sampled}"
    )
    print(
        f"Unique efficiency    : "
        f"{len(library) / sampled:.2%}"
    )
    print("=======================================")

    print()
    print(f"Library: {output}")
    print(f"Temporary top-100: {top_sequences}")


if __name__ == "__main__":
    print(ROOT)
    main()