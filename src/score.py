from pathlib import Path
import pandas as pd
import argparse

from src.utils import read_fasta
from src.scoring import apex_mean_mic
from src.scoring import APEX_PANEL

ROOT = Path(__file__).resolve().parents[1]

LIBRARY = ROOT / "generate" / "library.fasta"
APEX_DIR = ROOT / "apex"
WORKDIR = ROOT / "generate" / "apex"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of sequences to score. Default: all.",
    )

    args = parser.parse_args()

    seqs = read_fasta(LIBRARY)

    total = len(seqs)

    if args.n is not None:
        if args.n <= 0:
            raise ValueError("--n must be > 0")

        seqs = seqs[:args.n]

    print(
        f"Scoring {len(seqs)} / {total} sequences with APEX...",
        flush=True,
    )

    scores = apex_mean_mic(
        seqs=seqs,
        apex_dir=APEX_DIR,
        workdir=WORKDIR,
    )

    apex_df = pd.read_csv(
        WORKDIR / "apex_pred.csv",
        index_col=0,
    )[APEX_PANEL]

    success_rate = (apex_df <= 16).mean(axis=1)

    df = pd.DataFrame(
        {
            "sequence": list(scores.keys()),
            "mean_mic": list(scores.values()),
        }
    )

    df = df.sort_values("mean_mic", ascending=True)

    suffix = f"{len(seqs)}" if args.n is not None else "all"

    output = (ROOT / "generate" / f"apex_scores_{suffix}.csv")

    df.to_csv(output, index=False)

    print()
    print("========== APEX ==========")
    print(f"n          : {len(df)}")
    print(f"min MIC    : {df.mean_mic.min():.2f}")
    print(f"median MIC : {df.mean_mic.median():.2f}")
    print(f"mean MIC   : {df.mean_mic.mean():.2f}")
    print(f"p10 MIC    : {df.mean_mic.quantile(0.10):.2f}")
    print(f"p25 MIC    : {df.mean_mic.quantile(0.25):.2f}")
    print(f"p75 MIC    : {df.mean_mic.quantile(0.75):.2f}")
    print(f"p90 MIC    : {df.mean_mic.quantile(0.90):.2f}")
    print(f"MIC <= 16  : {(df.mean_mic <= 16).sum()}")
    print(f"MIC <= 32  : {(df.mean_mic <= 32).sum()}")
    print(f"MIC <= 64  : {(df.mean_mic <= 64).sum()}")
    print(f"SR mean       : {success_rate.mean():.3f}")
    print(f"SR median     : {success_rate.median():.3f}")
    print(f"SR max        : {success_rate.max():.3f}")

    print(f"SR >= 1/11    : {(success_rate >= 1/11).sum()}")
    print(f"SR >= 3/11    : {(success_rate >= 3/11).sum()}")
    print(f"SR >= 5/11    : {(success_rate >= 5/11).sum()}")
    print(f"SR >= 8/11    : {(success_rate >= 8/11).sum()}")
    print(f"SR == 11/11   : {(success_rate == 1.0).sum()}")
    print("===========================")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()