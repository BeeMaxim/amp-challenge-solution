#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import Levenshtein


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

MIN_LENGTH = 8
MAX_LENGTH = 50

LIBRARY_SIZE = 50_000
TOP_SIZE = 100

SIMILARITY_THRESHOLD = 0.8


def read_fasta(path: Path):
    """FASTA parser compatible with the official validator."""
    headers = []
    sequences = []

    header = None
    seq_parts = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(seq_parts))

            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line.upper())

    if header is not None:
        headers.append(header)
        sequences.append("".join(seq_parts))

    return headers, sequences


def check_library(path: Path):
    print("=" * 70)
    print("1. FULL LIBRARY")
    print("=" * 70)

    headers, sequences = read_fasta(path)
    errors = []

    print(f"Sequences: {len(sequences)}")

    # Number of sequences
    if len(sequences) != LIBRARY_SIZE:
        errors.append(
            f"Expected {LIBRARY_SIZE} sequences, got {len(sequences)}"
        )

    # Individual sequence checks
    seen = set()

    for i, (header, seq) in enumerate(
        zip(headers, sequences), start=1
    ):
        if not header.strip():
            errors.append(f"Record {i}: missing header")

        if not seq:
            errors.append(
                f"Record {i} ('{header}'): empty sequence"
            )
            continue

        invalid = set(seq) - STANDARD_AMINO_ACIDS
        if invalid:
            errors.append(
                f"Record {i} ('{header}'): "
                f"invalid characters {sorted(invalid)}"
            )

        if len(seq) < MIN_LENGTH:
            errors.append(
                f"Record {i} ('{header}'): "
                f"too short ({len(seq)} < {MIN_LENGTH})"
            )

        if len(seq) > MAX_LENGTH:
            errors.append(
                f"Record {i} ('{header}'): "
                f"too long ({len(seq)} > {MAX_LENGTH})"
            )

        if seq in seen:
            errors.append(
                f"Record {i} ('{header}'): duplicate sequence"
            )

        seen.add(seq)

    print(f"Unique sequences: {len(seen)}")
    print(f"Duplicates: {len(sequences) - len(seen)}")

    if errors:
        print(f"\nFAIL: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
    else:
        print("PASS")

    return set(sequences), errors


def check_overlap(library, reference):
    print()
    print("=" * 70)
    print("2. EXACT OVERLAP WITH antibacterial.fasta")
    print("=" * 70)

    overlap = library & reference

    print(f"Reference sequences: {len(reference)}")
    print(f"Exact overlap: {len(overlap)}")

    if overlap:
        print("FAIL")
        print("\nOverlapping sequences:")

        for seq in sorted(overlap):
            print(f"  {seq}")

        return False

    print("PASS")
    return True


def check_top(top_path: Path, library):
    print()
    print("=" * 70)
    print("3. TOP-100")
    print("=" * 70)

    headers, top_sequences = read_fasta(top_path)
    errors = []

    print(f"Top sequences: {len(top_sequences)}")
    print(f"Unique top sequences: {len(set(top_sequences))}")

    # Number of top sequences
    if len(top_sequences) != TOP_SIZE:
        errors.append(
            f"Expected {TOP_SIZE} sequences, "
            f"got {len(top_sequences)}"
        )

    seen = set()

    for i, seq in enumerate(top_sequences, start=1):

        # Must be part of full library
        if seq not in library:
            errors.append(
                f"Top record {i}: sequence not found "
                f"in full library"
            )

        # No duplicates
        if seq in seen:
            errors.append(
                f"Top record {i}: duplicate sequence"
            )

        seen.add(seq)

    if errors:
        print(f"FAIL: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return top_sequences, False

    print("PASS")

    return top_sequences, True


def check_top_similarity(top_sequences, reference):
    print()
    print("=" * 70)
    print("4. TOP-100 LEVENSHTEIN SIMILARITY")
    print("=" * 70)

    threshold = 0.8

    print(f"Valid condition: max Levenshtein.ratio <= {threshold}")

    results = []

    for seq in sorted(set(top_sequences)):
        max_score = -1.0
        best_ref = None

        for ref in reference:
            score = Levenshtein.ratio(seq, ref)

            if score > max_score:
                max_score = score
                best_ref = ref

        results.append((seq, max_score, best_ref))

    # Самые похожие последовательности сверху
    results.sort(key=lambda x: x[1], reverse=True)

    valid = [
        item for item in results
        if item[1] <= threshold
    ]

    invalid = [
        item for item in results
        if item[1] > threshold
    ]

    print(f"Top sequences checked : {len(results)}")
    print(f"Valid (<= 0.8)        : {len(valid)}")
    print(f"Invalid (> 0.8)       : {len(invalid)}")

    if results:
        print(f"Maximum similarity     : {results[0][1]:.4f}")

    print("\nClosest reference match for each top candidate:")

    for seq, score, ref in results[:10]:
        status = "FAIL" if score > threshold else "OK"

        print(f"  {score:.4f} [{status}]")
        print(f"    TOP: {seq}")
        print(f"    REF: {ref}")

    if invalid:
        print("\nInvalid top candidates:")

        for seq, score, ref in invalid:
            print(f"  {score:.4f}  {seq}")
            print(f"    closest: {ref}")

    if invalid:
        print("\nSTATUS: FAIL")
        return False

    print("\nSTATUS: PASS")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Local AMP Challenge 2027 validator"
    )

    parser.add_argument(
        "--library",
        type=Path,
        default=Path("generate/library.fasta"),
        help="Full 50k FASTA"
    )

    parser.add_argument(
        "--top",
        type=Path,
        default=Path("generate/top.fasta"),
        help="Top-100 FASTA"
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/antibacterial.fasta"),
        help="Challenge reference FASTA"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Check input files
    # ---------------------------------------------------------

    for path in (
        args.library,
        args.top,
        args.reference,
    ):
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            return 2

    # ---------------------------------------------------------
    # 1. Library
    # ---------------------------------------------------------

    library, library_errors = check_library(args.library)

    # ---------------------------------------------------------
    # Reference
    # ---------------------------------------------------------

    _, reference_sequences = read_fasta(args.reference)
    reference = set(reference_sequences)

    # ---------------------------------------------------------
    # 2. Exact overlap
    # ---------------------------------------------------------

    overlap_ok = check_overlap(
        library,
        reference
    )

    # ---------------------------------------------------------
    # 3. Top 100
    # ---------------------------------------------------------

    top_sequences, top_ok = check_top(
        args.top,
        library
    )

    # ---------------------------------------------------------
    # 4. Similarity
    # ---------------------------------------------------------

    similarity_ok = check_top_similarity(
        top_sequences,
        reference
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    all_ok = (
        not library_errors
        and overlap_ok
        and top_ok
        and similarity_ok
    )

    if all_ok:
        print("PASS")
        print("All local sequence checks passed.")
        return 0

    print("FAIL")
    print("One or more checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())