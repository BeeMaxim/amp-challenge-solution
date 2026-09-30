from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.vae import AA_TO_ID, BOS, EOS, PAD, SeqVAE, VAEConfig
from src.utils import set_seed, read_fasta



def prepare_sequences(
    path: Path,
    min_len: int = 8,
    max_len: int = 50,
) -> list[str]:

    raw = read_fasta(path)

    seen = set()
    result = []

    for seq in raw:
        if not (min_len <= len(seq) <= max_len):
            continue

        if any(aa not in AA_TO_ID for aa in seq):
            continue

        if seq in seen:
            continue

        seen.add(seq)
        result.append(seq)

    return result


def encode_sequence(seq: str, max_len: int) -> tuple[list[int], int]:
    tokens = [BOS]

    tokens.extend(
        AA_TO_ID[aa]
        for aa in seq
    )

    tokens.append(EOS)

    if len(tokens) > max_len + 2:
        raise ValueError(
            f"Sequence too long: {seq}"
        )

    length = len(tokens)

    tokens += [PAD] * (
        max_len + 2 - len(tokens)
    )

    return tokens, length


class PeptideDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        max_len: int,
    ):
        self.items = [
            encode_sequence(seq, max_len)
            for seq in sequences
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tokens, length = self.items[idx]

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(length, dtype=torch.long)


def collate(batch):
    tokens = torch.stack(
        [x[0] for x in batch]
    )

    lengths = torch.stack(
        [x[1] for x in batch]
    )

    return tokens, lengths


def vae_loss(
    logits,
    target,
    mu,
    logvar,
    beta: float,
):
    # logits: [B, T-1, vocab]
    # target: [B, T-1]

    reconstruction = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=PAD,
    )

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    total = reconstruction + beta * kl

    return total, reconstruction, kl


def evaluate(
    model,
    loader,
    device,
    beta: float,
):
    model.eval()

    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0
    n = 0

    with torch.no_grad():
        for tokens, lengths in loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)

            logits, mu, logvar = model(tokens, lengths)

            loss, rec, kl = vae_loss(
                logits,
                tokens[:, 1:],
                mu,
                logvar,
                beta,
            )

            bs = tokens.size(0)

            total_loss += loss.item() * bs
            total_rec += rec.item() * bs
            total_kl += kl.item() * bs
            n += bs

    return total_loss / n, total_rec / n, total_kl / n


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/training/training.fasta")

    parser.add_argument("--output", default="checkpoint/vae.pt")

    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--kl-warmup", type=int, default=30)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    sequences = prepare_sequences(
        Path(args.data),
        min_len=8,
        max_len=50,
    )

    print(f"Usable training sequences: {len(sequences)}")

    random.Random(args.seed).shuffle(sequences)

    split = int(len(sequences) * 0.9)

    train_sequences = sequences[:split]
    val_sequences = sequences[split:]

    print(f"Train: {len(train_sequences)}")
    print(f"Validation: {len(val_sequences)}")

    cfg = VAEConfig(
        max_len=50,
        embedding_dim=128,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=2,
        dropout=0.1,
    )

    train_ds = PeptideDataset(
        train_sequences,
        cfg.max_len,
    )

    val_ds = PeptideDataset(
        val_sequences,
        cfg.max_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = SeqVAE(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        if args.kl_warmup <= 0:
            beta = 1.0
        else:
            beta = min(1.0, epoch / args.kl_warmup)

        train_loss = 0.0
        train_rec = 0.0
        train_kl = 0.0
        n = 0

        for tokens, lengths in train_loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            logits, mu, logvar = model(tokens, lengths)

            loss, rec, kl = vae_loss(
                logits,
                tokens[:, 1:],
                mu,
                logvar,
                beta,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            bs = tokens.size(0)

            train_loss += loss.item() * bs
            train_rec += rec.item() * bs
            train_kl += kl.item() * bs
            n += bs

        train_loss /= n
        train_rec /= n
        train_kl /= n

        val_loss, _, _ = evaluate(
            model,
            val_loader,
            device,
            beta,
        )

        print(
            f"epoch={epoch:03d} "
            f"beta={beta:.3f} "
            f"train={train_loss:.4f} "
            f"rec={train_rec:.4f} "
            f"kl={train_kl:.4f} "
            f"val={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(cfg),
                    "seed": args.seed,
                    "train_size": len(train_sequences),
                    "val_size": len(val_sequences),
                },
                output_path,
            )

            print(f"saved {output_path}")


if __name__ == "__main__":
    main()
