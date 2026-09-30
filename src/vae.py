from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


AA = "ACDEFGHIKLMNPQRSTVWY"

PAD = 0
BOS = 1
EOS = 2
AA_OFFSET = 3

VOCAB = ["<pad>", "<bos>", "<eos>"] + list(AA)
VOCAB_SIZE = len(VOCAB)

AA_TO_ID = {aa: i + AA_OFFSET for i, aa in enumerate(AA)}
ID_TO_AA = {i + AA_OFFSET: aa for i, aa in enumerate(AA)}


@dataclass
class VAEConfig:
    max_len: int = 50
    embedding_dim: int = 128
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1


class SeqVAE(nn.Module):
    """
    Sequence VAE:

        peptide
          -> bidirectional GRU encoder
          -> mu, logvar
          -> z
          -> autoregressive GRU decoder
          -> peptide
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            cfg.embedding_dim,
            padding_idx=PAD,
        )

        self.encoder = nn.GRU(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        encoder_out_dim = cfg.hidden_dim * 2

        self.mu = nn.Linear(
            encoder_out_dim,
            cfg.latent_dim,
        )
        self.logvar = nn.Linear(
            encoder_out_dim,
            cfg.latent_dim,
        )

        self.z_to_hidden = nn.Linear(
            cfg.latent_dim,
            cfg.hidden_dim * cfg.num_layers,
        )

        self.decoder = nn.GRU(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.output = nn.Linear(
            cfg.hidden_dim,
            VOCAB_SIZE,
        )

    def encode(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ):
        x = self.embedding(tokens)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, h = self.encoder(packed)

        # h:
        # [num_layers * 2, batch, hidden]
        #
        # Take the final forward/backward states of the last layer.
        h_forward = h[-2]
        h_backward = h[-1]

        h_enc = torch.cat(
            [h_forward, h_backward],
            dim=-1,
        )

        mu = self.mu(h_enc)
        logvar = self.logvar(h_enc)

        z = self.reparameterize(mu, logvar)

        return mu, logvar, z

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_teacher_forcing(
        self,
        z: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        decoder_input contains:
            <bos> A C D ...

        Output predicts:
            A C D ... <eos>
        """

        batch_size = z.size(0)

        h0 = self.z_to_hidden(z)
        h0 = torch.tanh(h0)

        h0 = h0.view(
            self.cfg.num_layers,
            batch_size,
            self.cfg.hidden_dim,
        )

        x = self.embedding(decoder_input)

        out, _ = self.decoder(
            x,
            h0.contiguous(),
        )

        logits = self.output(out)

        return logits

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ):
        mu, logvar, z = self.encode(
            tokens,
            lengths,
        )

        decoder_input = tokens[:, :-1]

        logits = self.decode_teacher_forcing(
            z,
            decoder_input,
        )

        return logits, mu, logvar
