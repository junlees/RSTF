import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerAnomalyDetector(BaseModel):
    """
    Transformer Autoencoder for anomaly detection in CNC vibration time-series data.

    Architecture
    ─────────────
    1. Input Projection  : n_features → d_model
    2. Positional Encoding
    3. Transformer Encoder (N layers)
    4. Bottleneck        : d_model × window_size → latent_dim
    5. Latent Expansion  : latent_dim → d_model × window_size
    6. Transformer Decoder (N layers, cross-attention with encoder memory)
    7. Output Projection : d_model → n_features  (reconstruction head)
    8. Classifier        : latent_dim → num_classes (classification head)

    Forward output
    ──────────────
    logits       : (B, num_classes)
    reconstruction: (B, window_size, n_features)
    """

    def __init__(
        self,
        n_features: int = 3,
        window_size: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        # ── Input Projection ──────────────────────────────────────────────
        self.input_projection = nn.Linear(n_features, d_model)

        # ── Positional Encoding ───────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # ── Transformer Encoder ───────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Bottleneck ────────────────────────────────────────────────────
        self.encoder_to_latent = nn.Sequential(
            nn.Linear(d_model * window_size, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.latent_to_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, d_model * window_size),
        )

        # ── Transformer Decoder ───────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Output Projection (reconstruction) ───────────────────────────
        self.output_projection = nn.Linear(d_model, n_features)

        # ── Classification Head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, num_classes),
        )

        self._init_weights()

    # ── Weight Initialisation ─────────────────────────────────────────────
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Encode ────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor):
        """
        Args:
            x : (B, window_size, n_features)
        Returns:
            latent : (B, latent_dim)
            memory : (B, window_size, d_model)  — encoder output for cross-attn
        """
        h = self.input_projection(x)          # (B, W, d_model)
        h = self.pos_encoder(h)
        memory = self.transformer_encoder(h)  # (B, W, d_model)

        B = memory.size(0)
        latent = self.encoder_to_latent(memory.reshape(B, -1))  # (B, latent_dim)
        return latent, memory

    # ── Decode ────────────────────────────────────────────────────────────
    def decode(
        self, latent: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent : (B, latent_dim)
            memory : (B, window_size, d_model)
        Returns:
            reconstruction : (B, window_size, n_features)
        """
        B = latent.size(0)
        tgt = self.latent_to_decoder(latent).reshape(
            B, self.window_size, self.d_model
        )                                           # (B, W, d_model)
        tgt = self.pos_encoder(tgt)
        out = self.transformer_decoder(tgt, memory) # (B, W, d_model)
        reconstruction = self.output_projection(out)# (B, W, n_features)
        return reconstruction

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, window_size, n_features)
        Returns:
            logits         : (B, num_classes)
            reconstruction : (B, window_size, n_features)
        """
        latent, memory = self.encode(x)
        reconstruction = self.decode(latent, memory)
        logits = self.classifier(latent)
        return logits, reconstruction
