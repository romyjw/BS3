
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""





from pathlib import Path
import shutil
import tempfile
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric import seed_everything

from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


# ==========================
# CONFIG
# ==========================

batch_size = 16
num_workers = 0
num_per_epoch = 100000
learning_rate = 1e-4


# ==========================
# DATASET
# ==========================

class Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode):
        super().__init__()
        data = np.load("sdf.npz", allow_pickle=True)
        self.q_surface_points = data["q_surface_points"]
        self.surface_sdf = data["surface_sdf"]
        self.mode = v_training_mode

    def __len__(self):
        return num_per_epoch if self.mode == "training" else batch_size

    def __getitem__(self, v_idx):
        idx = np.random.choice(
            np.arange(self.q_surface_points.shape[0]),
            32768,
            replace=False,
        )

        return {
            "query_surface_points": torch.from_numpy(
                self.q_surface_points[idx]
            ).float(),
            "query_surface_sdf": torch.from_numpy(
                self.surface_sdf[idx]
            ).float(),
        }


# ==========================
# DATAMODULE
# ==========================

class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = batch_size
        self.num_worker = num_workers

    def train_dataloader(self):
        return DataLoader(
            Dataset("training"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            Dataset("validation"),
            batch_size=1,
            num_workers=self.num_worker,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            Dataset("testing"),
            batch_size=1,
            num_workers=self.num_worker,
            pin_memory=False,
        )


# ==========================
# MODEL COMPONENTS
# ==========================

class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs=6,
        logspace=True,
        input_dim=3,
        include_input=True,
        include_pi=True,
    ):
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = input_dim * (num_freqs * 2 + (1 if include_input else 0))

    def forward(self, x):
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class MLP_gelu(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ResidualCrossAttentionBlock_gelu(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.ln3 = nn.LayerNorm(width)
        self.mlp = MLP_gelu(width)

    def forward(self, x, data):
        data = self.ln2(data)
        x = x + self.attn(self.ln1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln3(x))
        return x


# ==========================
# MAIN MODEL
# ==========================

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent = nn.Parameter(torch.randn(1, 256, 32))

        width = 768
        self.embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        decoder_layer = nn.TransformerEncoderLayer(
            width, 12, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=8, norm=nn.LayerNorm(width)
        )

        self.latent_proj = nn.Linear(32, width)
        self.querier1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.output_proj1 = nn.Linear(width, 1)

    def forward(self, data):
        bs = data["query_surface_points"].shape[0]

        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent.expand(bs, -1, -1))

        q = self.query_proj(self.embedder(data["query_surface_points"]))
        q = self.querier1(q, latents)

        sdf_pred = self.output_proj1(q)[..., 0]
        return F.l1_loss(sdf_pred, data["query_surface_sdf"])


# ==========================
# LIGHTNING MODULE
# ==========================

class TrainAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("val/loss", loss, prog_bar=True)


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    logger = TensorBoardLogger("log", name="test")
    callbacks = [
        ModelSummary(max_depth=1),
        LearningRateMonitor(logging_interval="step"),
    ]

    model = TrainAutoEncoder()

    trainer = Trainer(
    accelerator="mps",
    devices=1,
    precision="32-true",
    logger=logger,
    callbacks=callbacks,
    )

    data_module = AutoEncoderDataModule()
    trainer.fit(model, data_module)
