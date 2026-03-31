import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=6, logspace=True, input_dim=3, include_input=True, include_pi=True):
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = input_dim * (self.num_freqs * 2 + (1 if include_input else 0))

    def forward(self, x):
        if self.num_freqs > 0:
            embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x




class SimpleEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)





class MLP_gelu(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock_gelu(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP_gelu(width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x, data):
        data = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        return x


class ModelNoPosenc(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 256, 32))

        width = 768
        
        self.embedder = SimpleEmbedder()
        
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        decoder_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8, norm=nn.LayerNorm(width))
        self.latent_proj = nn.Linear(32, width)

        self.querier1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.output_proj1 = nn.Linear(width, 1)

    def forward(self, v_data):
        bs = v_data["query_surface_points"].shape[0]
        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent.expand(bs, -1, -1))

        query_surface_points = v_data["query_surface_points"].to(latents.device)
        query_surface_feat = self.query_proj(self.embedder(query_surface_points))
        query_surface_feat = self.querier1(query_surface_feat, latents)
        predicted_surface_results = self.output_proj1(query_surface_feat)
        return predicted_surface_results

    def inference(self, v_res=64, batch_size=20000):
        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent)
        device, dtype = latents.device, latents.dtype

        grid = torch.linspace(-1, 1, v_res, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(grid, grid, grid, indexing="ij")
        query_points = torch.stack([X, Y, Z], dim=-1).reshape(1, -1, 3)
        sdf_chunks = []

        for i in range(0, query_points.shape[1], batch_size):
            q = query_points[:, i:i + batch_size]
            q_feat = self.query_proj(self.embedder(q))
            q_feat = self.querier1(q_feat, latents)
            sdf_chunks.append(self.output_proj1(q_feat)[..., 0].detach())

        sdf = torch.cat(sdf_chunks, dim=1).reshape(v_res, v_res, v_res)
        return sdf





class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 256, 32))

        width = 768
        self.embedder = FourierEmbedder(num_freqs=8, include_input=True, input_dim=3, include_pi=False)
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        decoder_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8, norm=nn.LayerNorm(width))
        self.latent_proj = nn.Linear(32, width)

        self.querier1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.output_proj1 = nn.Linear(width, 1)

    def forward(self, v_data):
        bs = v_data["query_surface_points"].shape[0]
        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent.expand(bs, -1, -1))

        query_surface_points = v_data["query_surface_points"].to(latents.device)
        query_surface_feat = self.query_proj(self.embedder(query_surface_points))
        query_surface_feat = self.querier1(query_surface_feat, latents)
        predicted_surface_results = self.output_proj1(query_surface_feat)
        return predicted_surface_results

    def inference(self, v_res=64, batch_size=20000):
        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent)
        device, dtype = latents.device, latents.dtype

        grid = torch.linspace(-1, 1, v_res, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(grid, grid, grid, indexing="ij")
        query_points = torch.stack([X, Y, Z], dim=-1).reshape(1, -1, 3)
        sdf_chunks = []

        for i in range(0, query_points.shape[1], batch_size):
            q = query_points[:, i:i + batch_size]
            q_feat = self.query_proj(self.embedder(q))
            q_feat = self.querier1(q_feat, latents)
            sdf_chunks.append(self.output_proj1(q_feat)[..., 0].detach())

        sdf = torch.cat(sdf_chunks, dim=1).reshape(v_res, v_res, v_res)
        return sdf
