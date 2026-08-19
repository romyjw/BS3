from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
import shutil
import tempfile
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import mcubes

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning_fabric import seed_everything

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm

# import gc, torch
# gc.collect()
# torch.cuda.empty_cache()
# print(torch.cuda.memory_summary())

batch_size = 4
num_workers = 0
num_per_epoch = 100000
learning_rate = 1e-4
data_name = "40985-arm_sdf.npz"
# data_name = "max_sdf.npz"

# ckpt = "log/test/fruit/fruit/epoch=10-step=68750.ckpt"
# ckpt = "log/max/version_0/checkpoints/epoch=9-step=62500.ckpt"
# ckpt = SCRIPT_DIR / "../spike_best_epoch31.ckpt"
# ckpt = "checkpoints/FE_final_sdf/last.ckpt"
ckpt = "checkpoints/269119-gear_sdf/last.ckpt"
is_training=False
seed=1
init_res = 256
seed_everything(seed)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode):
        super().__init__()
        data = np.load(data_name, allow_pickle=True)
        self.q_surface_points = data["q_surface_points"]
        self.surface_sdf = data["surface_sdf"]
        self.mode = v_training_mode

    def __len__(self):
        return num_per_epoch if self.mode == "training" else batch_size

    def __getitem__(self, v_idx):
        query_surface_points = self.q_surface_points
        idx = np.random.choice(np.arange(query_surface_points.shape[0]), 32768, replace=False)
        query_surface_points = query_surface_points[idx]
        query_surface_sdf = self.surface_sdf[idx]

        out = {
            "query_surface_points": query_surface_points,
            "query_surface_sdf": query_surface_sdf,
        }
        return out


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = batch_size
        self.num_worker = num_workers

    def train_dataloader(self):
        self.train_dataset = Dataset("training",)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            pin_memory=True,
            persistent_workers=(
                True if self.num_worker > 0 else False
            ),
        )

    def val_dataloader(self):
        self.valid_dataset = Dataset("validation",)

        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=self.num_worker,
            pin_memory=True,
            persistent_workers=(
                True if self.num_worker > 0 else False
            ),
            prefetch_factor=4 if self.num_worker > 0 else None,
        )

    def test_dataloader(self):
        self.test_dataset = Dataset("testing",)

        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_worker,
            pin_memory=True,
        )


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
    
class SoftEmbedder(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU() if mode==1 else nn.Softplus(),
            nn.Linear(128, 128),
            nn.GELU() if mode==1 else nn.Softplus(),
            nn.Linear(128, 128),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class MLP_gelu(nn.Module):
    def __init__(self, *,
                 width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock_gelu(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP_gelu(width=width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        data = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        return x


class MLP_softplus(nn.Module):
    def __init__(self, *,
                 width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.c_proj(self.softplus(self.c_fc(x)))

class ResidualCrossAttentionBlock_softplus(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP_softplus(width=width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        data = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        return x


class Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 256, 32))

        width = 768
        # Encoder
        # self.embedder = FourierEmbedder(num_freqs=8, include_input=True, input_dim=3, include_pi=False)
        self.embedder = SimpleEmbedder()
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8, norm=nn.LayerNorm(width))
        self.latent_proj = nn.Linear(32, width)

        # Query
        self.querier1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.output_proj1 = nn.Linear(width, 1)


    def forward(self, v_data, v_test=False):
        bs = v_data["query_surface_points"].shape[0]
        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent.expand(bs, -1, -1))

        query_surface_points = v_data["query_surface_points"]
        query_surface_feat = self.query_proj(self.embedder(query_surface_points))
        query_surface_feat = self.querier1(query_surface_feat, latents)
        predicted_surface_results = self.output_proj1(query_surface_feat)

        separate_surface_sdf_loss = F.l1_loss(predicted_surface_results[..., 0], v_data["query_surface_sdf"], reduction='none').mean(dim=-1)
        surface_sdf_loss = separate_surface_sdf_loss.mean()

        return surface_sdf_loss

    def inference(self, v_res=256):
        latent = self.latent_proj(self.latent)
        latents = self.decoder(latent)

        device = latents.device
        dtype = latents.dtype
        resolution = v_res
        query_points = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=device, dtype=dtype),
            torch.linspace(-1, 1, resolution, device=device, dtype=dtype),
            torch.linspace(-1, 1, resolution, device=device, dtype=dtype),
            indexing="ij"
        )

        query_points = torch.stack(query_points, dim=-1).reshape(1, -1, 3).expand(latents.shape[0], -1, -1)
        batch_size = 60000
        num_batches = int(np.ceil(query_points.shape[1] / batch_size))

        sdf = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, query_points.shape[1])
            query_surface_feat = self.query_proj(self.embedder(query_points[:, start:end]))

            query_feat = self.querier1(query_surface_feat, latents)

            predicted_surface_results = self.output_proj1(query_feat)

            sdf.append(predicted_surface_results[..., 0])
        predicted_surface_results = torch.cat(sdf, dim=1).reshape(-1, resolution, resolution, resolution)
        return predicted_surface_results


class ModelSoft1(Model):
    def __init__(
        self,
    ):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 256, 32))

        width = 768
        # Encoder
        # self.embedder = FourierEmbedder(num_freqs=8, include_input=True, input_dim=3, include_pi=False)
        self.embedder = SoftEmbedder(1)
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True, dropout=0.0, activation=F.gelu)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8, norm=nn.LayerNorm(width))
        self.latent_proj = nn.Linear(32, width)

        # Query
        self.querier1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.output_proj1 = nn.Linear(width, 1)


class ModelSoft2(ModelSoft1):
    def __init__(
        self,
    ):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 256, 32))

        width = 768
        # Encoder
        # self.embedder = FourierEmbedder(num_freqs=8, include_input=True, input_dim=3, include_pi=False)
        self.embedder = SoftEmbedder(2)
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True, dropout=0.0, activation=F.softplus)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8, norm=nn.LayerNorm(width))
        self.latent_proj = nn.Linear(32, width)

        # Query
        self.querier1 = ResidualCrossAttentionBlock_softplus(width, 12)
        self.output_proj1 = nn.Linear(width, 1)


class TrainAutoEncoder(pl.LightningModule):
    def __init__(self):
        super(TrainAutoEncoder, self).__init__()

        self.learning_rate = learning_rate
        # self.model = Model()
        # self.model = ModelSoft1()
        self.model = ModelSoft2()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
        }

    def training_step(self, batch, batch_idx):
        data = batch
        loss = self.model(data)
        self.log(
            "Training/Loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=data["query_surface_points"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss = self.model(data, v_test=True)
        self.log(
            "Validation/Validation_Loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=data["query_surface_points"].shape[0],
        )

        return loss

    def on_validation_epoch_end(self):
        results = self.model.inference(v_res=64)
        sdf = results[0, :, :, :].detach().cpu().numpy().astype(np.float32)
        vertices, triangles = mcubes.marching_cubes(sdf, 0)
        vertices = vertices / 64.0 * 2.0 - 1.0
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export("mesh.ply")
        return

    def test_step(self, batch, batch_idx):
        data = batch

        # Inference
        results = self.model.inference(v_res=256)
        sdf = results[0, :, :, :].detach().cpu().numpy().astype(np.float32)
        vertices, triangles = mcubes.marching_cubes(sdf, 0)
        vertices = vertices / 256.0 * 2.0 - 1.0
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export("mesh.ply")
        return 



if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("log", name="test")
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{SCRIPT_DIR}/../checkpoints/{data_name[:-4]}",
        save_last=True,
        save_top_k=1,
        monitor="Validation/Validation_Loss",
        mode="min",
    )
    callbacks = [ModelSummary(max_depth=1), lr_monitor, ckpt_cb]

    plmodel = TrainAutoEncoder()
    trainer = Trainer(
        devices="auto",
        num_nodes=1,
        precision="16-mixed",
        logger=logger,
        callbacks=callbacks,
        max_epochs=1000000,
    )
    data_module = AutoEncoderDataModule()
    if is_training:
        trainer.fit(plmodel, data_module)
    else:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
        plmodel.load_state_dict(ckpt["state_dict"])

        if True:
            model = plmodel.model
            model.cuda()
            # Find init position
            with torch.no_grad():
                latent = model.latent_proj(model.latent)
                latents = model.decoder(latent)
                device = latents.device
                dtype = latents.dtype

                # init_res = 64
                query_points = torch.meshgrid(
                    torch.linspace(-1, 1, init_res, device=device, dtype=dtype),
                    torch.linspace(-1, 1, init_res, device=device, dtype=dtype),
                    torch.linspace(-1, 1, init_res, device=device, dtype=dtype),
                    indexing="ij"
                )

                query_points = torch.stack(query_points, dim=-1).reshape(1, -1, 3).expand(latents.shape[0], -1, -1)
                batch_size = 30000
                num_batches = int(np.ceil(query_points.shape[1] / batch_size))

                sdf = []
                for i in tqdm(range(num_batches)):
                    start = i * batch_size
                    end = min((i + 1) * batch_size, query_points.shape[1])
                    query_surface_feat = model.query_proj(model.embedder(query_points[:, start:end]))
                    query_feat = model.querier1(query_surface_feat, latents)
                    predicted_surface_results = model.output_proj1(query_feat)
                    sdf.append(predicted_surface_results[..., 0])
                sdf = torch.cat(sdf, dim=1).reshape(-1, init_res, init_res, init_res)
                mask = (sdf < 0.1) # maybe change to abs?
                pos = query_points[0, mask[0].reshape(-1), :]
            
            v, f = mcubes.marching_cubes(sdf[0].cpu().numpy(), 0.0)
            print(f"Extracted {v.shape[0]} vertices and {f.shape[0]} faces")
            v = v / init_res * 2.0 - 1.0
            trimesh.Trimesh(v, f).export(f"{data_name[:-4]}.ply")
            exit()
            chunk_size = 70000  
            new_pos_chunks = []
            for p in torch.split(pos, chunk_size, dim=0):
                # Project points to surface
                p = p.detach().requires_grad_(True)
                query = model.query_proj(model.embedder(p[None, :]))
                feat = model.querier1(query, latents)
                query_sdf = model.output_proj1(feat)
                # Get the normal direction by autograd
                gradient = torch.autograd.grad(
                    outputs=query_sdf,
                    inputs=p,
                    grad_outputs=torch.ones_like(query_sdf),
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                length = torch.linalg.norm(gradient, dim=-1)
                norm = gradient / length[:, None]
                p = p - query_sdf.view(-1, 1) * norm
                new_pos_chunks.append(p.detach()) # Clean up grad here instead

            pos = torch.cat(new_pos_chunks, dim=0)
            
            norm_chunks = []
            hessian_chunks = []
            for p in torch.split(pos, chunk_size, dim=0):
                p = p.detach().requires_grad_(True)
                query = model.query_proj(model.embedder(p[None, :]))
                feat = model.querier1(query, latents)
                query_sdf = model.output_proj1(feat)
                # Get the normal direction by autograd
                gradient = torch.autograd.grad(
                    outputs=query_sdf,
                    inputs=p,
                    grad_outputs=torch.ones_like(query_sdf),
                    create_graph=True,
                    retain_graph=True
                )[0]
                length = torch.linalg.norm(gradient, dim=-1)
                norm = gradient / length[:, None]
                H_cols = []
                for i in range(3):
                    # Take the gradient of the i-th component of the first gradient
                    grad_of_grad_i = torch.autograd.grad(
                        outputs=gradient[:, i],
                        inputs=p,
                        grad_outputs=torch.ones_like(gradient[:, i]),
                        retain_graph=True,
                        create_graph=False
                    )[0]
                    H_cols.append(grad_of_grad_i)
                H = torch.stack(H_cols, dim=2)

                norm_chunks.append(norm.detach()) 
                hessian_chunks.append(H.detach())

            # Concatenate results
            norm = torch.cat(norm_chunks, dim=0)
            hessian = torch.cat(hessian_chunks, dim=0)

            mask = torch.linalg.norm(norm - torch.tensor([[0.,0.,1.]],device=device), dim=-1) < 1e-3
            t1 = torch.tensor([[0.,0.,1.]],device=device).expand_as(norm)
            t1[mask] = torch.tensor([[1.,0.,0.]],device=device)
            t2 = torch.cross(norm, t1)
            t1 = torch.cross(t2, norm)
            t1 = F.normalize(t1, dim=-1)
            t2 = F.normalize(t2, dim=-1)

            II_11 = torch.einsum('bi,bij,bj->b', t1, hessian, t1)
            II_12 = torch.einsum('bi,bij,bj->b', t1, hessian, t2)
            II_21 = torch.einsum('bi,bij,bj->b', t2, hessian, t1)
            II_22 = torch.einsum('bi,bij,bj->b', t2, hessian, t2)
            second_fundamental_form = torch.stack(
                [torch.stack([II_11, II_12], dim=1),
                 torch.stack([II_21, II_22], dim=1)],
                dim=1
            )

            # Eigen decomposition
            # The eigenvalues are the principal curvatures (k1, k2)
            # The eigenvectors are the principal directions in the tangent basis
            with torch.backends.cuda.preferred_linalg_library("magma"):
                principal_curvatures, principal_dirs_2d = torch.linalg.eigh(second_fundamental_form)
            T = torch.stack([t1, t2], dim=2)  # Tangent to world
            principal_directions_3d = torch.bmm(T, second_fundamental_form)

            pd1 = principal_directions_3d[:,:,0]
            pd2 = principal_directions_3d[:,:,1]
            pd1 = F.normalize(pd1, dim=-1)
            pd2 = F.normalize(pd2, dim=-1)


            pos = pos.detach().cpu().numpy()
            normal = norm.detach().cpu().numpy()
            pd1 = pd1.detach().cpu().numpy()
            pd2 = pd2.detach().cpu().numpy()
            principal_curvatures = principal_curvatures.detach().cpu().numpy()

            import open3d as o3d
            def draw(pos, normal, pd1, pd2):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # Move to position
                sphere.translate(pos)
                sphere.paint_uniform_color([0.7, 0.7, 0.7])
                arrow1 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.1, cone_height=0.02)
                R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
                arrow1.rotate(R, center=(0,0,0))
                arrow1.translate(pos)
                arrow1.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(normal * np.pi / 2), center=pos)
                arrow1.paint_uniform_color([1, 0, 0])

                arrow2 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.1, cone_height=0.02)
                R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
                arrow2.rotate(R, center=(0,0,0))
                arrow2.translate(pos)
                arrow2.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(pd1 * np.pi / 2), center=pos)
                arrow2.paint_uniform_color([0, 1, 0])
                arrow3 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.1, cone_height=0.02)
                R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
                arrow3.rotate(R, center=(0,0,0))
                arrow3.translate(pos)
                arrow3.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(pd2 * np.pi / 2), center=pos)
                arrow3.paint_uniform_color([0, 0, 1])
                return sphere+arrow2+arrow3+arrow1

            mesh = o3d.geometry.TriangleMesh()
            mesh += draw(pos[0], normal[0], pd1[0], pd2[0])
            o3d.io.write_triangle_mesh("viz.ply", mesh)
            np.save(
                f"{SCRIPT_DIR}/../models/spike/neural_surface_{seed}_{init_res}",
                    np.concatenate((pos, normal, principal_curvatures[:,0:1]+principal_curvatures[:,1:2], principal_curvatures[:,0:1]*principal_curvatures[:,1:2], pd1, pd2), axis=1).astype(np.float32)
            )
            np.save(
                f"{SCRIPT_DIR}/../models/spike/sdf_{seed}_{init_res}", sdf.cpu().detach().numpy())
            # trimesh.PointCloud(new_pos[0].detach().cpu().numpy()).export("point.ply")

        # trainer.test(plmodel, data_module)
