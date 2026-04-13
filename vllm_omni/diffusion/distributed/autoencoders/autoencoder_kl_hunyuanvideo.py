# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from diffusers.models.autoencoders import AutoencoderKLHunyuanVideo15
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL_base
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLHunyuanVideo(DistributedAutoencoderKL_base, AutoencoderKLHunyuanVideo15):
    """Distributed VAE for HunyuanVideo 1.5 (T2V and I2V).

    Uses diffusers-style overlapping tile split with linear blending for
    single-GPU and distributed decode.
    """

    def init_distributed(self):
        """Initialize distributed VAE and compute latent tile sizes."""
        super().init_distributed()

        spatial_ratio = getattr(self.config, "spatial_compression_ratio", 8)

        # Derive stride from tile_overlap_factor (set by parent __init__ / enable_tiling).
        # AutoencoderKLHunyuanVideo15 does not have tile_sample_stride_* attributes.
        self.tile_sample_stride_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor))
        self.tile_sample_stride_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor))

        self.tile_latent_min_height = self.tile_sample_min_height // spatial_ratio
        self.tile_latent_min_width = self.tile_sample_min_width // spatial_ratio
        self.tile_latent_stride_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor))
        self.tile_latent_stride_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor))

    # ---- tile-based split (diffusers-style overlapping tiles) ----

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """Split latent tensor into overlapping spatial tiles along H, W."""
        _, _, num_frames, height, width = z.shape

        stride_h = self.tile_latent_stride_height
        stride_w = self.tile_latent_stride_width
        blend_h = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_w = self.tile_sample_min_width - self.tile_sample_stride_width
        row_limit_h = self.tile_sample_stride_height
        row_limit_w = self.tile_sample_stride_width

        tiletask_list = []
        tile_id = 0
        for i in range(0, height, stride_h):
            for j in range(0, width, stride_w):
                tile = z[:, :, :, i : i + self.tile_latent_min_height, j : j + self.tile_latent_min_width]
                tiletask_list.append(
                    TileTask(tile_id, (i // stride_h, j // stride_w), tile, workload=tile.shape[-2] * tile.shape[-1])
                )
                tile_id += 1

        tile_spec = {
            "blend_h": blend_h,
            "blend_w": blend_w,
            "row_limit_h": row_limit_h,
            "row_limit_w": row_limit_w,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.decoder(task.tensor.contiguous())

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        blend_h = grid_spec.tile_spec["blend_h"]
        blend_w = grid_spec.tile_spec["blend_w"]
        row_limit_h = grid_spec.tile_spec["row_limit_h"]
        row_limit_w = grid_spec.tile_spec["row_limit_w"]

        # Build a 2D list mirroring diffusers' rows[][] so that in-place
        # blending on previous tiles is visible to later iterations.
        rows: list[list[torch.Tensor]] = []
        for i in range(grid_h):
            rows.append([coord_tensor_map[(i, j)] for j in range(grid_w)])

        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = rows[i][j]
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = self.blend_h(rows[i][j - 1], tile, blend_w)
                rows[i][j] = tile
                crop_h = min(row_limit_h, tile.shape[-2])
                crop_w = min(row_limit_w, tile.shape[-1])
                result_row.append(tile[:, :, :, :crop_h, :crop_w])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    # ---- decode override ----

    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        if not self.is_distributed_enabled():
            return super().decode(z, return_dict=return_dict, *args, **kwargs)

        logger.debug("HunyuanVideo VAE: distributed tiled decode with overlap blending")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=False,
        )

        if not return_dict:
            return (result,)
        return DecoderOutput(sample=result)
