# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DistributedAutoencoderKLHunyuanVideo tile split/merge/blend (CPU-only)."""

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _DummyHunyuanVae:
    """Minimal mock of DistributedAutoencoderKLHunyuanVideo for unit testing."""

    def __init__(
        self,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_overlap_factor=0.25,
        spatial_ratio=8,
    ):
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_overlap_factor = tile_overlap_factor
        self.tile_sample_stride_height = int(tile_sample_min_height * (1 - tile_overlap_factor))
        self.tile_sample_stride_width = int(tile_sample_min_width * (1 - tile_overlap_factor))
        self.tile_latent_min_height = tile_sample_min_height // spatial_ratio
        self.tile_latent_min_width = tile_sample_min_width // spatial_ratio
        self.tile_latent_stride_height = int(self.tile_latent_min_height * (1 - tile_overlap_factor))
        self.tile_latent_stride_width = int(self.tile_latent_min_width * (1 - tile_overlap_factor))
        self.dtype = torch.float32

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        # Mock: upsample latent by spatial_ratio=8 along H and W
        return z.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)


def _import_tile_split():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuanvideo import (
        DistributedAutoencoderKLHunyuanVideo,
    )

    return DistributedAutoencoderKLHunyuanVideo.tile_split


def _import_tile_exec():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuanvideo import (
        DistributedAutoencoderKLHunyuanVideo,
    )

    return DistributedAutoencoderKLHunyuanVideo.tile_exec


def _import_tile_merge():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuanvideo import (
        DistributedAutoencoderKLHunyuanVideo,
    )

    return DistributedAutoencoderKLHunyuanVideo.tile_merge


class TestTileSplit:
    def test_single_tile(self):
        tile_split = _import_tile_split()
        vae = _DummyHunyuanVae()
        z = torch.zeros(1, 16, 4, 16, 16)
        tasks, grid_spec = tile_split(vae, z)
        assert len(tasks) == 1
        assert grid_spec.grid_shape == (1, 1)

    def test_multiple_tiles_480p(self):
        tile_split = _import_tile_split()
        vae = _DummyHunyuanVae()
        z = torch.zeros(1, 16, 4, 60, 104)
        tasks, grid_spec = tile_split(vae, z)
        assert len(tasks) > 1
        grid_h, grid_w = grid_spec.grid_shape
        assert grid_h * grid_w == len(tasks)

    def test_grid_coords_are_unique(self):
        tile_split = _import_tile_split()
        vae = _DummyHunyuanVae()
        z = torch.zeros(1, 16, 4, 60, 104)
        tasks, _ = tile_split(vae, z)
        coords = [t.grid_coord for t in tasks]
        assert len(coords) == len(set(coords))

    def test_tile_ids_are_sequential(self):
        tile_split = _import_tile_split()
        vae = _DummyHunyuanVae()
        z = torch.zeros(1, 16, 4, 60, 104)
        tasks, _ = tile_split(vae, z)
        assert [t.tile_id for t in tasks] == list(range(len(tasks)))

    def test_tile_shape(self):
        tile_split = _import_tile_split()
        vae = _DummyHunyuanVae()
        z = torch.zeros(1, 16, 4, 60, 104)
        tasks, _ = tile_split(vae, z)
        for t in tasks:
            assert t.tensor.shape[-2] <= vae.tile_latent_min_height
            assert t.tensor.shape[-1] <= vae.tile_latent_min_width


class TestTileMerge:
    def _run_split_exec_merge(self, z):
        tile_split = _import_tile_split()
        tile_exec = _import_tile_exec()
        tile_merge = _import_tile_merge()
        vae = _DummyHunyuanVae()
        tasks, grid_spec = tile_split(vae, z)
        coord_tensor_map = {t.grid_coord: tile_exec(vae, t) for t in tasks}
        return tile_merge(vae, coord_tensor_map, grid_spec)

    def test_output_shape_single_tile(self):
        z = torch.zeros(1, 16, 4, 16, 16)
        result = self._run_split_exec_merge(z)
        assert result.shape[-2] == 16 * 8
        assert result.shape[-1] == 16 * 8

    def test_output_shape_480p(self):
        z = torch.ones(1, 4, 4, 60, 104)
        result = self._run_split_exec_merge(z)
        assert result.shape[0] == 1
        assert result.shape[-2] > 0
        assert result.shape[-1] > 0

    def test_uniform_latent_produces_uniform_output(self):
        """A constant latent should produce a constant output (blend seams vanish)."""
        z = torch.ones(1, 4, 2, 60, 104) * 0.5
        result = self._run_split_exec_merge(z)
        assert torch.allclose(result, result[0, 0, 0, 0, 0].expand_as(result), atol=1e-5)


class TestBlend:
    def test_blend_v_boundary(self):
        vae = _DummyHunyuanVae()
        a = torch.ones(1, 4, 2, 32, 32) * 0.0
        b = torch.ones(1, 4, 2, 32, 32) * 1.0
        blend_extent = 8
        result = vae.blend_v(a, b, blend_extent)
        assert result[:, :, :, 0, :].mean() < result[:, :, :, blend_extent - 1, :].mean()

    def test_blend_h_boundary(self):
        vae = _DummyHunyuanVae()
        a = torch.ones(1, 4, 2, 32, 32) * 0.0
        b = torch.ones(1, 4, 2, 32, 32) * 1.0
        blend_extent = 8
        result = vae.blend_h(a, b, blend_extent)
        assert result[:, :, :, :, 0].mean() < result[:, :, :, :, blend_extent - 1].mean()

    def test_blend_v_no_change_beyond_extent(self):
        vae = _DummyHunyuanVae()
        a = torch.zeros(1, 4, 2, 32, 32)
        b = torch.ones(1, 4, 2, 32, 32) * 2.0
        result = vae.blend_v(a, b, blend_extent=4)
        assert torch.all(result[:, :, :, 4:, :] == 2.0)

    def test_blend_h_no_change_beyond_extent(self):
        vae = _DummyHunyuanVae()
        a = torch.zeros(1, 4, 2, 32, 32)
        b = torch.ones(1, 4, 2, 32, 32) * 2.0
        result = vae.blend_h(a, b, blend_extent=4)
        assert torch.all(result[:, :, :, :, 4:] == 2.0)
