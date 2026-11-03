# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.models.hunyuan_image3.autoencoder import AutoencoderKLConv3D

logger = init_logger(__name__)


class DistributedAutoencoderKLHunyuan(AutoencoderKLConv3D, DistributedVaeMixin):
    @classmethod
    def from_config(cls, config: Any, **kwargs: Any):
        model = super().from_config(config, **kwargs)
        model.init_distributed()
        return model

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    @property
    def use_tiling(self) -> bool:
        return self.use_spatial_tiling

    @use_tiling.setter
    def use_tiling(self, use_tiling: bool) -> None:
        self.use_spatial_tiling = use_tiling

    def advance_posterior_rng(
        self,
        image_shape: tuple[int, ...],
        generator: torch.Generator | None,
        device: torch.device,
    ) -> None:
        """Consume an image posterior's draw without running the encoder.

        Match DiagonalGaussianDistribution.sample, including autocast and the
        distributed tile executor's output cast. HunyuanImage3 conditions on
        single-frame images, expanded to ffactor_temporal by encode().
        """
        if len(image_shape) not in (3, 4, 5) or (len(image_shape) == 5 and image_shape[2] != 1):
            raise ValueError(f"Expected a single-frame image, got shape {image_shape}.")
        batch_size = 1 if len(image_shape) == 3 else image_shape[0]
        height, width = image_shape[-2:]
        shape = (
            batch_size,
            self.config.latent_channels,
            1,
            height // self.ffactor_spatial,
            width // self.ffactor_spatial,
        )
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else self.dtype
        if (
            self.use_spatial_tiling
            and (height > self.tile_sample_min_size or width > self.tile_sample_min_size)
            and self.is_distributed_enabled()
        ):
            # encode_tile_split sets GridSpec.output_dtype to the model dtype.
            dtype = self.dtype
        randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, _, height, width = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = int(self.tile_sample_min_size - blend_extent)

        tiletask_list: list[TileTask] = []
        for i in range(0, height, overlap_size):
            for j in range(0, width, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // overlap_size, j // overlap_size),
                        tile,
                        workload=tile.shape[3] * tile.shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={"blend_extent": blend_extent, "row_limit": row_limit},
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.decoder(task.tensor)

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_extent"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_extent"])
                result_row.append(tile[:, :, :, : grid_spec.tile_spec["row_limit"], : grid_spec.tile_spec["row_limit"]])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, _, height, width = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = int(self.tile_latent_min_size - blend_extent)

        tiletask_list: list[TileTask] = []
        for i in range(0, height, overlap_size):
            for j in range(0, width, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // overlap_size, j // overlap_size),
                        tile,
                        workload=tile.shape[3] * tile.shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={"blend_extent": blend_extent, "row_limit": row_limit},
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.encoder(task.tensor)

    def encode_tile_merge(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        return self.tile_merge(coord_tensor_map, grid_spec)

    def spatial_tiled_encode(self, x: torch.Tensor):
        if not self.is_distributed_enabled():
            return super().spatial_tiled_encode(x)

        logger.debug("Encode running with distributed executor")
        return self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            broadcast_result=True,
        )

    def spatial_tiled_decode(self, z: torch.Tensor):
        if not self.is_distributed_enabled():
            return super().spatial_tiled_decode(z)

        logger.debug("Decode running with distributed executor")
        return self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=True,
        )
