import numpy as np
import torch
from typing import Any, Dict, Optional, Union
from optimum.onnxruntime.modeling_diffusion import ORTModelVaeDecoder, ORTModelVaeEncoder
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput


ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}

import logging
logger = logging.getLogger(__name__)

class ORTModelTiledVaeWrapper(object):
    def __init__(
        self,
        wrapped: Union[ORTModelVaeDecoder, ORTModelVaeEncoder],
        decoder: bool,
        window: int,
        overlap: float,
    ):
        self.wrapped = wrapped
        self.decoder = decoder
        self.tiled = False
        self.set_window_size(window, overlap)

    def set_tiled(self, tiled: bool = True):
        self.tiled = tiled

    def set_window_size(self, window: int, overlap: float):
        self.tile_latent_min_size = window
        self.tile_sample_min_size = window * 8
        self.tile_overlap_factor = overlap

    def __call__(self, latent_sample=None, sample=None, **kwargs):
        # convert latent/sample type to match model for mixed fp16/fp32 support
        # sample_dtype = next(
        #     (
        #         input.type
        #         for input in self.wrapped.session.get_inputs()
        #         if input.name == "sample" or input.name == "latent_sample"
        #     ),
        #     "tensor(float)",
        # )
        # sample_dtype = ORT_TO_NP_TYPE[sample_dtype]

        # if latent_sample is not None and latent_sample.dtype != sample_dtype:
        #     logger.debug("converting VAE latent sample dtype to %s", sample_dtype)
        #     latent_sample = latent_sample.astype(sample_dtype)

        # if sample is not None and sample.dtype != sample_dtype:
        #     logger.debug("converting VAE sample dtype to %s", sample_dtype)
        #     sample = sample.astype(sample_dtype)

        if self.tiled:
            if self.decoder:
                return self.tiled_decode(latent_sample, **kwargs)
            else:
                return self.tiled_encode(sample, **kwargs)
        else:
            if self.decoder:
                return self.wrapped(latent_sample=latent_sample)
            else:
                return self.wrapped(sample=sample)

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def blend_v(self, a, b, blend_extent):
        for y in range(min(a.shape[2], b.shape[2], blend_extent)):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, y, :
            ] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        for x in range(min(a.shape[3], b.shape[3], blend_extent)):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    @torch.no_grad()
    def tiled_encode(
        self, x, return_dict: bool = True
    ) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.
        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is:
        different from non-tiled encoding due to each tile using a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            x (`torch.FloatTensor`): Input batch of images. return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`AutoencoderKLOutput`] instead of a plain tuple.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = torch.from_numpy(self.wrapped(sample=tile.numpy())[0])
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2).numpy()
        if not return_dict:
            return (moments,)

        return AutoencoderKLOutput(latent_dist=moments)

    @torch.no_grad()
    def tiled_decode(
        self, z, return_dict: bool = True
    ):
        r"""Decode a batch of images using a tiled decoder.
        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled decoding is:
        different from non-tiled decoding due to each tile using a different decoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
            `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)

        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                decoded = torch.from_numpy(self.wrapped(latent_sample=tile.numpy())[0])
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        dec = dec.numpy()

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


