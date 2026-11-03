# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV reuse must preserve conditional-image draws before initial diffusion noise."""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.hunyuan_image3.autoencoder as autoencoder_module
import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as pipeline_module
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import DistributedAutoencoderKLHunyuan
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerEncodeOutput
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Text2ImagePipeline,
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.native import ResnetBlock
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline
from vllm_omni.diffusion.models.hunyuan_image3.request_layout import HunyuanPreparedLayout

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(
    params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}))]
)
def pipeline(request, monkeypatch):
    device = torch.device(request.param)
    if device.type == "cpu":
        # Dispatch depends on the host platform, even when this test targets CPU.
        monkeypatch.setattr(autoencoder_module, "ResnetBlock", ResnetBlock)
    model = object.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(model)
    # Real encoder, reduced widths and geometry; no pretrained weights or collectives.
    model.vae = (
        DistributedAutoencoderKLHunyuan(
            in_channels=3,
            out_channels=3,
            latent_channels=2,
            block_out_channels=(32, 32, 64),
            layers_per_block=1,
            ffactor_spatial=4,
            ffactor_temporal=2,
            sample_size=32,
            sample_tsize=8,
            only_encoder=True,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    monkeypatch.setattr(model.vae, "is_distributed_enabled", lambda: False)
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: device))
    monkeypatch.setattr(pipeline_module, "build_batch_2d_rope", lambda **kwargs: (torch.zeros(1), torch.zeros(1)))
    model.model = SimpleNamespace(device=device)
    model.config = SimpleNamespace(attention_head_dim=2, rope_theta=10000)
    model.hf_config = SimpleNamespace(cfg_distilled=False)
    model._tkwrapper = SimpleNamespace(boi_token_id=1, eos_token_id=2, end_recaption_token_id=3, end_answer_token_id=4)
    return model


def _reference(height, width):
    return JointImageInfo(
        vae_image_info=ImageInfo(image_type="vae", image_tensor=torch.zeros(1, 3, height, width), image_token_length=1),
        vision_image_info=ImageInfo(image_type="siglip2", image_tensor=torch.zeros(1, 3), image_token_length=1),
        vision_encoder_kwargs={"spatial_shapes": torch.ones(2), "pixel_attention_mask": torch.ones(1)},
    )


def _prepare(pipeline, mode, generator_kind, cfg):
    pipeline.hf_config.cfg_distilled = cfg == "distilled"
    rows = 2 if cfg == "dual" else 1
    layout = HunyuanPreparedLayout(
        tokenizer_output=TokenizerEncodeOutput(
            tokens=torch.zeros(rows, 8, dtype=torch.long),
            joint_image_slices=[[slice(1, 2), slice(2, 3), slice(3, 4)] for _ in range(rows)],
            gen_timestep_scatter_index=torch.full((rows, 1), 4, dtype=torch.long),
            real_pos=torch.full((rows, 1), 8, dtype=torch.long),
        ),
        rope_image_info=[[] for _ in range(rows)],
        generated_image_info=ImageInfo(image_type="gen_image", image_width=32, image_height=32),
    )
    kwargs: dict[str, object] = {"seed": 0}
    if generator_kind != "seed":
        device = torch.device("cpu") if generator_kind == "cpu-list" else pipeline.device
        generator = torch.Generator(device=device).manual_seed(42)
        # Explicit generators can arrive already advanced; do not reconstruct from seed.
        torch.randn(7, generator=generator, device=device)
        kwargs["generator"] = generator if generator_kind == "single" else [generator]
    if mode in {"hit", "partial"}:
        kwargs["kv_computed_tokens"] = (4,) * (rows - 1) + (4 if mode == "hit" else 3,)
    elif mode == "import":
        kwargs["ar_kv_data"] = {0: {"key": torch.zeros(1), "value": torch.zeros(1)}}
    image = _reference(32, 32)
    return pipeline.prepare_model_inputs(
        prompt="edit",
        image_size=(32, 32),
        prepared_layout=layout,
        guidance_scale=1.0 if cfg == "off" else 5.0,
        batch_cond_image_info=[[image, image, _reference(32, 48)]],
        **kwargs,
    )


def _noise_and_state(pipeline, inputs):
    generator = inputs["generator"]
    request_generator = generator[0] if isinstance(generator, list) else generator
    before = request_generator.get_state().clone()
    noise = HunyuanImage3Text2ImagePipeline.prepare_latents(
        SimpleNamespace(latent_scale_factor=4, scheduler=SimpleNamespace()),
        1,
        2,
        [32, 32],
        torch.bfloat16,
        pipeline.device,
        generator,
    )
    return before, noise, request_generator.get_state()


@pytest.mark.parametrize("mode", ["hit", "import"])
@pytest.mark.parametrize(
    ("generator_kind", "cfg"), [("seed", "off"), ("single", "dual"), ("list", "distilled"), ("cpu-list", "dual")]
)
def test_cache_reuse_preserves_noise_and_generator_state(pipeline, monkeypatch, mode, generator_kind, cfg):
    reference = _noise_and_state(pipeline, _prepare(pipeline, "miss", generator_kind, cfg))
    partial = _noise_and_state(pipeline, _prepare(pipeline, "partial", generator_kind, cfg))
    for expected, actual in zip(reference, partial, strict=True):
        assert torch.equal(expected, actual)

    monkeypatch.setattr(pipeline.vae, "encode", lambda *args, **kwargs: pytest.fail("KV hit must skip VAE encoding"))
    reused_inputs = _prepare(pipeline, mode, generator_kind, cfg)
    assert reused_inputs["cond_vae_images"] is None
    assert reused_inputs["cond_vit_images"] is None
    reused = _noise_and_state(pipeline, reused_inputs)
    for expected, actual in zip(reference, reused, strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("tiling", ["off", "native", "distributed"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_replay_matches_real_posterior_under_autocast(pipeline, monkeypatch, tiling, weight_dtype):
    if pipeline.device.type == "cpu" and weight_dtype == torch.bfloat16:
        pytest.skip("CPU GroupNorm cannot mix FP16 autocast with BF16 weights.")
    vae = pipeline.vae.to(dtype=weight_dtype)
    vae.use_spatial_tiling = tiling != "off"
    if tiling == "distributed":
        monkeypatch.setattr(vae, "is_distributed_enabled", lambda: True)

        def execute(image, operator, broadcast_result):
            # Exercise real split/encode/merge, replacing only tile communication.
            tasks, grid = operator.split(image)
            assert broadcast_result
            tiles = {task.grid_coord: operator.exec(task).to(grid.output_dtype) for task in tasks}
            return operator.merge(tiles, grid)

        vae.distributed_executor = SimpleNamespace(execute=execute)

    for height, width in [(32, 32), (32, 48), (48, 32)]:
        pixels = torch.zeros(1, 3, 1, height, width, device=pipeline.device)
        reference = torch.Generator(device=pipeline.device).manual_seed(0)
        replay = torch.Generator(device=pipeline.device).manual_seed(0)
        with torch.autocast(pipeline.device.type, dtype=torch.float16):
            posterior = vae.encode(pixels).latent_dist
            expected_dtype = weight_dtype if tiling == "distributed" and max(height, width) > 32 else torch.float16
            assert posterior.parameters.dtype == expected_dtype
            assert posterior.mean.shape == (1, 2, 1, height // 4, width // 4)
            posterior.sample(reference)
            vae.advance_posterior_rng(pixels.shape, replay, pipeline.device)
        assert torch.equal(reference.get_state(), replay.get_state())
