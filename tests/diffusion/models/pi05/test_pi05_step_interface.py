# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract of the π0.5 step interface.

``sample_actions`` is ``initialize_noise`` + ``encode_prefix`` + a loop of
``denoise_step``. A step-level caller (step execution, formed-batch execution)
drives the same three calls itself and keeps each request's prefix K/V between
steps. These tests pin what such a caller relies on:

  * the three calls reproduce ``sample_actions`` bit for bit;
  * ``encode_prefix`` returns state the request owns: it never borrows
    ``model.kv_cache``, and serving other requests leaves it unchanged;
  * ``Pi05KVCache`` is ``(layers, batch, kv_heads, prefix + suffix, head_dim)``
    in the action expert's K/V dtype, a denoising step leaves the prefix slots
    unchanged and rewrites every suffix slot before reading it;
  * a request's prefix slots can be copied into any row of a batch cache, and
    a batched step then gives each row its own request's velocity.

All tests run on CPU with shrunk backbones and random weights.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.pi05 import modeling_pi05
from vllm_omni.diffusion.models.pi05.config import Pi05Config
from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
    GemmaVariantConfig,
    Pi05AdaRMSNorm,
    Pi05ForActionPrediction,
    Pi05KVCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.slow]

_SHRUNK_GEMMA = {
    "gemma_2b": GemmaVariantConfig(2048, 2, 1024, 8, 1, 256),
    "gemma_300m": GemmaVariantConfig(1024, 2, 512, 8, 1, 256),
}
_TEXT_LEN = 3
_PREFIX_LEN = 3 * 256 + _TEXT_LEN


@pytest.fixture(scope="module")
def model():
    """A shrunk model whose chunk depends on the prefix K/V.

    AdaRMS ``dense`` is zero-initialized, which closes every expert gate and
    leaves the chunk independent of the K/V. Open the gates.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr(modeling_pi05, "get_gemma_config", lambda variant: _SHRUNK_GEMMA[variant])
    try:
        model = Pi05ForActionPrediction(
            Pi05Config(max_action_dim=8, max_state_dim=8, chunk_size=4, n_action_steps=4)
        ).eval()
    finally:
        patch.undo()
    generator = torch.Generator().manual_seed(0)
    for module in model.modules():
        if isinstance(module, Pi05AdaRMSNorm) and module.dense is not None:
            module.dense.weight.data.normal_(std=0.02, generator=generator)
    return model


@pytest.fixture(autouse=True)
def _pristine_model(model):
    """Every test starts and ends on the eager whole-request defaults."""
    model.kv_cache = None
    model.cuda_graphs = None
    yield
    model.kv_cache = None
    model.cuda_graphs = None


def _observation(index: int):
    """Model inputs of request ``index``: distinct images, cameras and prompt."""
    images = [torch.full((1, 3, 224, 224), index / 3) for _ in range(3)]
    image_masks = [torch.tensor([camera <= index]) for camera in range(3)]
    lang_tokens = torch.tensor([[index + 1, 4, 2]])
    lang_masks = torch.tensor([[True, True, index == 0]])
    return images, image_masks, lang_tokens, lang_masks


def _prefix(cache: Pi05KVCache) -> tuple[torch.Tensor, torch.Tensor]:
    return cache.key[:, :, :, : cache.prefix_len].clone(), cache.value[:, :, :, : cache.prefix_len].clone()


def _run_steps(model, prefix_pad_masks, kv_cache, noise, num_steps):
    """``sample_actions``' Euler loop over the public step call."""
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        timestep = torch.full((x_t.shape[0],), 1.0 + step * dt, dtype=torch.float32)
        x_t = x_t + dt * model.denoise_step(prefix_pad_masks, kv_cache, x_t, timestep)
    return x_t


def test_initialize_noise_is_the_noise_sample_actions_draws(model):
    observation = _observation(0)
    lang_tokens = observation[2]
    with torch.no_grad():
        noise = model.initialize_noise(lang_tokens, torch.Generator().manual_seed(7))
        from_noise = model.sample_actions(*observation, noise=noise, num_steps=2)
        from_generator = model.sample_actions(*observation, num_steps=2, generator=torch.Generator().manual_seed(7))

    assert noise.shape == (1, model.action_horizon, model.action_dim)
    assert noise.dtype == torch.float32
    assert torch.equal(from_noise, from_generator)


def test_initialize_noise_gives_each_request_its_own_stream(model):
    lang_tokens = torch.zeros(2, _TEXT_LEN, dtype=torch.long)

    def seeded(*seeds):
        return [torch.Generator().manual_seed(seed) for seed in seeds]

    batched = model.initialize_noise(lang_tokens, seeded(3, 5))
    reordered = model.initialize_noise(lang_tokens, seeded(5, 3))
    alone = model.initialize_noise(lang_tokens[:1], seeded(5))

    assert torch.equal(batched[1], alone[0])
    assert torch.equal(reordered[0], alone[0])
    assert not torch.equal(batched[0], batched[1])
    with pytest.raises(ValueError, match="Expected 2 generators"):
        model.initialize_noise(lang_tokens, seeded(3))


def test_step_calls_reproduce_sample_actions(model):
    """The decomposition is exact: same operators, same order, same buffers."""
    observation = _observation(1)
    with torch.no_grad():
        noise = model.initialize_noise(observation[2], torch.Generator().manual_seed(11))
        expected = model.sample_actions(*observation, noise=noise, num_steps=3)
        changed = model.sample_actions(*_observation(2), noise=noise, num_steps=3)
        prefix_pad_masks, kv_cache = model.encode_prefix(*observation)
        stepped = _run_steps(model, prefix_pad_masks, kv_cache, noise, num_steps=3)

    assert not torch.equal(changed, expected), "conditioning must affect the fixed-noise chunk"
    assert torch.equal(stepped, expected)


def test_kv_cache_layout_and_dtype(model):
    with torch.no_grad():
        prefix_pad_masks, kv_cache = model.encode_prefix(*_observation(0))

    text_config = model.paligemma_with_expert.paligemma.config.text_config
    expert_k_proj = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.k_proj
    shape = (
        text_config.num_hidden_layers,
        1,
        text_config.num_key_value_heads,
        _PREFIX_LEN + model.action_horizon,
        text_config.head_dim,
    )
    assert isinstance(kv_cache, Pi05KVCache)
    assert kv_cache.key.shape == kv_cache.value.shape == shape
    assert kv_cache.key.dtype == kv_cache.value.dtype == expert_k_proj.weight.dtype
    assert (kv_cache.batch_size, kv_cache.prefix_len, kv_cache.suffix_len) == (1, _PREFIX_LEN, model.action_horizon)
    assert kv_cache.fits(1, _PREFIX_LEN)
    assert not kv_cache.fits(2, _PREFIX_LEN) and not kv_cache.fits(1, _PREFIX_LEN + 1)
    assert prefix_pad_masks.shape == (1, _PREFIX_LEN)
    assert prefix_pad_masks.dtype == torch.bool


def test_encode_prefix_returns_request_owned_state(model):
    """Serving other requests, on either path, leaves a request's prefix as it was."""
    observation = _observation(0)
    with torch.no_grad():
        noise = model.initialize_noise(observation[2], torch.Generator().manual_seed(1))
        expected = model.sample_actions(*observation, noise=noise, num_steps=2)

        shared = model.new_kv_cache(1, _PREFIX_LEN)
        shared.key.fill_(float("nan"))
        shared.value.fill_(float("nan"))
        model.kv_cache = shared

        mask_a, cache_a = model.encode_prefix(*observation)
        assert cache_a is not shared
        assert torch.isnan(shared.key).all() and torch.isnan(shared.value).all(), "encode_prefix wrote model.kv_cache"
        key_a, value_a = _prefix(cache_a)
        mask_a_before = mask_a.clone()

        mask_b, cache_b = model.encode_prefix(*_observation(1))
        model.sample_actions(*_observation(2), num_steps=1)  # whole-request path, over ``shared``
        _run_steps(model, mask_b, cache_b, noise, num_steps=1)

        stepped = _run_steps(model, mask_a, cache_a, noise, num_steps=2)

    storages = {cache.key.untyped_storage().data_ptr() for cache in (cache_a, cache_b, shared)}
    storages |= {cache.value.untyped_storage().data_ptr() for cache in (cache_a, cache_b, shared)}
    assert len(storages) == 6
    assert mask_a.untyped_storage().data_ptr() != mask_b.untyped_storage().data_ptr()
    assert torch.equal(mask_a, mask_a_before)
    assert all(torch.equal(now, before) for now, before in zip(_prefix(cache_a), (key_a, value_a)))
    assert torch.equal(stepped, expected)


def test_encode_prefix_stays_off_the_cuda_graphs(model, mocker):
    """Graph outputs are static buffers; a request must not keep one."""
    graphs = mocker.Mock(spec=["embed_prefix", "prefix_forward", "denoise_step"])
    model.cuda_graphs = graphs
    with torch.no_grad():
        model.encode_prefix(*_observation(0))

    graphs.embed_prefix.assert_not_called()
    graphs.prefix_forward.assert_not_called()


def test_denoise_step_keeps_the_prefix_and_rewrites_the_suffix(model):
    observation = _observation(0)
    with torch.no_grad():
        prefix_pad_masks, kv_cache = model.encode_prefix(*observation)
        key, value = _prefix(kv_cache)
        x_t = model.initialize_noise(observation[2], torch.Generator().manual_seed(2))
        timestep = torch.tensor([0.5])

        kv_cache.key[:, :, :, _PREFIX_LEN:].zero_()
        kv_cache.value[:, :, :, _PREFIX_LEN:].zero_()
        clean = model.denoise_step(prefix_pad_masks, kv_cache, x_t, timestep)
        kv_cache.key[:, :, :, _PREFIX_LEN:].fill_(float("nan"))
        kv_cache.value[:, :, :, _PREFIX_LEN:].fill_(float("nan"))
        poisoned = model.denoise_step(prefix_pad_masks, kv_cache, x_t, timestep)

    assert all(torch.equal(now, before) for now, before in zip(_prefix(kv_cache), (key, value)))
    assert torch.isfinite(kv_cache.key).all() and torch.isfinite(kv_cache.value).all()
    assert torch.equal(poisoned, clean)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 1, 0)])
def test_batched_step_over_repacked_prefixes_matches_each_request(model, order):
    """Rows carry their request: the batch a prefix sits in, and its row, do not
    change the request's velocity."""
    with torch.no_grad():
        requests = []
        for index in range(3):
            observation = _observation(index)
            prefix_pad_masks, kv_cache = model.encode_prefix(*observation)
            x_t = model.initialize_noise(observation[2], torch.Generator().manual_seed(index))
            timestep = torch.tensor([1.0 - 0.25 * index])
            alone = model.denoise_step(prefix_pad_masks, kv_cache, x_t, timestep)
            requests.append((prefix_pad_masks, kv_cache, x_t, timestep, alone))

        workspace = model.new_kv_cache(len(order), _PREFIX_LEN)
        workspace.key.fill_(float("nan"))
        workspace.value.fill_(float("nan"))
        for row, index in enumerate(order):
            kv_cache = requests[index][1]
            workspace.key[:, row : row + 1, :, :_PREFIX_LEN].copy_(kv_cache.key[:, :, :, :_PREFIX_LEN])
            workspace.value[:, row : row + 1, :, :_PREFIX_LEN].copy_(kv_cache.value[:, :, :, :_PREFIX_LEN])
        batched = model.denoise_step(
            torch.cat([requests[index][0] for index in order]),
            workspace,
            torch.cat([requests[index][2] for index in order]),
            torch.cat([requests[index][3] for index in order]),
        )

    assert batched.shape == (len(order), model.action_horizon, model.action_dim)
    for row, index in enumerate(order):
        torch.testing.assert_close(batched[row : row + 1], requests[index][4], atol=1e-4, rtol=1e-5)
    assert not torch.allclose(requests[0][4], requests[1][4], atol=1e-4, rtol=1e-5)
