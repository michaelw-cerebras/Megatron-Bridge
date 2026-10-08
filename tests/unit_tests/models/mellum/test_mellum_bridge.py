import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from megatron.bridge import AutoBridge
from megatron.bridge.models.mellum import MellumBridge
from megatron.bridge.models.mellum.modeling_mellum.layer_specs import MellumTopKRouter, yarn_frequency_ratio


pytestmark = pytest.mark.unit


@pytest.fixture
def hf_config():
    return AutoConfig.for_model(
        "mellum",
        architectures=["MellumForCausalLM"],
        hidden_size=64,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        initializer_range=0.02,
        dtype="bfloat16",
        use_cache=False,
        bos_token_id=0,
        eos_token_id=28,
        pad_token_id=None,
        sliding_window=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        mlp_layer_types=["sparse"] * 4,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 128,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.2772588722239782,
            },
        },
    )


def test_registration_and_nonstandard_head_dimension(hf_config):
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    assert provider.hidden_size == 64
    assert provider.kv_channels == 32  # deliberately not hidden_size / heads
    assert provider.window_attn_skip_freq == [1, 1, 1, 0]
    assert provider.window_size == (15, 0)
    assert provider.moe_ffn_hidden_size == 32
    assert not provider.apply_rope_fusion
    assert provider.moe_router_dtype is None
    assert provider.moe_router_pre_softmax


def test_yarn_matches_hf_frequencies_and_concentration(hf_config):
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    inv_freq, concentration = ROPE_INIT_FUNCTIONS["yarn"](hf_config, torch.device("cpu"), layer_type="full_attention")
    ordinary = 1 / (provider.rotary_base ** (torch.arange(0, provider.kv_channels, 2).float() / provider.kv_channels))
    torch.testing.assert_close(ordinary * yarn_frequency_ratio(provider)[: provider.kv_channels // 2], inv_freq)
    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_concentration_factor

    assert _yarn_get_concentration_factor(16.0, provider.yarn_mscale, 0.0) == pytest.approx(concentration)


def test_export_preserves_hybrid_rope(hf_config):
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    result = MellumBridge.megatron_to_hf_config(provider)
    assert result["head_dim"] == 32
    assert result["norm_topk_prob"]
    assert result["layer_types"] == hf_config.layer_types
    assert result["rope_parameters"]["full_attention"]["attention_factor"] == pytest.approx(1.2772588722239782)


def test_hf_config_save_reload(hf_config, tmp_path):
    hf_config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    assert restored.rope_parameters == hf_config.rope_parameters
    assert restored.layer_types == hf_config.layer_types


def test_router_scores_in_fp32_but_returns_gate_dtype(monkeypatch):
    from megatron.core.transformer.moe.router import TopKRouter

    def route(self, logits, padding_mask=None):
        assert logits.dtype == torch.float32
        return logits.softmax(-1), torch.ones_like(logits, dtype=torch.bool)

    monkeypatch.setattr(TopKRouter, "routing", route)
    router = MellumTopKRouter.__new__(MellumTopKRouter)
    torch.nn.Module.__init__(router)
    logits = torch.tensor([[1, 2, 3]], dtype=torch.bfloat16, requires_grad=True)
    probs, routing_map = router.routing(logits)
    assert probs.dtype == torch.bfloat16
    assert routing_map.dtype == torch.bool
    probs[0, 0].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "field,value",
    [("norm_topk_prob", False), ("hidden_act", "gelu"), ("mlp_layer_types", ["dense"] * 4), ("attention_bias", True)],
)
def test_unsupported_variants_fail_explicitly(hf_config, field, value):
    hf_config = copy.deepcopy(hf_config)
    assert hasattr(hf_config, field)
    setattr(hf_config, field, value)
    with pytest.raises(ValueError):
        MellumBridge().provider_bridge(SimpleNamespace(config=hf_config))
