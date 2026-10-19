"""v1_1 Hugging Face compatibility: graph selection, forward, and save/reload parity.

The exported checkpoint is handed to other teams, so the risk being tested here
is not a crash but a *silent* one: a checkpoint that loads cleanly into the
original sequential graph and reports plausible-but-wrong evaluation numbers.
"""

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from megatron.bridge.models.mellum.modeling_mellum.hf_reference import (
    SUPPORTED_SURGERY_VARIANTS,
    MellumConfig,
    MellumForCausalLM,
    register_hf_fallback,
    resolve_surgery_variant,
)


pytestmark = pytest.mark.unit

register_hf_fallback()


def _toy_config(**extra: object) -> MellumConfig:
    fields: dict[str, object] = {
        "hidden_size": 32,
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 0.001,
        "initializer_range": 0.02,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "mlp_layer_types": ["sparse", "sparse"],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    }
    fields.update(extra)
    return MellumConfig(**fields)


# --- graph selection ------------------------------------------------------


def test_missing_marker_defaults_to_the_original_graph():
    assert resolve_surgery_variant(_toy_config()) == "v0"


def test_legacy_architecture_string_still_selects_v1():
    assert resolve_surgery_variant(_toy_config(megatron_bridge_architecture="parallel_attention_moe")) == "v1"


def test_explicit_variant_marker_wins():
    assert resolve_surgery_variant(_toy_config(megatron_bridge_surgery_variant="v1_1")) == "v1_1"


def test_historical_v4_marker_selects_canonical_v1_1():
    assert resolve_surgery_variant(_toy_config(megatron_bridge_surgery_variant="v4")) == "v1_1"


@pytest.mark.parametrize("variant", ["v2", "v3", "v3_1", "v5_1", "v5_2"])
def test_cross_layer_variants_are_refused_rather_than_run_as_sequential(variant):
    # These rearrange across layer boundaries and cannot be expressed per-layer.
    # Loading must fail loudly; the dangerous outcome is a plausible wrong number.
    with pytest.raises(ValueError, match="silently evaluate a different computation graph"):
        MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant=variant))
    assert variant not in SUPPORTED_SURGERY_VARIANTS


# --- forward --------------------------------------------------------------


def _layer_outputs(model: MellumForCausalLM, hidden: torch.Tensor):
    layer = model.model.layers[0]
    rotary_dim = model.config.head_dim
    angles = torch.zeros(1, hidden.shape[1], rotary_dim)
    rotary = (angles.cos(), angles.sin())
    mask = torch.zeros(hidden.shape[0], 1, hidden.shape[1], hidden.shape[1])
    return layer, rotary, mask


def test_v1_1_forward_adds_the_correction_branch_to_the_parallel_sum():
    torch.manual_seed(0)
    model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1_1"))
    model.eval()
    hidden = torch.randn(1, 4, model.config.hidden_size)
    layer, rotary, mask = _layer_outputs(model, hidden)

    with torch.no_grad():
        attention, _ = layer.self_attn(layer.input_layernorm(hidden), rotary, mask)
        expected = (
            hidden + attention + layer.mlp(layer.post_attention_layernorm(hidden)) + layer.compensatory_mlp(hidden)
        )
        actual = layer(hidden, mask=mask, rotary=rotary)
    torch.testing.assert_close(actual, expected)


def test_v1_1_differs_from_both_v0_and_v1():
    # If v1_1 collapsed to either, the arm would be silently meaningless.
    torch.manual_seed(1)
    hidden = torch.randn(1, 4, 32)
    outputs = {}
    for variant in ("v0", "v1", "v1_1"):
        torch.manual_seed(1)
        model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant=variant)).eval()
        layer, rotary, mask = _layer_outputs(model, hidden)
        with torch.no_grad():
            outputs[variant] = layer(hidden, mask=mask, rotary=rotary)

    assert not torch.allclose(outputs["v1_1"], outputs["v0"], atol=1e-5)
    assert not torch.allclose(outputs["v1_1"], outputs["v1"], atol=1e-5)


def test_zeroing_the_correction_output_recovers_v1_exactly():
    # The degenerate check: with c_theta == 0 the graph must equal plain parallel.
    torch.manual_seed(2)
    model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1_1")).eval()
    for layer in model.model.layers:
        torch.nn.init.zeros_(layer.compensatory_mlp.down_proj.weight)

    # Copy the shared weights across rather than reseeding: building v1_1 draws
    # extra randomness for the correction branch, so two seeded constructions
    # would not share attention and MLP weights.
    parallel = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1")).eval()
    shared = {k: v for k, v in model.state_dict().items() if "compensatory_mlp" not in k}
    missing, unexpected = parallel.load_state_dict(shared, strict=False)
    assert not missing and not unexpected

    tokens = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens).logits, parallel(tokens).logits)


# --- save / reload round trip ---------------------------------------------


def test_round_trip_preserves_every_parameter_and_the_logits(tmp_path):
    torch.manual_seed(3)
    model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1_1")).eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        before = model(tokens).logits

    model.save_pretrained(tmp_path, safe_serialization=True)

    # The marker must survive serialization, or a reload silently builds v0.
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["megatron_bridge_surgery_variant"] == "v1_1"

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert type(reloaded).__name__ == "MellumForCausalLM"

    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    assert set(original_state) == set(reloaded_state)
    correction_keys = [key for key in reloaded_state if "compensatory_mlp" in key]
    assert len(correction_keys) == 3 * model.config.num_hidden_layers
    for key, value in original_state.items():
        torch.testing.assert_close(value, reloaded_state[key], msg=f"{key} changed across the round trip")

    with torch.no_grad():
        after = reloaded(tokens).logits
    torch.testing.assert_close(before, after)


def test_round_trip_config_is_loadable_through_autoconfig(tmp_path):
    model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1_1"))
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = AutoConfig.from_pretrained(tmp_path)
    assert resolve_surgery_variant(config) == "v1_1"


def test_compensatory_branch_matches_ivan_shape_and_names():
    model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1_1"))
    branch = model.model.layers[0].compensatory_mlp
    hidden = model.config.hidden_size
    assert branch.up_proj.in_features == hidden
    assert branch.up_proj.out_features == hidden
    assert branch.down_proj.in_features == hidden
    assert branch.down_proj.out_features == hidden
    assert set(branch.state_dict()) == {"norm.weight", "up_proj.weight", "down_proj.weight"}


def test_non_v1_1_checkpoints_carry_no_correction_weights(tmp_path):
    model = MellumForCausalLM(_toy_config(megatron_bridge_surgery_variant="v1"))
    assert not any("compensatory_mlp" in key for key in model.state_dict())
