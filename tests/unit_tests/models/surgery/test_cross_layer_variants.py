"""v2 (shifted) and v3 (adjacent) — the variants that couple neighbouring layers."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer

from megatron.bridge.models.surgery import (
    AdjacentDecoderParallelLayer,
    ShiftedAttentionMLPLayer,
    apply_surgery,
    get_variant,
    make_pre_wrap_hook,
    wire_model,
    wrap_spec,
)


pytestmark = pytest.mark.unit


class _Base(TransformerLayer):
    """Stand-in for a model family's own layer class."""


def _base_spec() -> ModuleSpec:
    return ModuleSpec(module=_Base, params={}, submodules=SimpleNamespace(marker=object()))


def _config(num_layers: int = 4, **fields: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "num_layers": num_layers,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "apply_residual_connection_post_layernorm": False,
        "hidden_dropout": 0.0,
        "mlp_chunks_for_prefill": 1,
        "mlp_chunks_for_training": 1,
        "cuda_graph_impl": "none",
        "inference_fuse_tp_communication": False,
        "recompute_granularity": None,
        "cpu_offloading": False,
        "overlap_moe_expert_parallel_comm": False,
        "attention_backend": SimpleNamespace(name="flash"),
        "fp32_residual_connection": False,
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


class _Scale(torch.nn.Module):
    """Deterministic stand-in for attention or an MLP."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return self.factor * value, None


class _Shift(torch.nn.Module):
    """Deterministic stand-in for a norm."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.offset


def _fake_layer(layer_class: type, index: int, config: SimpleNamespace) -> torch.nn.Module:
    """Build a layer with stub sublayers, bypassing Megatron's constructor."""
    layer = layer_class.__new__(layer_class)
    torch.nn.Module.__init__(layer)
    layer.config = config
    # Distinct factors per layer so a mis-wired neighbour changes the result.
    layer.input_layernorm = _Shift(0.0)
    layer.self_attention = _Scale(2.0 + index)
    layer.pre_mlp_layernorm = _Shift(0.0)
    layer.mlp = _Scale(10.0 + index)
    return layer


def _fake_model(layer_class: type, num_layers: int, config: SimpleNamespace) -> SimpleNamespace:
    layers = torch.nn.ModuleList([_fake_layer(layer_class, index, config) for index in range(num_layers)])
    return SimpleNamespace(config=config, decoder=SimpleNamespace(layers=layers))


def _run_decoder(model: SimpleNamespace, value: torch.Tensor) -> torch.Tensor:
    """Drive the layers the way TransformerBlock does."""
    hidden = value
    for layer in model.decoder.layers:
        hidden, _ = layer(hidden)
    return hidden


def _reference_sequential(value: torch.Tensor, num_layers: int) -> torch.Tensor:
    hidden = value
    for index in range(num_layers):
        hidden = hidden + (2.0 + index) * hidden
        hidden = hidden + (10.0 + index) * hidden
    return hidden


# --- registry -------------------------------------------------------------


def test_cross_layer_variants_are_registered_with_wiring():
    for variant_id, slug, layer_class in (
        ("v2", "shifted", ShiftedAttentionMLPLayer),
        ("v3", "adjacent", AdjacentDecoderParallelLayer),
    ):
        variant = get_variant(variant_id)
        assert variant.slug == slug
        assert variant.layer_module is layer_class
        assert variant.needs_wiring is True
        assert "no_pipeline_parallel" in variant.config_requirements


def test_pipeline_parallel_is_rejected_for_cross_layer_variants():
    with pytest.raises(ValueError, match="no_pipeline_parallel"):
        wrap_spec(_base_spec(), variant="v2")(_config(4, pipeline_model_parallel_size=2))


def test_expert_parallel_overlap_is_rejected_for_cross_layer_variants():
    with pytest.raises(ValueError, match="no_expert_parallel_overlap"):
        wrap_spec(_base_spec(), variant="v3")(_config(4, overlap_moe_expert_parallel_comm=True))


def test_v1_still_allows_pipeline_parallel():
    # v1 is layer-local, so PP is fine; only the cross-layer variants need PP=1.
    assert "no_pipeline_parallel" not in get_variant("v1").config_requirements


def test_shifted_requires_a_contiguous_run():
    with pytest.raises(ValueError, match="contiguous run"):
        wrap_spec(_base_spec(), variant="v2", params={"layers": {"kind": "indices", "indices": [0, 2]}})(_config(4))


def test_adjacent_requires_whole_groups():
    with pytest.raises(ValueError, match="multiple of group_size"):
        wrap_spec(
            _base_spec(),
            variant="v3",
            params={"layers": {"kind": "indices", "indices": [0, 1, 2]}, "group_size": 2},
        )(_config(4))


def test_adjacent_group_size_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="group_size"):
        get_variant("v3").resolve_params({"group_size": 0})


# --- v2 shifted -----------------------------------------------------------


def test_shifted_forward_matches_the_recurrence():
    config = _config(3)
    model = _fake_model(ShiftedAttentionMLPLayer, 3, config)
    wire_model(model, variant="v2")

    value = torch.tensor([1.0])
    output = _run_decoder(model, value)

    # u_0 = x + a_0(x); u_i = u_{i-1} + m_{i-1}(u_{i-1}) + a_i(u_{i-1}); y = u_2 + m_2(u_2)
    u = value + 2.0 * value
    u = u + 10.0 * u + 3.0 * u
    u = u + 11.0 * u + 4.0 * u
    expected = u + 12.0 * u
    torch.testing.assert_close(output, expected)


def test_shifted_run_of_one_layer_equals_the_original_layer():
    # A single-layer run degenerates to attention-then-MLP, which is the
    # strongest available check that the schedule drops nothing.
    config = _config(1)
    model = _fake_model(ShiftedAttentionMLPLayer, 1, config)
    wire_model(model, variant="v2")

    value = torch.tensor([1.0])
    torch.testing.assert_close(_run_decoder(model, value), _reference_sequential(value, 1))


def test_shifted_uses_every_sublayer_exactly_once():
    config = _config(4)
    model = _fake_model(ShiftedAttentionMLPLayer, 4, config)
    wire_model(model, variant="v2")
    layers = list(model.decoder.layers)

    plans = [layer.shift_plan for layer in layers]
    assert plans[0].previous_mlp is None
    for index in range(1, 4):
        assert plans[index].previous_mlp.mlp is layers[index - 1].mlp
    assert [plan.closes_run for plan in plans] == [False, False, False, True]


class _Nonlinear(torch.nn.Module):
    """Nonlinear stand-in: a linear stub cannot distinguish a mis-ordered schedule."""

    def __init__(self, seed: int, hidden: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(hidden, hidden, generator=generator) / hidden**0.5)

    def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return torch.tanh(value @ self.weight), None


def _nonlinear_model(layer_class: type, num_layers: int, hidden: int, config: SimpleNamespace) -> SimpleNamespace:
    layers = []
    for index in range(num_layers):
        layer = layer_class.__new__(layer_class)
        torch.nn.Module.__init__(layer)
        layer.config = config
        layer.input_layernorm = torch.nn.Identity()
        layer.pre_mlp_layernorm = torch.nn.Identity()
        layer.self_attention = _Nonlinear(100 + index, hidden)
        layer.mlp = _Nonlinear(200 + index, hidden)
        layers.append(layer)
    return SimpleNamespace(config=config, decoder=SimpleNamespace(layers=torch.nn.ModuleList(layers)))


def _attn(model: SimpleNamespace, index: int, value: torch.Tensor) -> torch.Tensor:
    return model.decoder.layers[index].self_attention(value)[0]


def _mlp(model: SimpleNamespace, index: int, value: torch.Tensor) -> torch.Tensor:
    return model.decoder.layers[index].mlp(value)[0]


def test_shifted_matches_a_literal_transcription_over_many_nonlinear_layers():
    hidden, num_layers = 8, 6
    config = _config(num_layers)
    model = _nonlinear_model(ShiftedAttentionMLPLayer, num_layers, hidden, config)
    wire_model(model, variant="v2")

    value = torch.randn(3, 1, hidden, generator=torch.Generator().manual_seed(0))

    # Literal transcription of the documented recurrence, written independently
    # of the layer implementation.
    u = value + _attn(model, 0, value)
    for index in range(1, num_layers):
        u = u + _mlp(model, index - 1, u) + _attn(model, index, u)
    expected = u + _mlp(model, num_layers - 1, u)

    torch.testing.assert_close(_run_decoder(model, value), expected)


def test_adjacent_matches_a_literal_transcription_over_many_nonlinear_layers():
    hidden, num_layers = 8, 6
    config = _config(num_layers)
    model = _nonlinear_model(AdjacentDecoderParallelLayer, num_layers, hidden, config)
    wire_model(model, variant="v3", params={"group_size": 3})

    value = torch.randn(3, 1, hidden, generator=torch.Generator().manual_seed(1))

    hidden_state = value
    for start in range(0, num_layers, 3):
        total = hidden_state
        for index in range(start, start + 3):
            attention = _attn(model, index, hidden_state)
            total = total + attention + _mlp(model, index, hidden_state + attention)
        hidden_state = total

    torch.testing.assert_close(_run_decoder(model, value), hidden_state)


@pytest.mark.parametrize(
    "variant_id,layer_class", [("v2", ShiftedAttentionMLPLayer), ("v3", AdjacentDecoderParallelLayer)]
)
def test_cross_layer_variants_are_not_secretly_the_original_graph(variant_id, layer_class):
    # A rearrangement that silently collapsed back to the sequential graph would
    # look like a suspiciously good training curve rather than an error.
    hidden, num_layers = 8, 6
    config = _config(num_layers)
    model = _nonlinear_model(layer_class, num_layers, hidden, config)
    wire_model(model, variant=variant_id)

    value = torch.randn(3, 1, hidden, generator=torch.Generator().manual_seed(2))

    sequential = value
    for index in range(num_layers):
        sequential = sequential + _attn(model, index, sequential)
        sequential = sequential + _mlp(model, index, sequential)

    surgical = _run_decoder(model, value)
    assert not torch.allclose(surgical, sequential, atol=1e-5)


def test_shifted_runs_every_sublayer_exactly_once_under_a_call_counter():
    hidden, num_layers = 4, 5
    config = _config(num_layers)
    model = _nonlinear_model(ShiftedAttentionMLPLayer, num_layers, hidden, config)
    wire_model(model, variant="v2")

    counts: dict[str, int] = {}

    def counting_hook(name: str):
        def hook(module: torch.nn.Module, args: object, output: object) -> None:
            counts[name] = counts.get(name, 0) + 1

        return hook

    for index, layer in enumerate(model.decoder.layers):
        layer.self_attention.register_forward_hook(counting_hook(f"a{index}"))
        layer.mlp.register_forward_hook(counting_hook(f"m{index}"))

    _run_decoder(model, torch.randn(2, 1, hidden))

    # Dropping or double-counting an MLP is the failure mode a shifted schedule
    # is most prone to, and it would move the loss without raising.
    assert counts == {f"{kind}{index}": 1 for index in range(num_layers) for kind in ("a", "m")}


def test_shifted_layer_refuses_to_run_unwired():
    layer = _fake_layer(ShiftedAttentionMLPLayer, 0, _config(1))
    with pytest.raises(RuntimeError, match="unwired"):
        layer(torch.tensor([1.0]))


# --- v3 adjacent ----------------------------------------------------------


def test_adjacent_forward_sums_group_deltas():
    config = _config(2)
    model = _fake_model(AdjacentDecoderParallelLayer, 2, config)
    wire_model(model, variant="v3")

    value = torch.tensor([1.0])
    output = _run_decoder(model, value)

    # delta_i(x) = a_i(x) + m_i(x + a_i(x)); y = x + delta_0(x) + delta_1(x)
    delta_0 = 2.0 * value + 10.0 * (value + 2.0 * value)
    delta_1 = 3.0 * value + 11.0 * (value + 3.0 * value)
    torch.testing.assert_close(output, value + delta_0 + delta_1)


def test_adjacent_group_of_one_equals_the_original_decoder():
    config = _config(2)
    model = _fake_model(AdjacentDecoderParallelLayer, 2, config)
    wire_model(model, variant="v3", params={"group_size": 1})

    value = torch.tensor([1.0])
    torch.testing.assert_close(_run_decoder(model, value), _reference_sequential(value, 2))


def test_adjacent_passthrough_members_do_not_touch_the_input():
    config = _config(4)
    model = _fake_model(AdjacentDecoderParallelLayer, 4, config)
    wire_model(model, variant="v3")
    layers = list(model.decoder.layers)

    assert layers[0].adjacent_plan.group is None
    assert layers[2].adjacent_plan.group is None
    assert len(layers[1].adjacent_plan.group) == 2
    assert len(layers[3].adjacent_plan.group) == 2

    value = torch.tensor([5.0])
    passed_through, _ = layers[0](value)
    torch.testing.assert_close(passed_through, value)


def test_adjacent_group_of_three_sums_three_deltas():
    config = _config(3)
    model = _fake_model(AdjacentDecoderParallelLayer, 3, config)
    wire_model(model, variant="v3", params={"group_size": 3})

    value = torch.tensor([1.0])
    expected = value
    for index in range(3):
        attention = (2.0 + index) * value
        expected = expected + attention + (10.0 + index) * (value + attention)
    torch.testing.assert_close(_run_decoder(model, value), expected)


def test_adjacent_layer_refuses_to_run_unwired():
    layer = _fake_layer(AdjacentDecoderParallelLayer, 0, _config(1))
    with pytest.raises(RuntimeError, match="unwired"):
        layer(torch.tensor([1.0]))


# --- wiring plumbing ------------------------------------------------------


def test_wiring_keeps_the_state_dict_identical_to_the_original():
    # The whole toolkit rests on this: cross-layer references must not register
    # a second owner for any module.
    config = _config(4)
    plain = _fake_model(_Base, 4, config)
    wired = _fake_model(ShiftedAttentionMLPLayer, 4, config)
    before = set(wired.decoder.layers.state_dict())
    wire_model(wired, variant="v2")
    after = set(wired.decoder.layers.state_dict())

    assert after == before
    assert after == set(plain.decoder.layers.state_dict())


def test_wiring_rejects_a_partial_decoder():
    config = _config(8)
    model = _fake_model(ShiftedAttentionMLPLayer, 4, config)
    with pytest.raises(ValueError, match="expects all 8 layers on one stage"):
        wire_model(model, variant="v2")


def test_wiring_rejects_a_variant_that_needs_none():
    model = _fake_model(_Base, 2, _config(2))
    with pytest.raises(ValueError, match="does not use cross-layer wiring"):
        wire_model(model, variant="v1")


def test_pre_wrap_hook_rejects_virtual_pipeline_chunks():
    config = _config(2)
    hook = make_pre_wrap_hook(variant="v2")
    with pytest.raises(ValueError, match="single model chunk"):
        hook([_fake_model(ShiftedAttentionMLPLayer, 2, config), _fake_model(ShiftedAttentionMLPLayer, 2, config)])


def test_pre_wrap_hook_wires_and_returns_the_model_list():
    config = _config(2)
    model = _fake_model(ShiftedAttentionMLPLayer, 2, config)
    assert make_pre_wrap_hook(variant="v2")([model]) == [model]
    assert model.decoder.layers[1].shift_plan.closes_run is True


class _Provider(SimpleNamespace):
    """Minimal provider exposing the hook registration Bridge provides."""

    def __init__(self, **fields: object) -> None:
        super().__init__(**fields)
        self.transformer_layer_spec = None
        self.hooks: list[object] = []

    def register_pre_wrap_hook(self, hook: object, prepend: bool = False) -> None:
        self.hooks.append(hook)


def test_apply_surgery_registers_a_hook_only_for_cross_layer_variants():
    layer_local = _Provider(**vars(_config(4)))
    apply_surgery(layer_local, base_spec=_base_spec(), variant="v1")
    assert layer_local.transformer_layer_spec is not None
    assert layer_local.hooks == []

    cross_layer = _Provider(**vars(_config(4)))
    described = apply_surgery(cross_layer, base_spec=_base_spec(), variant="v2")
    assert len(cross_layer.hooks) == 1
    assert described["id"] == "v2"
    assert described["parallelization_ratio"] == pytest.approx(1.0)


def test_apply_surgery_registers_no_hook_when_nothing_is_rewritten():
    provider = _Provider(**vars(_config(4)))
    apply_surgery(provider, base_spec=_base_spec(), variant="v2", params={"layers": "none"})
    assert provider.hooks == []


def test_degenerate_selector_leaves_cross_layer_variants_inert():
    base = _base_spec()
    for variant_id in ("v2", "v3"):
        block = wrap_spec(base, variant=variant_id, params={"layers": "none"})(_config(4))
        assert [spec is base for spec in block.layer_specs] == [True] * 4
