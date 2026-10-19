"""Cross-state injection — parallel decoders with a learned gate (v3_2 scalar, v3_3 nonlinear)."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.surgery import (
    GATE_ATTRIBUTE,
    CrossInjectedParallelLayer,
    ScalarCrossGate,
    gate_is_active,
    get_variant,
    initialize_cross_gate,
    wire_model,
)


pytestmark = pytest.mark.unit

HIDDEN = 8


def _config(num_layers: int = 2, **fields: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "num_layers": num_layers,
        "hidden_size": HIDDEN,
        "init_method_std": 0.02,
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
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return self.factor * value, None


def _layer(config: SimpleNamespace, index: int) -> CrossInjectedParallelLayer:
    layer = CrossInjectedParallelLayer.__new__(CrossInjectedParallelLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = index + 1
    layer.input_layernorm = torch.nn.Identity()
    layer.pre_mlp_layernorm = torch.nn.Identity()
    layer.self_attention = _Scale(2.0 + index)
    layer.mlp = _Scale(10.0 + index)
    setattr(layer, GATE_ATTRIBUTE, ScalarCrossGate())
    return layer


class _FakeModel(torch.nn.Module):
    """Minimal stand-in exposing the surfaces wiring touches."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([_layer(config, index) for index in range(2)])
        # A parameter outside the decoder, so freezing is seen to cover it.
        self.embedding = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)


def _model(config: SimpleNamespace) -> _FakeModel:
    return _FakeModel(config)


def _run(model: _FakeModel, value: torch.Tensor) -> torch.Tensor:
    hidden = value
    for layer in model.decoder.layers:
        hidden, _ = layer(hidden)
    return hidden


# --- registry -------------------------------------------------------------


@pytest.mark.parametrize(
    "variant_id,slug,gate", [("v3_2", "cross-scalar", "scalar"), ("v3_3", "cross-nonlinear", "nonlinear")]
)
def test_cross_injection_variants_are_registered(variant_id, slug, gate):
    variant = get_variant(variant_id)
    assert variant.slug == slug
    assert variant.layer_module is CrossInjectedParallelLayer
    assert variant.default_params["gate"] == gate
    assert variant.default_params["direction"] == "both"
    assert variant.default_params["freeze_original"] is False
    assert variant.adds_parameters is True
    assert variant.selection_rule == "contiguous_pairs"
    assert "no_pipeline_parallel" in variant.config_requirements


def test_v3_3_defaults_to_a_quarter_width_gate():
    assert get_variant("v3_3").default_params["gate_dim_ratio"] == 0.25


def test_odd_runs_are_rejected_because_the_formulation_is_pairwise():
    variant = get_variant("v3_2")
    params = variant.resolve_params(None)
    with pytest.raises(ValueError, match="must be even"):
        variant.validate_selection(frozenset({0, 1, 2}), params)
    variant.validate_selection(frozenset({0, 1, 2, 3}), params)


@pytest.mark.parametrize(
    "params,message",
    [
        ({"gate": "quadratic"}, "gate must be one of"),
        ({"direction": "sideways"}, "direction must be one of"),
        ({"gate_dim_ratio": 0}, "positive gate_dim_ratio"),
        ({"freeze_original": "yes"}, "must be a boolean"),
    ],
)
def test_bad_parameters_are_rejected(params, message):
    with pytest.raises(ValueError, match=message):
        get_variant("v3_3").resolve_params(params)


# --- the defining property: starts exactly at v3 --------------------------


def test_zero_initialized_gate_reproduces_plain_adjacent_parallel():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2")

    value = torch.randn(2, 1, HIDDEN, generator=torch.Generator().manual_seed(0))
    output = _run(model, value)

    # v3: o = x + a1 + a2 + m1(x + a1) + m2(x + a2)
    a1, a2 = 2.0 * value, 3.0 * value
    expected = value + a1 + a2 + 10.0 * (value + a1) + 11.0 * (value + a2)
    torch.testing.assert_close(output, expected)


def test_scalar_gate_injects_the_other_branch_attention():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2")
    layers = list(model.decoder.layers)
    with torch.no_grad():
        getattr(layers[0], GATE_ATTRIBUTE).alpha.fill_(0.5)
        getattr(layers[1], GATE_ATTRIBUTE).alpha.fill_(0.25)

    value = torch.randn(2, 1, HIDDEN, generator=torch.Generator().manual_seed(1))
    output = _run(model, value)

    a1, a2 = 2.0 * value, 3.0 * value
    # Branch 1's MLP sees branch 2's attention and vice versa.
    expected = value + a1 + a2 + 10.0 * (value + a1 + 0.5 * a2) + 11.0 * (value + a2 + 0.25 * a1)
    torch.testing.assert_close(output, expected)


def test_injection_enters_before_the_mlp_norm():
    # Norm placement was an explicit decision; a norm that saw only h_i would
    # make alpha behave completely differently.
    config = _config()
    model = _model(config)
    layers = list(model.decoder.layers)

    seen: list[torch.Tensor] = []

    class _RecordingNorm(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            seen.append(value.clone())
            return value

    # Swap before wiring: the plan snapshots module references, so a later
    # replacement would never be seen by the join layer.
    layers[0].pre_mlp_layernorm = _RecordingNorm()
    layers[1].pre_mlp_layernorm = _RecordingNorm()
    wire_model(model, variant="v3_2")
    with torch.no_grad():
        getattr(layers[0], GATE_ATTRIBUTE).alpha.fill_(0.5)

    value = torch.randn(1, 1, HIDDEN, generator=torch.Generator().manual_seed(2))
    _run(model, value)

    a1, a2 = 2.0 * value, 3.0 * value
    torch.testing.assert_close(seen[0], value + a1 + 0.5 * a2)


# --- direction ------------------------------------------------------------


@pytest.mark.parametrize(
    "direction,expected",
    [("both", [True, True]), ("forward", [False, True]), ("backward", [True, False])],
)
def test_direction_selects_which_branches_receive_an_injection(direction, expected):
    assert [gate_is_active(direction, position=p, group_size=2) for p in (0, 1)] == expected


def test_forward_direction_only_feeds_the_second_branch():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2", params={"direction": "forward"})
    layers = list(model.decoder.layers)
    with torch.no_grad():
        getattr(layers[0], GATE_ATTRIBUTE).alpha.fill_(0.5)
        getattr(layers[1], GATE_ATTRIBUTE).alpha.fill_(0.25)

    value = torch.randn(1, 1, HIDDEN, generator=torch.Generator().manual_seed(3))
    output = _run(model, value)

    a1, a2 = 2.0 * value, 3.0 * value
    # Only a1 -> m2 is active; branch 1's MLP is untouched.
    expected = value + a1 + a2 + 10.0 * (value + a1) + 11.0 * (value + a2 + 0.25 * a1)
    torch.testing.assert_close(output, expected)


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError, match="Unknown v3_2/v3_3 direction"):
        gate_is_active("diagonal", position=0, group_size=2)


# --- initialization and wiring --------------------------------------------


def test_scalar_gate_initializes_to_exactly_zero():
    gate = ScalarCrossGate()
    with torch.no_grad():
        gate.alpha.fill_(3.0)
    initialize_cross_gate(gate, _config(), seed=1, layer_number=1)
    assert gate.alpha.item() == 0.0


def test_wiring_marks_one_joining_layer_and_keeps_both_gates():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2")
    layers = list(model.decoder.layers)
    assert layers[0].cross_injection_plan.group is None
    assert len(layers[1].cross_injection_plan.group) == 2
    assert len(layers[1].cross_injection_plan.gates) == 2


def test_layer_refuses_to_run_unwired():
    with pytest.raises(RuntimeError, match="unwired"):
        _layer(_config(), 0)(torch.randn(1, 1, HIDDEN))


def test_freeze_original_leaves_only_the_gates_trainable():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2", params={"freeze_original": True})
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(GATE_ATTRIBUTE in n.split(".") for n in trainable)
    # Parameters outside the decoder must freeze too.
    assert not model.embedding.weight.requires_grad


def test_full_finetune_is_the_default():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2")
    # Nothing is frozen, so v3_2/v3_3 stays directly comparable to the v3 arm.
    assert all(p.requires_grad for p in model.parameters())


def test_gradients_reach_the_gate():
    config = _config()
    model = _model(config)
    wire_model(model, variant="v3_2")
    _run(model, torch.randn(1, 1, HIDDEN)).sum().backward()
    alpha = getattr(model.decoder.layers[0], GATE_ATTRIBUTE).alpha
    # Zero init must not mean zero gradient, or the gate could never learn.
    assert alpha.grad is not None and alpha.grad.abs().item() > 0


# --- manifest -------------------------------------------------------------


def _hf_config() -> dict:
    return {"hidden_size": 2304, "num_hidden_layers": 28}


def test_manifest_describes_the_scalar_gate_not_v1_1_tensors():
    # Regression: the description used to key only on adds_parameters and emit
    # v1_1's compensatory_mlp names for every parameter-adding variant.
    from megatron.bridge.models.surgery import describe_added_parameters

    variant = get_variant("v3_2")
    entries = describe_added_parameters(variant, variant.resolve_params(None), _hf_config())
    names = [entry["hf_parameter"] for entry in entries]
    assert names == ["model.layers.*.cross_gate.alpha"]
    assert all("compensatory_mlp" not in name for name in names)
    assert entries[0]["initialization"] == "zero"


def test_manifest_describes_the_nonlinear_gate_shapes():
    from megatron.bridge.models.surgery import describe_added_parameters

    variant = get_variant("v3_3")
    entries = describe_added_parameters(variant, variant.resolve_params(None), _hf_config())
    by_name = {entry["hf_parameter"].rsplit(".", 2)[-2]: entry for entry in entries}
    inner = 2304 // 4
    assert by_name["v_proj"]["shape"] == [inner, 2304]
    assert by_name["u_proj"]["shape"] == [inner, 2304]
    assert by_name["w_proj"]["shape"] == [2304, inner]
    # Only the output projection is zeroed; the others must be random or the
    # gate would have no gradient path.
    assert by_name["w_proj"]["initialization"] == "zero"
    assert by_name["v_proj"]["initialization"] == "normal"


def test_gate_dim_ratio_changes_the_described_shapes():
    from megatron.bridge.models.surgery import describe_added_parameters

    variant = get_variant("v3_3")
    entries = describe_added_parameters(variant, variant.resolve_params({"gate_dim_ratio": 1.0}), _hf_config())
    assert entries[0]["shape"] == [2304, 2304]


def test_v1_1_description_is_unchanged():
    from megatron.bridge.models.surgery import describe_added_parameters

    variant = get_variant("v1_1")
    entries = describe_added_parameters(variant, variant.resolve_params(None), _hf_config())
    assert all("compensatory_mlp" in entry["hf_parameter"] for entry in entries)


def test_graph_only_variants_describe_nothing():
    from megatron.bridge.models.surgery import describe_added_parameters

    variant = get_variant("v1")
    assert describe_added_parameters(variant, variant.resolve_params(None), _hf_config()) == []
