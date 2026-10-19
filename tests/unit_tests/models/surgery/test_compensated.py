"""v1_1 — parallel attention/MLP plus a trainable compensatory MLP."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.surgery import (
    CORRECTION_ATTRIBUTE,
    CompensatedParallelLayer,
    build_manifest,
    freeze_except_compensatory,
    get_variant,
    initialize_compensatory_mlp,
)


pytestmark = pytest.mark.unit

HIDDEN = 8


def _config(num_layers: int = 4, **fields: object) -> SimpleNamespace:
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
        "attention_backend": SimpleNamespace(name="flash"),
        "fp32_residual_connection": False,
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


class _Correction(torch.nn.Module):
    """Stand-in for the Megatron MLP branch, same (output, bias) contract."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(hidden)
        self.linear_fc1 = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear_fc2 = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear_fc2(torch.nn.functional.silu(self.linear_fc1(self.norm(value)))), None


def _layer(config: SimpleNamespace, *, correction: torch.nn.Module | None = None) -> CompensatedParallelLayer:
    """Build a v1_1 layer with stub sublayers, bypassing Megatron's constructor."""

    class _Scale(torch.nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.factor = factor

        def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
            return self.factor * value, None

    layer = CompensatedParallelLayer.__new__(CompensatedParallelLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 1
    layer.input_layernorm = torch.nn.Identity()
    layer.pre_mlp_layernorm = torch.nn.Identity()
    layer.self_attention = _Scale(2.0)
    layer.mlp = _Scale(3.0)
    setattr(layer, CORRECTION_ATTRIBUTE, correction if correction is not None else _Correction())
    return layer


# --- registry -------------------------------------------------------------


def test_v1_1_is_registered_and_declares_added_parameters():
    variant = get_variant("v1_1")
    assert variant.slug == "compensated"
    assert variant.layer_module is CompensatedParallelLayer
    assert variant.adds_parameters is True
    assert variant.needs_wiring is True
    assert variant.default_params["init"] == "random"
    # v1_1 is layer-local, so unlike v2/v3/v3_1 it must NOT forbid pipeline parallelism.
    assert "no_pipeline_parallel" not in variant.config_requirements


def test_init_scheme_is_validated():
    with pytest.raises(ValueError, match="init must be one of"):
        get_variant("v1_1").resolve_params({"init": "xavier"})
    assert get_variant("v1_1").resolve_params({"init": "random"})["init"] == "random"


# --- forward --------------------------------------------------------------


def test_forward_adds_the_correction_branch_to_the_parallel_sum():
    layer = _layer(_config())
    value = torch.randn(2, 1, HIDDEN, generator=torch.Generator().manual_seed(0))
    output, context = layer(value)

    correction, _ = getattr(layer, CORRECTION_ATTRIBUTE)(value)
    expected = value + 2.0 * value + 3.0 * value + correction
    torch.testing.assert_close(output, expected)
    assert context is None


def test_zeroing_the_output_projection_makes_v1_1_equal_v1():
    # The branch contributes nothing at step 0, so the run begins from the
    # plain parallel model instead of a perturbed one.
    correction = _Correction()
    torch.nn.init.zeros_(correction.linear_fc2.weight)
    layer = _layer(_config(), correction=correction)

    value = torch.randn(2, 1, HIDDEN, generator=torch.Generator().manual_seed(1))
    output, _ = layer(value)
    torch.testing.assert_close(output, value + 2.0 * value + 3.0 * value)


def test_forward_rejects_a_bias_producing_correction_branch():
    class _Biased(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return value, torch.zeros_like(value)

    layer = _layer(_config(), correction=_Biased())
    with pytest.raises(ValueError, match="bias-free correction branch"):
        layer(torch.randn(2, 1, HIDDEN))


def test_forward_rejects_cross_attention_context_mask():
    layer = _layer(_config())
    with pytest.raises(ValueError, match="cross-attention context masks"):
        layer(torch.randn(2, 1, HIDDEN), context_mask=torch.ones(1))


# --- initialization -------------------------------------------------------


def test_random_init_is_reproducible_and_layer_dependent():
    config = _config()
    first = _Correction()
    second = _Correction()
    initialize_compensatory_mlp(first, config, scheme="random", seed=7, layer_number=3)
    initialize_compensatory_mlp(second, config, scheme="random", seed=7, layer_number=3)
    torch.testing.assert_close(first.linear_fc1.weight, second.linear_fc1.weight)

    other_layer = _Correction()
    initialize_compensatory_mlp(other_layer, config, scheme="random", seed=7, layer_number=4)
    assert not torch.allclose(first.linear_fc1.weight, other_layer.linear_fc1.weight)


def test_unknown_init_scheme_is_rejected():
    with pytest.raises(ValueError, match="Unknown v1_1 correction init scheme"):
        initialize_compensatory_mlp(_Correction(), _config(), scheme="orthogonal", seed=1, layer_number=1)


def test_initialization_leaves_no_uninitialized_parameter():
    # perform_initialization=False means nothing else writes to this branch.
    correction = _Correction()
    for parameter in correction.parameters():
        parameter.data.fill_(float("nan"))
    initialize_compensatory_mlp(correction, _config(), scheme="random", seed=3, layer_number=2)
    assert all(torch.isfinite(parameter).all() for parameter in correction.parameters())
    torch.testing.assert_close(correction.norm.weight, torch.ones_like(correction.norm.weight))


# --- freezing -------------------------------------------------------------


def test_freeze_leaves_only_the_correction_branch_trainable():
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([_layer(_config())])
    model.embedding = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)

    trainable, frozen = freeze_except_compensatory(model)

    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert names and all(f".{CORRECTION_ATTRIBUTE}." in name for name in names)
    assert not any(p.requires_grad for n, p in model.named_parameters() if f".{CORRECTION_ATTRIBUTE}." not in n)
    assert trainable > 0 and frozen > 0


def test_gradients_reach_only_the_correction_branch():
    layer = _layer(_config())
    freeze_except_compensatory(layer)
    value = torch.randn(2, 1, HIDDEN, requires_grad=False)
    output, _ = layer(value)
    output.sum().backward()

    correction = getattr(layer, CORRECTION_ATTRIBUTE)
    assert correction.linear_fc1.weight.grad is not None
    assert correction.linear_fc1.weight.grad.abs().sum() > 0


# --- manifest -------------------------------------------------------------


def test_manifest_records_added_parameters():
    added = [{"name": "decoder.layers.*.compensatory_mlp.linear_fc1.weight", "shape": [HIDDEN, HIDDEN]}]
    manifest = build_manifest(
        variant_id="v1_1",
        variant_slug="compensated",
        equation="y = x + a + m + c",
        params={"init": "random"},
        source_hf_directory="/src",
        model_type="mellum",
        parameter_tensors=2,
        copy_mode="hardlink",
        file_count=3,
        total_bytes=10,
        adds_parameters=added,
    )
    assert manifest["adds_parameters"] == added
    # Inherited tensors are still byte-identical; the new ones are listed separately.
    assert manifest["weights_identical_to_source"] is True


def test_graph_only_variants_record_no_added_parameters():
    manifest = build_manifest(
        variant_id="v1",
        variant_slug="parallel",
        equation="y = x + a + m",
        params={},
        source_hf_directory="/src",
        model_type="mellum",
        parameter_tensors=2,
        copy_mode="hardlink",
        file_count=3,
        total_bytes=10,
    )
    assert manifest["adds_parameters"] == []
