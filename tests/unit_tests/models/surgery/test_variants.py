from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer

from megatron.bridge.models.surgery import (
    ParallelAttentionMLPLayer,
    describe_surgery,
    get_variant,
    list_variants,
    wrap_spec,
)


pytestmark = pytest.mark.unit


class _Base(TransformerLayer):
    """Stand-in for a model family's own layer class."""


def _base_spec() -> ModuleSpec:
    return ModuleSpec(module=_Base, params={"tag": "family-owned"}, submodules=SimpleNamespace(marker=object()))


def _bare_layer() -> ParallelAttentionMLPLayer:
    """Build a v1 layer without Megatron's heavy constructor.

    ``MegatronModule.__call__`` reads ``config.cuda_graph_impl`` before dispatching
    to ``forward``, so the stub config must define it.
    """
    layer = ParallelAttentionMLPLayer.__new__(ParallelAttentionMLPLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        cuda_graph_impl="none",
        fp32_residual_connection=False,
        inference_fuse_tp_communication=False,
    )
    return layer


def _config(num_layers: int = 8, **fields: object) -> SimpleNamespace:
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
        "attention_backend": SimpleNamespace(name="flash"),
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def test_variant_numbers_are_stable_and_documented():
    ids = [variant.id for variant in list_variants()]
    assert ids == sorted(ids)
    for variant in list_variants():
        assert variant.id.startswith("v")
        assert variant.slug and variant.equation and variant.summary


def test_v1_rewrites_only_the_layer_class():
    base = _base_spec()
    surgical = get_variant("v1").make_layer_spec(base)
    assert surgical.module is ParallelAttentionMLPLayer
    assert surgical.params == base.params
    # Submodules are shared by reference: surgery must not alter what a layer is
    # made of, only how its sublayers are wired.
    assert surgical.submodules is base.submodules


def test_v0_refuses_to_rewrite_layers():
    with pytest.raises(ValueError, match="does not rewrite individual layers"):
        get_variant("v0").make_layer_spec(_base_spec())


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown surgery variant"):
        get_variant("v99")


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="has no parameter"):
        get_variant("v1").resolve_params({"layer": "all"})


def test_wrap_spec_applies_the_variant_to_selected_layers_only():
    base = _base_spec()
    build = wrap_spec(base, variant="v1", params={"layers": {"kind": "every_k", "k": 4}})
    block = build(_config(8))
    assert isinstance(block, TransformerBlockSubmodules)
    modules = [spec.module for spec in block.layer_specs]
    assert modules == [
        ParallelAttentionMLPLayer,
        _Base,
        _Base,
        _Base,
        ParallelAttentionMLPLayer,
        _Base,
        _Base,
        _Base,
    ]


def test_wrap_spec_supplies_the_final_layer_norm():
    # TransformerBlockSubmodules.layer_norm defaults to None, which would silently
    # drop the decoder's final norm and quietly change the model.
    block = wrap_spec(_base_spec(), variant="v1")(_config(4))
    assert block.layer_norm is not None


def test_wrap_spec_accepts_a_callable_base_spec():
    calls: list[object] = []

    def family_spec(config: object) -> ModuleSpec:
        calls.append(config)
        return _base_spec()

    config = _config(2)
    block = wrap_spec(family_spec, variant="v1")(config)
    assert calls == [config]
    assert all(spec.module is ParallelAttentionMLPLayer for spec in block.layer_specs)


def test_degenerate_selector_returns_the_untouched_family_spec():
    # The invariant every future variant inherits: with an empty layer selector a
    # variant must be indistinguishable from the original model.
    base = _base_spec()
    block = wrap_spec(base, variant="v1", params={"layers": "none"})(_config(5))
    assert [spec is base for spec in block.layer_specs] == [True] * 5

    v0_block = wrap_spec(base, variant="v0")(_config(5))
    assert [spec is base for spec in v0_block.layer_specs] == [True] * 5


def test_requirements_are_only_enforced_when_layers_are_rewritten():
    unsupported = _config(4, hidden_dropout=0.1)
    with pytest.raises(ValueError, match="no_hidden_dropout"):
        wrap_spec(_base_spec(), variant="v1")(unsupported)
    # Nothing is rewritten, so the unsupported feature is irrelevant.
    wrap_spec(_base_spec(), variant="v1", params={"layers": "none"})(unsupported)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("apply_residual_connection_post_layernorm", True, "pre_norm_residual"),
        ("mlp_chunks_for_training", 2, "no_mlp_chunking"),
        ("cuda_graph_impl", "local", "no_cuda_graphs"),
        ("inference_fuse_tp_communication", True, "no_fused_tp_inference"),
        ("recompute_granularity", "selective", "no_selective_recompute"),
        ("cpu_offloading", True, "no_cpu_offloading"),
        ("attention_backend", SimpleNamespace(name="local"), "non_local_attention_backend"),
    ],
)
def test_each_unmet_requirement_fails_closed(field, value, message):
    with pytest.raises(ValueError, match=message):
        wrap_spec(_base_spec(), variant="v1")(_config(4, **{field: value}))


def test_describe_surgery_records_the_parallelization_ratio():
    config = _config(8, window_attn_skip_freq=[1, 1, 1, 0, 1, 1, 1, 0])
    described = describe_surgery(
        config,
        variant="v1",
        params={"layers": {"kind": "config_list", "field": "window_attn_skip_freq"}},
    )
    assert described["id"] == "v1"
    assert described["slug"] == "parallel"
    assert described["rewritten_layers"] == [0, 1, 2, 4, 5, 6]
    assert described["parallelization_ratio"] == pytest.approx(0.75)


def test_pipeline_parallel_slices_specs_to_the_local_stage(monkeypatch):
    # layer_specs is local to the pipeline stage: index i is global layer i+offset.
    # Getting this wrong builds every layer on every stage.
    import megatron.bridge.models.surgery.spec as spec_module

    monkeypatch.setattr(spec_module, "get_transformer_layer_offset", lambda config, vp_stage=None: 4)
    monkeypatch.setattr(spec_module, "get_num_layers_to_build", lambda config, vp_stage=None: 4)
    build = wrap_spec(_base_spec(), variant="v1", params={"layers": {"kind": "indices", "indices": [0, 5]}})
    block = build(_config(8, pipeline_model_parallel_size=2))
    assert len(block.layer_specs) == 4
    assert [spec.module for spec in block.layer_specs] == [
        _Base,
        ParallelAttentionMLPLayer,
        _Base,
        _Base,
    ]


def test_v1_forward_feeds_the_same_input_to_attention_and_mlp():
    class Add(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.value

    class Attention(torch.nn.Module):
        def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
            return 2 * value, None

    class Mlp(torch.nn.Module):
        def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
            return 3 * value, None

    layer = _bare_layer()
    layer.input_layernorm = Add(10)
    layer.self_attention = Attention()
    layer.pre_mlp_layernorm = Add(20)
    layer.mlp = Mlp()

    value = torch.tensor([1.0])
    output, context = layer(value)
    torch.testing.assert_close(output, value + 2 * (value + 10) + 3 * (value + 20))
    assert context is None


def test_v1_forward_rejects_bias_producing_sublayers():
    class Biased(torch.nn.Module):
        def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            return value, torch.zeros_like(value)

    class Clean(torch.nn.Module):
        def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
            return value, None

    layer = _bare_layer()
    layer.input_layernorm = torch.nn.Identity()
    layer.pre_mlp_layernorm = torch.nn.Identity()
    layer.self_attention = Biased()
    layer.mlp = Clean()
    with pytest.raises(ValueError, match="bias-free attention output"):
        layer(torch.tensor([1.0]))

    layer.self_attention = Clean()
    layer.mlp = Biased()
    with pytest.raises(ValueError, match="bias-free MLP output"):
        layer(torch.tensor([1.0]))


def test_v1_forward_rejects_cross_attention_context_mask():
    with pytest.raises(ValueError, match="cross-attention context masks"):
        _bare_layer()(torch.tensor([1.0]), context_mask=torch.tensor([1.0]))
