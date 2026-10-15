"""v3_1 — parallel decoders with a merged expert pool and global top-(g*k) routing."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.surgery import (
    MergedPoolDecoderLayer,
    compute_merged_routing,
    get_variant,
    wire_model,
)
from megatron.bridge.models.surgery.merged_moe import check_merged_pool_support


pytestmark = pytest.mark.unit

EXPERTS_PER_POOL = 8
TOPK = 2
HIDDEN = 4


def _moe_config(**fields: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "num_moe_experts": EXPERTS_PER_POOL,
        "num_layers": 2,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_pre_softmax": True,
        "moe_router_score_function": "softmax",
        "moe_router_num_groups": None,
        "moe_router_group_topk": None,
        "moe_router_topk_scaling_factor": None,
        "moe_expert_capacity_factor": None,
        "moe_z_loss_coeff": None,
        "moe_input_jitter_eps": None,
        "moe_router_enable_expert_bias": False,
        "moe_router_force_load_balancing": False,
        "moe_latent_size": None,
        "moe_shared_expert_intermediate_size": None,
        "fp32_residual_connection": False,
        "inference_fuse_tp_communication": False,
        "cuda_graph_impl": "none",
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


class _Router(torch.nn.Module):
    """Stand-in router exposing the gating/topk surface the merge relies on."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.topk = TOPK
        self.layer_number = 1

    def gating(self, value: torch.Tensor) -> torch.Tensor:
        # Shape [s, b, E]; the content is fixed so routing is checkable by hand.
        return self.logits.expand(value.shape[0], value.shape[1], EXPERTS_PER_POOL).clone()

    def get_aux_loss_coeff(self, name: str) -> float:
        return 0.0


class _MoE(torch.nn.Module):
    """Stand-in MoE layer recording the routing it was handed."""

    def __init__(self, logits: torch.Tensor, config: SimpleNamespace, scale: float) -> None:
        super().__init__()
        self.router = _Router(logits)
        self.config = config
        self.scale = scale
        self.token_dispatcher = SimpleNamespace(config=config)
        self.seen_probs: torch.Tensor | None = None
        self.seen_capacity_factor: float | None = None

    def shared_experts_compute(self, hidden_states: torch.Tensor) -> None:
        return None

    def preprocess(self, hidden_states, probs, routing_map):
        self.seen_probs = probs
        self.seen_routing_map = routing_map
        self.seen_capacity_factor = self.config.moe_expert_capacity_factor
        return hidden_states, probs

    def dispatch(self, hidden_states, probs):
        return hidden_states, probs

    def routed_experts_compute(self, hidden_states, probs):
        # Weighted sum over this pool's selected experts, mimicking combine().
        weight = probs.sum(dim=-1).reshape(hidden_states.shape[0], hidden_states.shape[1], 1)
        return self.scale * hidden_states * weight, None

    def combine(self, output):
        return output

    def postprocess(self, output, shared_expert_output):
        return output


def _layer(config: SimpleNamespace, logits_a: torch.Tensor, logits_b: torch.Tensor, index: int) -> torch.nn.Module:
    layer = MergedPoolDecoderLayer.__new__(MergedPoolDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.input_layernorm = torch.nn.Identity()
    layer.pre_mlp_layernorm = torch.nn.Identity()

    class _Attn(torch.nn.Module):
        def forward(self, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
            return value * (0.5 + index), None

    layer.self_attention = _Attn()
    layer.mlp = _MoE(logits_a if index % 2 == 0 else logits_b, config, scale=1.0 + index)
    return layer


def _model(logits_a: torch.Tensor, logits_b: torch.Tensor, config: SimpleNamespace) -> SimpleNamespace:
    layers = torch.nn.ModuleList([_layer(config, logits_a, logits_b, index) for index in range(2)])
    return SimpleNamespace(config=config, decoder=SimpleNamespace(layers=layers))


# --- registry -------------------------------------------------------------


def test_v3_1_is_registered_as_a_cross_layer_variant():
    variant = get_variant("v3_1")
    assert variant.slug == "merged-pool"
    assert variant.layer_module is MergedPoolDecoderLayer
    assert variant.needs_wiring is True
    assert variant.selection_rule == "contiguous_groups"
    assert variant.default_params["group_size"] == 2
    assert "no_pipeline_parallel" in variant.config_requirements


def test_v3_1_id_survives_a_round_trip_through_the_manifest():
    from megatron.bridge.models.surgery import legacy_architecture

    assert legacy_architecture("v3_1") == "surgery_v3_1"


# --- joint routing --------------------------------------------------------


def _routing_for(logits_a: torch.Tensor, logits_b: torch.Tensor):
    config = _moe_config()
    layers = [_MoE(logits_a, config, 1.0), _MoE(logits_b, config, 1.0)]
    value = torch.zeros(1, 1, HIDDEN)
    return layers, compute_merged_routing(layers, [value, value], variant_id="v3_1")


def test_budget_can_go_entirely_to_one_pool():
    # The whole point of the variant: a token may spend 4+0 instead of 2+2.
    high = torch.tensor([[10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0]])
    low = torch.full((1, EXPERTS_PER_POOL), -10.0)
    _, routing = _routing_for(high, low)

    assert int(routing.selected_per_pool[0]) == 4
    assert int(routing.selected_per_pool[1]) == 0
    assert routing.routing_map[0][0, :4].all()
    assert not routing.routing_map[1].any()


def test_equal_logits_reproduce_the_fixed_split():
    same = torch.arange(EXPERTS_PER_POOL, dtype=torch.float32).unsqueeze(0)
    _, routing = _routing_for(same, same.clone())
    assert int(routing.selected_per_pool[0]) == 2
    assert int(routing.selected_per_pool[1]) == 2


def test_selection_total_always_equals_pools_times_topk():
    # Active expert FLOPs must stay equal to v3's; this is the budget invariant.
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        logits_a = torch.randn(1, EXPERTS_PER_POOL, generator=generator)
        logits_b = torch.randn(1, EXPERTS_PER_POOL, generator=generator)
        _, routing = _routing_for(logits_a, logits_b)
        assert int(sum(routing.selected_per_pool)) == 2 * TOPK


def test_probabilities_are_renormalized_once_over_the_merged_selection():
    logits_a = torch.tensor([[3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    logits_b = torch.tensor([[2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, routing = _routing_for(logits_a, logits_b)

    total = routing.probs[0].sum() + routing.probs[1].sum()
    torch.testing.assert_close(total, torch.tensor(1.0))
    # Each pool's slice must NOT sum to 1 on its own; that would be the old
    # per-pool renormalization sneaking back in and restoring the fixed split.
    assert not torch.isclose(routing.probs[0].sum(), torch.tensor(1.0))
    assert not torch.isclose(routing.probs[1].sum(), torch.tensor(1.0))


def test_merged_probabilities_match_a_hand_computed_softmax():
    logits_a = torch.tensor([[2.0, 1.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
    logits_b = torch.tensor([[1.5, 0.5, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
    _, routing = _routing_for(logits_a, logits_b)

    merged = torch.cat([logits_a, logits_b], dim=-1)
    scores = torch.softmax(merged, dim=-1)
    top_scores, top_indices = torch.topk(scores, k=2 * TOPK, dim=-1)
    expected = top_scores / top_scores.sum(dim=-1, keepdim=True)

    actual = torch.cat(routing.probs, dim=-1).gather(1, top_indices)
    torch.testing.assert_close(actual, expected)


def test_unselected_experts_carry_zero_probability():
    logits_a = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    logits_b = torch.full((1, EXPERTS_PER_POOL), -20.0)
    _, routing = _routing_for(logits_a, logits_b)
    merged_probs = torch.cat(routing.probs, dim=-1)
    merged_map = torch.cat(routing.routing_map, dim=-1)
    assert (merged_probs[~merged_map] == 0).all()


# --- preconditions --------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("moe_router_pre_softmax", False),
        ("moe_router_score_function", "sigmoid"),
        ("moe_expert_capacity_factor", 1.5),
        ("moe_router_enable_expert_bias", True),
        ("moe_router_group_topk", 2),
        ("moe_shared_expert_intermediate_size", 512),
        ("moe_token_dispatcher_type", "allgather"),
    ],
)
def test_unsupported_router_settings_are_rejected(field, value):
    config = _moe_config(**{field: value})
    logits = torch.zeros(1, EXPERTS_PER_POOL)
    layers = [_MoE(logits, config, 1.0), _MoE(logits.clone(), config, 1.0)]
    with pytest.raises(ValueError, match="cannot merge expert pools"):
        check_merged_pool_support(layers, variant_id="v3_1")


def test_mismatched_pool_sizes_are_rejected():
    logits = torch.zeros(1, EXPERTS_PER_POOL)
    first = _MoE(logits, _moe_config(), 1.0)
    second = _MoE(logits.clone(), _moe_config(num_moe_experts=EXPERTS_PER_POOL * 2), 1.0)
    with pytest.raises(ValueError, match="identical pools"):
        check_merged_pool_support([first, second], variant_id="v3_1")


def test_a_dense_layer_cannot_join_a_merged_pool():
    dense = torch.nn.Identity()
    with pytest.raises(ValueError, match="routed MoE"):
        check_merged_pool_support([dense, dense], variant_id="v3_1")


# --- layer forward and wiring ---------------------------------------------


def test_forward_feeds_both_pools_the_shared_post_attention_state():
    config = _moe_config()
    logits_a = torch.tensor([[3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    logits_b = torch.tensor([[2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model = _model(logits_a, logits_b, config)
    wire_model(model, variant="v3_1", params={"group_size": 2})

    value = torch.ones(1, 1, HIDDEN)
    first, _ = model.decoder.layers[0](value)
    torch.testing.assert_close(first, value)  # pass-through

    output, _ = model.decoder.layers[1](first)

    # z = x + a_0(x) + a_1(x) = x * (1 + 0.5 + 1.5)
    shared = value * 3.0
    pools = [model.decoder.layers[index].mlp for index in range(2)]
    assert all(pool.seen_probs is not None for pool in pools)
    assert all(pool.seen_capacity_factor == 1.0 for pool in pools)
    assert config.moe_expert_capacity_factor is None
    # Both pools received a slice of one globally renormalized distribution.
    torch.testing.assert_close(sum(pool.seen_probs.sum() for pool in pools), torch.tensor(1.0))
    expected = shared
    for index, pool in enumerate(pools):
        weight = pool.seen_probs.sum(dim=-1).reshape(1, 1, 1)
        expected = expected + (1.0 + index) * shared * weight
    torch.testing.assert_close(output, expected)


def test_wiring_marks_one_joining_layer_per_group():
    config = _moe_config()
    logits = torch.zeros(1, EXPERTS_PER_POOL)
    model = _model(logits, logits.clone(), config)
    wire_model(model, variant="v3_1", params={"group_size": 2})

    assert model.decoder.layers[0].merged_pool_plan.group is None
    assert len(model.decoder.layers[1].merged_pool_plan.group) == 2


def test_wiring_rejects_a_pool_that_cannot_be_merged():
    config = _moe_config(moe_router_pre_softmax=False)
    logits = torch.zeros(1, EXPERTS_PER_POOL)
    model = _model(logits, logits.clone(), config)
    with pytest.raises(ValueError, match="cannot merge expert pools"):
        wire_model(model, variant="v3_1", params={"group_size": 2})


def test_wiring_keeps_the_state_dict_identical():
    config = _moe_config()
    logits = torch.zeros(1, EXPERTS_PER_POOL)
    model = _model(logits, logits.clone(), config)
    before = set(model.decoder.layers.state_dict())
    wire_model(model, variant="v3_1", params={"group_size": 2})
    assert set(model.decoder.layers.state_dict()) == before


def test_layer_refuses_to_run_unwired():
    config = _moe_config()
    logits = torch.zeros(1, EXPERTS_PER_POOL)
    layer = _layer(config, logits, logits.clone(), 0)
    with pytest.raises(RuntimeError, match="unwired"):
        layer(torch.ones(1, 1, HIDDEN))


def test_gradients_reach_both_pools_routers():
    config = _moe_config()
    logits_a = torch.tensor([[3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    logits_b = torch.tensor([[2.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    layers = [_MoE(logits_a, config, 1.0), _MoE(logits_b, config, 1.0)]
    value = torch.zeros(1, 1, HIDDEN)
    routing = compute_merged_routing(layers, [value, value], variant_id="v3_1")

    # Both pools' logits must stay in the graph, or one router would never learn.
    (routing.probs[0].sum() + routing.probs[1].sum() * 2).backward()
    assert logits_a.grad is not None and logits_a.grad.abs().sum() > 0
    assert logits_b.grad is not None and logits_b.grad.abs().sum() > 0
