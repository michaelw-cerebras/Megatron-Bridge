from types import SimpleNamespace

import pytest

from megatron.bridge.models.surgery import normalize_selector, resolve_selector, selector_ratio


pytestmark = pytest.mark.unit


def _config(num_layers: int = 8, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(num_layers=num_layers, **fields)


def test_shorthands_expand_and_resolve():
    assert normalize_selector("all") == {"kind": "all"}
    assert normalize_selector("none") == {"kind": "none"}
    assert resolve_selector("all", config=_config(4)) == frozenset({0, 1, 2, 3})
    assert resolve_selector("none", config=_config(4)) == frozenset()


def test_every_k_matches_the_mellum_three_to_one_pattern():
    # Mellum2 interleaves 3 sliding-window layers per dense layer; selecting the
    # three non-dense positions gives the slide deck's p=0.75 arm.
    config = _config(8)
    dense = resolve_selector({"kind": "every_k", "k": 4, "offset": 3}, config=config)
    assert dense == frozenset({3, 7})
    assert selector_ratio(dense, 8) == pytest.approx(0.25)


def test_config_list_selector_reads_a_per_layer_field():
    config = _config(4, window_attn_skip_freq=[1, 1, 1, 0])
    selector = {"kind": "config_list", "field": "window_attn_skip_freq"}
    assert resolve_selector(selector, config=config) == frozenset({0, 1, 2})
    assert selector_ratio(resolve_selector(selector, config=config), 4) == pytest.approx(0.75)

    equals = {"kind": "config_list", "field": "window_attn_skip_freq", "equals": 0}
    assert resolve_selector(equals, config=config) == frozenset({3})


def test_indices_selector_round_trips_and_deduplicates():
    config = _config(6)
    assert resolve_selector({"kind": "indices", "indices": [0, 5, 5]}, config=config) == frozenset({0, 5})


@pytest.mark.parametrize(
    "selector,message",
    [
        ("everything", "shorthand"),
        ({"kind": "nope"}, "kind"),
        ({"kind": "indices", "indices": [0, 9]}, "outside"),
        ({"kind": "indices", "indices": 3}, "list of integers"),
        ({"kind": "every_k", "k": 0}, "k"),
        ({"kind": "every_k", "k": 2, "offset": 2}, "offset"),
        ({"kind": "config_list", "field": "missing_field"}, "no per-layer field"),
    ],
)
def test_malformed_selectors_are_rejected(selector, message):
    with pytest.raises(ValueError, match=message):
        resolve_selector(selector, config=_config(6))


def test_config_list_length_must_match_num_layers():
    config = _config(4, layer_types=[1, 1])
    with pytest.raises(ValueError, match="entries but the model has 4 layers"):
        resolve_selector({"kind": "config_list", "field": "layer_types"}, config=config)


def test_bool_is_not_accepted_as_a_layer_index():
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_selector({"kind": "indices", "indices": [True]}, config=_config(4))
