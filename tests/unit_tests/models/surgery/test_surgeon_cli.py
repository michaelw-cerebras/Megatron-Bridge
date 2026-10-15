import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.models.surgery import MANIFEST_NAME, manifest_variant, read_manifest


REPO = Path(__file__).resolve().parents[4]
SURGEON_DIR = REPO / "mw_scripts/model_surgeon"
SPEC = importlib.util.spec_from_file_location("model_surgeon_cli", SURGEON_DIR / "surgeon.py")
surgeon = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = surgeon
SPEC.loader.exec_module(surgeon)
pytestmark = pytest.mark.unit


def _hf_checkpoint(root: Path, *, model_type: str = "mellum", architecture: str = "MellumForCausalLM") -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": [architecture], "model_type": model_type, "hidden_size": 8})
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 23},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    (root / "tokenizer.json").write_text("{}")
    return root


def test_hardlink_surgery_preserves_weights_and_records_the_variant(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    manifest = surgeon.clone_hf_checkpoint(source, target, variant_id="v1", copy_mode="hardlink")

    assert manifest["schema_version"] == 2
    assert manifest["variant"]["id"] == "v1"
    assert manifest["variant"]["slug"] == "parallel"
    assert manifest["variant"]["params"]["layers"] == {"kind": "all"}
    assert manifest["weights_identical_to_source"] is True
    assert manifest["model_type"] == "mellum"
    assert manifest["parameter_tensors"] == 2

    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        assert os.path.samefile(source / name, target / name)
    assert not os.path.samefile(source / "config.json", target / "config.json")
    assert not os.path.samefile(source / "tokenizer.json", target / "tokenizer.json")

    target_config = json.loads((target / "config.json").read_text())
    assert target_config["megatron_bridge_architecture"] == "parallel_attention_moe"
    assert target_config["megatron_bridge_surgery_variant"] == "v1"
    assert json.loads((target / MANIFEST_NAME).read_text())["artifact_format"] == "huggingface_safetensors"


def test_surgery_is_not_mellum_specific(tmp_path):
    source = _hf_checkpoint(tmp_path / "source", model_type="llama", architecture="LlamaForCausalLM")
    manifest = surgeon.clone_hf_checkpoint(source, tmp_path / "target", variant_id="v1")
    assert manifest["model_type"] == "llama"


def test_variant_parameters_are_recorded(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    manifest = surgeon.clone_hf_checkpoint(
        source,
        tmp_path / "target",
        variant_id="v1",
        params={"layers": {"kind": "every_k", "k": 4, "offset": 0}},
    )
    assert manifest["variant"]["params"]["layers"] == {"kind": "every_k", "k": 4, "offset": 0}


def test_copy_surgery_hashes_independent_files(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    manifest = surgeon.clone_hf_checkpoint(source, target, variant_id="v1", copy_mode="copy")
    source_file = source / "model-00001-of-00002.safetensors"
    target_file = target / source_file.name
    assert not os.path.samefile(source_file, target_file)
    assert source_file.read_bytes() == target_file.read_bytes()
    copied = [record for record in manifest["files"] if record["path"] != "config.json"]
    assert all(str(record["verification"]).startswith("sha256:") for record in copied)

    result = surgeon.verify_checkpoint(source, target)
    assert result.variant_id == "v1"
    assert result.parameter_tensors == 2
    assert result.weight_shards == 2
    assert result.same_inode_shards == 0
    assert result.sha256_shards == 2


def test_weight_verification_rejects_changed_shard(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v1", copy_mode="copy")
    (target / "model-00001-of-00002.safetensors").write_bytes(b"changed-weight-shard")

    with pytest.raises(ValueError, match="Weight shard differs"):
        surgeon.verify_checkpoint(source, target)


def test_dry_run_writes_nothing_and_missing_shard_is_rejected(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    manifest = surgeon.clone_hf_checkpoint(source, target, variant_id="v1", dry_run=True)
    assert manifest["file_count"] == 5
    assert not target.exists()
    (source / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing shards"):
        surgeon.clone_hf_checkpoint(source, target, variant_id="v1")


def test_surgery_on_a_derived_checkpoint_is_rejected(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    first = tmp_path / "first"
    surgeon.clone_hf_checkpoint(source, first, variant_id="v1")
    with pytest.raises(ValueError, match="already carries a surgery marker"):
        surgeon.clone_hf_checkpoint(first, tmp_path / "second", variant_id="v1")


def test_unknown_variant_is_rejected_before_any_write(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="Unknown surgery variant"):
        surgeon.clone_hf_checkpoint(source, target, variant_id="v42")
    assert not target.exists()


def test_schema_v1_manifests_still_resolve_to_a_variant(tmp_path):
    # Checkpoints produced before the registry existed must keep training and
    # resuming; their manifests only carried an architecture string.
    derived = _hf_checkpoint(tmp_path / "legacy")
    config = json.loads((derived / "config.json").read_text())
    config["megatron_bridge_architecture"] = "parallel_attention_moe"
    (derived / "config.json").write_text(json.dumps(config))
    (derived / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_format": "huggingface_safetensors",
                "target_architecture": "parallel_attention_moe",
                "parameter_tensors": 2,
            }
        )
    )
    variant_id, params = manifest_variant(read_manifest(derived))
    assert variant_id == "v1"
    assert params["layers"] == {"kind": "all"}


def test_unreadable_manifest_schema_is_rejected(tmp_path):
    derived = _hf_checkpoint(tmp_path / "derived")
    (derived / MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 99, "artifact_format": "huggingface_safetensors"})
    )
    with pytest.raises(ValueError, match="Unsupported surgery manifest schema_version"):
        read_manifest(derived)


def test_model_id_resolves_through_huggingface_hub(tmp_path, monkeypatch):
    source = _hf_checkpoint(tmp_path / "downloaded")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(source)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    resolved = surgeon.resolve_hf_source(
        "JetBrains/Mellum2-12B-A2.5B-Thinking",
        target_dir=tmp_path / "models/parallel",
        cache_dir=tmp_path / "cache",
        revision="commit-sha",
    )
    assert resolved == source.resolve()
    assert calls[0]["repo_id"] == "JetBrains/Mellum2-12B-A2.5B-Thinking"
    assert calls[0]["revision"] == "commit-sha"
    assert calls[0]["cache_dir"] == str(tmp_path / "cache")

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(surgeon, "PIPELINE_HF_CACHE", tmp_path / "missing-pipeline-cache")
    surgeon.resolve_hf_source(
        "JetBrains/Mellum2-12B-A2.5B-Thinking",
        target_dir=tmp_path / "models/parallel",
        cache_dir=None,
        revision=None,
    )
    assert calls[1]["cache_dir"] == str(tmp_path / "models/.cache/huggingface/hub")


def _load_mellum_train():
    scripts = REPO / "mw_scripts/mellum2_3xh200"
    train_spec = importlib.util.spec_from_file_location("mellum_surgery_train", scripts / "train.py")
    train = importlib.util.module_from_spec(train_spec)
    sys.path.insert(0, str(scripts))
    try:
        train_spec.loader.exec_module(train)
    finally:
        sys.path.remove(str(scripts))
    return train


def _model_config(num_layers: int = 4) -> SimpleNamespace:
    """Minimal stand-in for a provider, enough for manifest and requirement checks."""
    return SimpleNamespace(
        transformer_layer_spec=None,
        perform_initialization=False,
        num_layers=num_layers,
        pipeline_model_parallel_size=1,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        mlp_chunks_for_prefill=1,
        mlp_chunks_for_training=1,
        cuda_graph_impl="none",
        inference_fuse_tp_communication=False,
        recompute_granularity=None,
        cpu_offloading=False,
        attention_backend=SimpleNamespace(name="flash"),
    )


def _mellum_provider():
    """Build a real tiny Mellum provider so the family spec actually composes."""
    from transformers import AutoConfig

    from megatron.bridge import AutoBridge

    hf_config = AutoConfig.for_model(
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
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    provider.perform_initialization = False
    return provider


def test_training_builds_the_variant_spec_from_the_manifest(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v1")

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_mellum_provider())
    original_permute_fusion = cfg.model.moe_permute_fusion
    settings = {
        "model": {"architecture": "v1", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    train._apply_architecture(cfg, settings)

    from megatron.bridge.models.mellum.modeling_mellum.layer_specs import MellumSelfAttention
    from megatron.bridge.models.surgery import ParallelAttentionMLPLayer

    block = cfg.model.transformer_layer_spec(cfg.model)
    assert [spec.module for spec in block.layer_specs] == [ParallelAttentionMLPLayer] * 4
    # Surgery must not disturb what the family put inside the layer.
    assert block.layer_specs[0].submodules.self_attention.module is MellumSelfAttention
    assert block.layer_norm is not None
    # Kernel choices belong to the training config, not to graph surgery.
    assert cfg.model.moe_permute_fusion is original_permute_fusion


def test_mellum_sliding_window_layers_can_be_parallelized_alone(tmp_path):
    # The slide deck's Mellum2 arm: 3 sliding-window blocks parallel, dense block
    # left sequential. Same variant, different ratio.
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(
        source,
        target,
        variant_id="v1",
        params={"layers": {"kind": "config_list", "field": "window_attn_skip_freq"}},
    )

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_mellum_provider())
    settings = {
        "model": {"architecture": "v1", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    train._apply_architecture(cfg, settings)

    from megatron.bridge.models.surgery import ParallelAttentionMLPLayer, describe_surgery

    block = cfg.model.transformer_layer_spec(cfg.model)
    parallel = [spec.module is ParallelAttentionMLPLayer for spec in block.layer_specs]
    assert parallel == [True, True, True, False]
    described = describe_surgery(
        cfg.model,
        variant="v1",
        params={"layers": {"kind": "config_list", "field": "window_attn_skip_freq"}},
    )
    assert described["parallelization_ratio"] == pytest.approx(0.75)


def test_training_wires_a_cross_layer_variant_on_a_real_provider(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v2")

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_mellum_provider())
    settings = {
        "model": {"architecture": "v2", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    train._apply_architecture(cfg, settings)

    from megatron.bridge.models.surgery import ShiftedAttentionMLPLayer

    block = cfg.model.transformer_layer_spec(cfg.model)
    assert [spec.module for spec in block.layer_specs] == [ShiftedAttentionMLPLayer] * 4
    # v2 couples neighbours, so the provider must also carry the wiring hook.
    assert cfg.model.pre_wrap_hook is not None


def test_training_registers_merged_pool_logging_metrics(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v3_1")

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_mellum_provider())
    settings = {
        "model": {"architecture": "v3_1", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    train._apply_architecture(cfg, settings)

    assert cfg.model.moe_additional_logging_metrics == (
        "merged_pool_frac_0",
        "merged_pool_frac_1",
    )


def test_training_accepts_the_legacy_architecture_name(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v1")

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_mellum_provider())
    settings = {
        "model": {"architecture": "parallel_attention_moe", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    train._apply_architecture(cfg, settings)
    assert cfg.model.transformer_layer_spec is not None


def test_training_rejects_a_config_that_disagrees_with_the_checkpoint(tmp_path):
    # The weights are byte-identical across variants, so a mismatched pairing
    # would otherwise train the wrong graph without any error.
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v1")

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_model_config())
    settings = {
        "model": {"architecture": "v2", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    with pytest.raises(ValueError, match="was produced for variant v1"):
        train._apply_architecture(cfg, settings)


def test_training_leaves_the_original_architecture_untouched(tmp_path):
    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_model_config())
    for name in ("original", "v0"):
        settings = {"model": {"architecture": name, "hf_path": "org/model"}, "run": {"resume_checkpoint": None}}
        train._apply_architecture(cfg, settings)
        assert cfg.model.transformer_layer_spec is None


def test_initial_import_must_not_randomly_initialize(tmp_path):
    source = _hf_checkpoint(tmp_path / "source")
    target = tmp_path / "target"
    surgeon.clone_hf_checkpoint(source, target, variant_id="v1")

    train = _load_mellum_train()
    cfg = SimpleNamespace(model=_model_config())
    cfg.model.perform_initialization = True
    settings = {
        "model": {"architecture": "v1", "hf_path": str(target)},
        "run": {"resume_checkpoint": None},
    }
    with pytest.raises(ValueError, match="must disable Megatron random initialization"):
        train._apply_architecture(cfg, settings)
