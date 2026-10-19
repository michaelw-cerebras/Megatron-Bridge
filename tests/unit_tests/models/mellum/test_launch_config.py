import copy
import hashlib
import importlib.util
import json
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "mellum_configuration", REPO / "mw_scripts/mellum2_3xh200/configuration.py"
)
configuration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configuration)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("world_size", "accumulation"), [(16, 12), (24, 8)])
def test_expert_mesh_does_not_divide_batch_again(world_size: int, accumulation: int) -> None:
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    assert configuration.validate_parallelism(config, world_size=world_size, experts=64) == accumulation


def test_batch_128_requires_16_gpu_mesh() -> None:
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    config["train"]["global_batch_size"] = 128
    assert configuration.validate_parallelism(config, world_size=16, experts=64) == 8
    with pytest.raises(ValueError, match=r"MBS\*DP=24"):
        configuration.validate_parallelism(config, world_size=24, experts=64)


def test_parallel_architecture_name_is_accepted(tmp_path: Path) -> None:
    source = (REPO / "mw_scripts/mellum2_3xh200/config.yaml").read_text()
    target = tmp_path / "parallel.yaml"
    target.write_text(source.replace("architecture: original", "architecture: parallel_attention_moe"))
    assert configuration.read_config(target)["model"]["architecture"] == "parallel_attention_moe"


@pytest.mark.parametrize(
    ("nodes_override", "expected_nodes"),
    [(None, ["172.31.210.225"]), ("node-a node-b node-c", ["node-a", "node-b", "node-c"])],
)
def test_launcher_derives_topology_without_contacting_real_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nodes_override: str | None, expected_nodes: list[str]
) -> None:
    # Intercept SSH entirely: this test never accesses a node or a GPU.
    ssh_stub = tmp_path / "ssh"
    ssh_log = tmp_path / "ssh.jsonl"
    ssh_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shlex, sys\n"
        "with open(os.environ['MELLUM_SSH_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "command = shlex.split(sys.argv[-1])\n"
        "if command == ['id', '-u']: sys.stdout.write('2087\\n')\n"
        "if command == ['id', '-g']: sys.stdout.write('9004\\n')\n"
    )
    ssh_stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("MELLUM_SSH_TEST_LOG", str(ssh_log))
    monkeypatch.setenv("MELLUM_ENV_FILE", "")  # Never read the user's real credential file in tests.
    monkeypatch.delenv("MELLUM_MASTER_ADDR", raising=False)
    monkeypatch.delenv("MELLUM_MASTER_PORT", raising=False)
    if nodes_override is None:
        monkeypatch.delenv("MELLUM_NODES", raising=False)
    else:
        monkeypatch.setenv("MELLUM_NODES", nodes_override)
    subprocess.run(["bash", str(REPO / "mw_scripts/mellum2_3xh200/launch.sh"), "--dry-run"], check=True)
    calls = [json.loads(line) for line in ssh_log.read_text().splitlines()]
    assert {call[-2] for call in calls} == set(expected_nodes)
    assert all(call[0] == "-F" and call[1].endswith("/.ssh/config") for call in calls)
    launches = [(call[-2], shlex.split(call[-1])) for call in calls if shlex.split(call[-1])[:2] == ["docker", "exec"]]
    assert len(launches) == len(expected_nodes)
    for rank, (node, command) in enumerate(launches):
        assert node == expected_nodes[rank]
        assert command[-6] == "dry-run"
        assert command[-4:] == [str(rank), str(len(expected_nodes)), expected_nodes[0], "29541"]


def test_all_variant_configs_are_complete_and_valid() -> None:
    config_dir = REPO / "mw_scripts/mellum2_3xh200/config"
    shared_variants = ("v0", "v1", "v2", "v3", "v3_1", "v1_1")
    expected = {f"{family}-{variant}.yaml" for family in ("base", "thinking") for variant in shared_variants}
    expected.update({"base-v3_2.yaml", "base-v3_3.yaml"})
    assert {path.name for path in config_dir.glob("*.yaml")} == expected
    for path in config_dir.glob("*.yaml"):
        config = configuration.read_config(path)
        family, variant = path.stem.split("-")
        assert config["model"]["architecture"] == variant
        assert family in config["model"]["hf_path"].lower()
        assert config["train"]["train_tokens"] == 10_000_000_000
        assert config["run"]["save_interval"] == 100
        assert config["run"]["keep_last_checkpoints"] == 3
        assert config["wandb"]["run_name"].endswith("_10B")
        assert config["run"]["directory"].endswith("_10B")
        assert sum(config["dataset"]["subsets"].values()) == pytest.approx(100.0)
        # v1_1 trains only its correction branches, so it fits a larger micro batch.
        # GBS stays 192 across every arm, so the optimizer batch size is identical
        # and only the accumulation count changes.
        expected_micro_batch = 2 if variant == "v1_1" else 1
        expected_accumulation = 192 // (expected_micro_batch * 8)
        assert config["train"]["micro_batch_size"] == expected_micro_batch
        assert configuration.validate_parallelism(config, world_size=8, experts=64) == expected_accumulation
        assert config["optimizer"]["lr"] == 0.00003


def test_launcher_help_lists_only_canonical_variant_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MELLUM_ENV_FILE", "")
    result = subprocess.run(
        ["bash", str(REPO / "mw_scripts/mellum2_3xh200/launch.sh"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "base-v{0,1,1.1,2,3,3.1,3.2,3.3}" in result.stdout
    assert "thinking-v{0,1,1.1,2,3,3.1}" in result.stdout
    assert "5_1" not in result.stdout
    assert "5_2" not in result.stdout


def test_launcher_accepts_dotted_variant_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MELLUM_ENV_FILE", "")
    result = subprocess.run(
        [
            "bash",
            str(REPO / "mw_scripts/mellum2_3xh200/launch.sh"),
            "--variant",
            "base-v3.3",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "config/base-v3_3.yaml" in result.stdout


def test_recompute_is_runtime_only_and_validated(tmp_path: Path) -> None:
    source = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config/base-v3_2.yaml")
    tuned = copy.deepcopy(source)
    tuned["performance"]["recompute_method"] = "block"
    tuned["performance"]["recompute_num_layers"] = 2
    assert configuration.config_digest(tuned) == configuration.config_digest(source)

    target = REPO / "mw_scripts/mellum2_3xh200/config/base-v3_2.yaml"
    text = target.read_text().replace(
        "  moe_permute_fusion: true\n", "  moe_permute_fusion: true\n  recompute_method: block\n"
    )
    with pytest.raises(ValueError, match="must be set together"):
        path = tmp_path / "invalid-mellum-recompute.yaml"
        path.write_text(text)
        configuration.read_config(path)


@pytest.mark.parametrize(
    "variant",
    ["base-v2", "base-v3_1", "thinking-v3_1", "base-v1_1", "thinking-v1_1", "base-v3_2", "base-v3_3"],
)
def test_launcher_selects_variant_config_without_contacting_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    ssh_log = tmp_path / "ssh.jsonl"
    ssh_stub = tmp_path / "ssh"
    ssh_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shlex, sys\n"
        "with open(os.environ['MELLUM_SSH_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "command = shlex.split(sys.argv[-1])\n"
        "if command == ['id', '-u']: sys.stdout.write('2087\\n')\n"
        "if command == ['id', '-g']: sys.stdout.write('9004\\n')\n"
    )
    ssh_stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("MELLUM_SSH_TEST_LOG", str(ssh_log))
    monkeypatch.setenv("MELLUM_ENV_FILE", "")
    monkeypatch.delenv("MELLUM_CONFIG", raising=False)
    subprocess.run(
        [
            "bash",
            str(REPO / "mw_scripts/mellum2_3xh200/launch.sh"),
            "--variant",
            variant,
            "--dry-run",
        ],
        check=True,
    )
    calls = [shlex.split(json.loads(line)[-1]) for line in ssh_log.read_text().splitlines()]
    launch = next(call for call in calls if call[:2] == ["docker", "exec"])
    assert launch[-5].endswith(f"/config/{variant}.yaml")


def test_folded_mesh_tp2_ep8_is_valid():
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    config["parallelism"]["tensor_model_parallel_size"] = 2
    assert configuration.validate_parallelism(config, world_size=24, experts=64) == 16


@pytest.fixture
def container_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Intercept container preparation without contacting Docker or real nodes."""
    ssh_log = tmp_path / "ssh.jsonl"
    ssh_stub = tmp_path / "ssh"
    ssh_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shlex, sys\n"
        "with open(os.environ['MELLUM_SSH_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "command = shlex.split(sys.argv[-1])\n"
        "state = os.environ.get('MELLUM_TEST_CONTAINER_STATE', 'absent')\n"
        "if command[:2] == ['docker', 'inspect']:\n"
        "    if '-f' not in command: sys.exit(1 if state == 'absent' else 0)\n"
        "    template = command[command.index('-f') + 1]\n"
        "    if template == '{{.State.Running}}':\n"
        "        sys.stdout.write('false\\n' if state == 'stopped' else 'true\\n')\n"
        "    elif 'if not .RW' in template:\n"
        "        sys.stdout.write('/test/data /workspace/ml-olympus\\n')\n"
        "    elif state != 'wrong-mounts':\n"
        "        sys.stdout.write('/test/output /workspace/ml-olympus/codesign_pretrain/runs\\n')\n"
    )
    ssh_stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("MELLUM_SSH_TEST_LOG", str(ssh_log))
    monkeypatch.setenv("MELLUM_ENV_FILE", "")
    monkeypatch.setenv("MELLUM_NODES", "node-a node-b")
    monkeypatch.delenv("MELLUM_CONTAINER", raising=False)
    monkeypatch.delenv("MELLUM_MEMORY_GIB", raising=False)
    monkeypatch.setenv("MELLUM_HOST_DATA", "/test/data")
    monkeypatch.setenv("MELLUM_HOST_RUNS", "/test/output")
    monkeypatch.setenv("MELLUM_REMOTE_RUNS", "/workspace/ml-olympus/codesign_pretrain/runs")

    def invoke(state: str):
        monkeypatch.setenv("MELLUM_TEST_CONTAINER_STATE", state)
        result = subprocess.run(
            ["bash", str(REPO / "mw_scripts/mellum2_3xh200/launch.sh"), "--prepare-containers"],
            capture_output=True,
            text=True,
            check=False,
        )
        calls = (
            [shlex.split(json.loads(line)[-1]) for line in ssh_log.read_text().splitlines()]
            if ssh_log.exists()
            else []
        )
        return result, calls

    return invoke


def test_new_container_mounts_output_writable_and_data_readonly(container_launcher) -> None:
    result, calls = container_launcher("absent")
    assert result.returncode == 0, result.stderr
    runs = [call for call in calls if call[:2] == ["docker", "run"]]
    assert len(runs) == 2
    for command in runs:
        assert command[command.index("--name") + 1] == "mellum2_3xh200"
        assert command[command.index("--memory") + 1] == "1024g"
        assert command[command.index("--memory-swap") + 1] == "1024g"
        assert "nofile=262144:262144" in command
        assert "/test/data:/workspace/ml-olympus:ro" in command
        assert command[command.index("--mount") + 1] == (
            "type=bind,src=/test/output,dst=/workspace/ml-olympus/codesign_pretrain/runs"
        )
        assert calls[calls.index(command) - 1] == ["mkdir", "-p", "--", "/test/output"]
    assert not any(call[:2] in (["docker", "rm"], ["docker", "stop"]) for call in calls)


def test_container_memory_budget_override(container_launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MELLUM_MEMORY_GIB", "768")
    result, calls = container_launcher("absent")
    assert result.returncode == 0, result.stderr
    for command in (call for call in calls if call[:2] == ["docker", "run"]):
        assert command[command.index("--memory") + 1] == "768g"
        assert command[command.index("--memory-swap") + 1] == "768g"


@pytest.mark.parametrize("value", ["", "0", "-1", "max", "1.5", "1024g"])
def test_invalid_memory_budget_rejected_before_ssh(
    container_launcher, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MELLUM_MEMORY_GIB", value)
    result, calls = container_launcher("absent")
    assert result.returncode == 2
    assert "MELLUM_MEMORY_GIB" in result.stderr
    assert not calls


@pytest.mark.parametrize("state", ["running", "wrong-mounts", "stopped"])
def test_existing_container_is_never_modified(container_launcher, state: str) -> None:
    result, calls = container_launcher(state)
    assert result.returncode == (0 if state == "running" else 1)
    assert all(call[:2] == ["docker", "inspect"] for call in calls)
    if state == "wrong-mounts":
        assert "must mount data read-only" in result.stderr
    elif state == "stopped":
        assert "start it explicitly" in result.stderr


@pytest.mark.parametrize("key", ["MELLUM_HOST_RUNS", "MELLUM_REMOTE_RUNS"])
@pytest.mark.parametrize("path", ["/", "relative/path", "/test/with,comma"])
def test_invalid_output_mount_path_rejected_before_ssh(
    container_launcher, monkeypatch: pytest.MonkeyPatch, key: str, path: str
) -> None:
    monkeypatch.setenv(key, path)
    # No SSH call is expected, so the fixture's log file will not be created.
    result = subprocess.run(
        ["bash", str(REPO / "mw_scripts/mellum2_3xh200/launch.sh"), "--prepare-containers"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Output mount paths" in result.stderr
    assert not Path(os.environ["MELLUM_SSH_TEST_LOG"]).exists()


def test_yaml_duplicate_and_typo_rejected(tmp_path):
    target = tmp_path / "config.yaml"
    source = (REPO / "mw_scripts/mellum2_3xh200/config.yaml").read_text()
    target.write_text(source.replace("  seed: 1234", "  seed: 1234\n  seed: 1"))
    with pytest.raises(ValueError, match="Duplicate"):
        configuration.read_config(target)
    target.write_text(source.replace("  lr: 0.00003", "  learning_rate: 0.00003"))
    with pytest.raises(ValueError, match="optimizer requires"):
        configuration.read_config(target)


def test_index_counts_and_truncated_binary(tmp_path):
    prefix = tmp_path / "shard"
    Path(f"{prefix}.idx").write_bytes(
        b"MMIDIDX\x00\x00"
        + struct.pack("<QBQQ", 1, 4, 2, 3)
        + struct.pack("<ii", 3, 4)
        + struct.pack("<qq", 0, 12)
        + struct.pack("<qqq", 0, 1, 2)
    )
    Path(f"{prefix}.bin").write_bytes(b"\x00" * 28)
    assert configuration.index_token_count(prefix) == 7
    Path(f"{prefix}.bin").write_bytes(b"\x00" * 24)
    with pytest.raises(ValueError, match="truncated bin"):
        configuration.index_token_count(prefix)


def _write_index(prefix, tokens):
    import numpy as np

    prefix.parent.mkdir(parents=True, exist_ok=True)
    values = np.array(tokens, dtype="<i4")
    Path(f"{prefix}.bin").write_bytes(values.tobytes())
    Path(f"{prefix}.idx").write_bytes(
        b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, 1, 2) + struct.pack("<iqqq", len(tokens), 0, 0, 1)
    )


def test_reserved_token_scan_requires_absence(tmp_path):
    prefix = tmp_path / "shard"
    _write_index(prefix, [0, 26, 33, 100])
    configuration.check_reserved_ids_absent(prefix, {27, 28, 29, 30, 31, 32})
    _write_index(prefix, [0, 29, 100])
    with pytest.raises(ValueError, match="renamed reserved"):
        configuration.check_reserved_ids_absent(prefix, {27, 28, 29, 30, 31, 32})


def test_mixture_weights_and_frozen_shards(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoConfig, AutoTokenizer

    vocab = {"eod": 0, "word": 1}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="eod"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": {
                    "eos_id": 0,
                    "json_path": "tokenizer.json",
                    "json_sha256": hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
                }
            }
        )
    )
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda _: SimpleNamespace(get_vocab=lambda: vocab, all_special_ids=[0])
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda _: SimpleNamespace(vocab_size=2))
    _write_index(tmp_path / "a/first", [1, 0])
    _write_index(tmp_path / "a/second", [1, 1, 1, 1, 1, 0])
    _write_index(tmp_path / "b/third", [1, 0])
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    config["dataset"]["subsets"] = {str(tmp_path / "a"): 3, str(tmp_path / "b"): 1}
    mixture = configuration.prepare_mixture(config, hf_snapshot="unused")
    assert [x["weight"] for x in mixture["shards"]] == pytest.approx([0.1875, 0.5625, 0.25])
    configuration.validate_snapshot_files(mixture)
    _write_index(tmp_path / "a/first", [1, 1, 0])
    with pytest.raises(ValueError, match="Frozen dataset file changed"):
        configuration.validate_snapshot_files(mixture)


def test_resume_and_exit_do_not_change_mixture_identity():
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    digest = configuration.config_digest(config)
    config["run"].update(
        resume_checkpoint="/tmp/checkpoints",
        override_checkpoint_scheduler=True,
        reset_optimizer_on_resume=True,
        exit_interval=3,
    )
    assert configuration.config_digest(config) == digest
    config["optimizer"]["lr"] *= 2
    assert configuration.config_digest(config) != digest


def test_scheduler_recovery_rejects_unrelated_config_changes():
    old_config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    new_config = copy.deepcopy(old_config)
    new_config["optimizer"]["lr"] /= 10
    new_config["optimizer"]["min_lr"] /= 10
    new_config["scheduler"]["lr_warmup_iters"] += 1
    new_config["run"].update(resume_checkpoint="/tmp/checkpoints", override_checkpoint_scheduler=True)
    assert configuration.is_scheduler_recovery(old_config, new_config)
    new_config["train"]["seed"] += 1
    assert not configuration.is_scheduler_recovery(old_config, new_config)


@pytest.mark.parametrize("policy", ["strict", "skip", None])
def test_reserved_check_yaml_policy(tmp_path: Path, policy: str | None) -> None:
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    if policy is None:
        config["dataset"].pop("reserved_token_check", None)
    else:
        config["dataset"]["reserved_token_check"] = policy
    target = tmp_path / "config.yaml"
    target.write_text(configuration.yaml.safe_dump(config))
    loaded = configuration.read_config(target)
    assert loaded["dataset"].get("reserved_token_check", "strict") == (policy or "strict")


@pytest.mark.parametrize("policy", [True, None, "sample", 1, [], {}])
def test_invalid_reserved_check_policy_rejected(tmp_path: Path, policy: object) -> None:
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    config["dataset"]["reserved_token_check"] = policy
    target = tmp_path / "config.yaml"
    target.write_text(configuration.yaml.safe_dump(config))
    with pytest.raises(ValueError, match="reserved_token_check"):
        configuration.read_config(target)


def test_reserved_check_policy_is_immutable_and_preserves_old_strict_digest() -> None:
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    assert config["dataset"]["reserved_token_check"] == "skip"
    skip_digest = configuration.config_digest(config)
    config["dataset"].pop("reserved_token_check")
    old_strict_digest = configuration.config_digest(config)
    assert skip_digest != old_strict_digest
    config["dataset"]["reserved_token_check"] = "strict"
    assert configuration.config_digest(config) == old_strict_digest


@pytest.fixture
def reserved_mixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[configuration.TrainingSettings, Path]:
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoConfig, AutoTokenizer

    vocab = {"eod": 0, "word": 1, **{f"filler_{i}": i for i in range(2, 27)}, "<|extra_token_1|>": 27}
    target_vocab = {token: index for token, index in vocab.items() if index != 27} | {"<|im_start|>": 27}
    token_file = tmp_path / "tokenizer.json"
    Tokenizer(WordLevel(vocab, unk_token="eod")).save(str(token_file))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": {
                    "eos_id": 0,
                    "json_path": "tokenizer.json",
                    "json_sha256": hashlib.sha256(token_file.read_bytes()).hexdigest(),
                }
            }
        )
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: SimpleNamespace(get_vocab=lambda: target_vocab))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda _: SimpleNamespace(vocab_size=28))
    prefix = tmp_path / "data/shard"
    _write_index(prefix, [1, 27, 0])
    config = configuration.read_config(REPO / "mw_scripts/mellum2_3xh200/config.yaml")
    config["dataset"]["subsets"] = {str(prefix.parent): 100}
    return config, prefix


def test_skip_never_scans_bins_and_records_unverified_ids(
    reserved_mixture: tuple[configuration.TrainingSettings, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config, prefix = reserved_mixture

    def unexpected_scan(*args: object) -> None:
        pytest.fail("Skip mode must not scan bin contents")

    monkeypatch.setattr(configuration, "check_reserved_ids_absent", unexpected_scan)
    # A bin without an index must remain outside the frozen mixture.
    (prefix.parent / "incomplete.bin").write_bytes(b"unfinished")
    mixture = configuration.prepare_mixture(config, hf_snapshot="unused")
    assert mixture["reserved_token_check"] == "skip"
    assert mixture["reserved_ids_checked_absent"] == {}
    assert mixture["reserved_ids_not_checked"] == {str(prefix.parent.parent / "manifest.json"): [27]}
    assert [shard["prefix"] for shard in mixture["shards"]] == [str(prefix)]
    assert mixture["shards"][0]["tokens"] == 3
    assert mixture["shards"][0]["weight"] == 1.0
    assert "UNVERIFIED" in caplog.text
    configuration.validate_snapshot_files(mixture)


@pytest.mark.parametrize("explicit_policy", [False, True])
def test_strict_and_legacy_policy_reject_renamed_ids(
    reserved_mixture: tuple[configuration.TrainingSettings, Path], explicit_policy: bool
) -> None:
    config, _ = reserved_mixture
    if explicit_policy:
        config["dataset"]["reserved_token_check"] = "strict"
    else:
        config["dataset"].pop("reserved_token_check")
    with pytest.raises(ValueError, match="renamed reserved token IDs"):
        configuration.prepare_mixture(config, hf_snapshot="unused")


def test_strict_records_successful_reserved_id_scan(
    reserved_mixture: tuple[configuration.TrainingSettings, Path],
) -> None:
    config, prefix = reserved_mixture
    config["dataset"]["reserved_token_check"] = "strict"
    _write_index(prefix, [1, 1, 0])
    mixture = configuration.prepare_mixture(config, hf_snapshot="unused")
    assert mixture["reserved_token_check"] == "strict"
    assert mixture["reserved_ids_checked_absent"] == {str(prefix.parent.parent / "manifest.json"): [27]}
    assert mixture["reserved_ids_not_checked"] == {}


def test_skip_still_rejects_truncated_bins(reserved_mixture: tuple[configuration.TrainingSettings, Path]) -> None:
    config, prefix = reserved_mixture
    Path(f"{prefix}.bin").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="truncated bin"):
        configuration.prepare_mixture(config, hf_snapshot="unused")


def test_skip_still_rejects_tokenizer_checksum_changes(
    reserved_mixture: tuple[configuration.TrainingSettings, Path],
) -> None:
    config, prefix = reserved_mixture
    (prefix.parent.parent / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="Tokenizer checksum mismatch"):
        configuration.prepare_mixture(config, hf_snapshot="unused")


def test_skip_still_rejects_ordinary_token_mapping_changes(
    reserved_mixture: tuple[configuration.TrainingSettings, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    config, _ = reserved_mixture
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda _: SimpleNamespace(get_vocab=lambda: {"eod": 0, "word": 2})
    )
    with pytest.raises(ValueError, match="Token ID mapping is incompatible"):
        configuration.prepare_mixture(config, hf_snapshot="unused")


def test_skip_still_rejects_eod_mismatch(reserved_mixture: tuple[configuration.TrainingSettings, Path]) -> None:
    config, _ = reserved_mixture
    config["dataset"]["eod_id"] = 1
    with pytest.raises(ValueError, match="Dataset EOD differs"):
        configuration.prepare_mixture(config, hf_snapshot="unused")
