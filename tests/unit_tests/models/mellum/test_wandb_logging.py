import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml


REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "mw_scripts/mellum2_3xh200"
pytestmark = pytest.mark.unit


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in (
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
        "WANDB_HOST",
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_MODE",
        "NETRC",
    ):
        monkeypatch.delenv(name, raising=False)
    return importlib.import_module("configuration"), importlib.import_module("wandb_logging")


def test_maps_real_bridge_logger_without_credentials(modules: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    from megatron.bridge.training.config import LoggerConfig

    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
    settings["run"]["resume_checkpoint"] = None
    logger = LoggerConfig()
    tracking.apply_wandb_config(logger, settings)
    assert logger.wandb_project == settings["wandb"]["project"]
    assert logger.wandb_entity == settings["wandb"]["entity"]
    assert logger.wandb_exp_name == settings["wandb"]["run_name"]
    assert logger.wandb_save_dir == str(tmp_path / "wandb")
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    settings["wandb"]["enabled"] = False
    tracking.apply_wandb_config(logger, settings)
    assert logger.wandb_project is None


@pytest.mark.parametrize("field", ["entity", "project", "run_name", "base_url", "enabled"])
def test_tracking_does_not_change_frozen_training_digest(modules: tuple[ModuleType, ModuleType], field: str) -> None:
    configuration, _ = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    digest = configuration.config_digest(settings)
    settings["wandb"][field] = False if field == "enabled" else "different"
    assert configuration.config_digest(settings) == digest
    settings.pop("wandb")
    assert configuration.config_digest(settings) == digest


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": "synthetic-do-not-store"},
        {"enabled": "true"},
        {"project": ""},
        {"base_url": "https://user:synthetic@example.com"},
        {"base_url": "http://example.com"},
    ],
)
def test_invalid_or_secret_wandb_fields_rejected(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path, overrides: dict[str, object]
) -> None:
    configuration, _ = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["wandb"].update(overrides)
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="wandb"):
        configuration.read_config(target)


def test_old_yaml_without_tracking_is_supported(modules: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings.pop("wandb")
    target = tmp_path / "old.yaml"
    target.write_text(yaml.safe_dump(settings))
    settings = configuration.read_config(target)
    tracking.prepare_wandb_environment(settings, rank=15, world_size=16)
    assert "WANDB_RUN_ID" not in os.environ


def test_only_last_rank_persists_identity_and_resume_reuses_it(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-key-not-a-real-credential")
    tracking.prepare_wandb_environment(settings, rank=0, world_size=16)
    assert not (tmp_path / "metadata").exists()
    assert "WANDB_RUN_ID" not in os.environ
    tracking.prepare_wandb_environment(settings, rank=15, world_size=16)
    first_id = os.environ["WANDB_RUN_ID"]
    record = tmp_path / "metadata/wandb_run.json"
    assert json.loads(record.read_text())["id"] == first_id
    assert "synthetic-key-not-a-real-credential" not in record.read_text()
    assert os.environ["WANDB_RESUME"] == "never"
    assert os.environ["WANDB_BASE_URL"] == "https://cerebras.wandb.io"
    assert os.environ["NETRC"] == str(tracking.DEFAULT_NETRC)
    settings["run"]["resume_checkpoint"] = str(tmp_path / "checkpoints")
    tracking.prepare_wandb_environment(settings, rank=23, world_size=24)
    assert os.environ["WANDB_RUN_ID"] == first_id
    assert os.environ["WANDB_RESUME"] == "allow"
    assert not list(record.parent.glob("wandb_run.previous.*.json"))
    settings["wandb"]["project"] = "other-project"
    with pytest.raises(ValueError, match="destination differs"):
        tracking.prepare_wandb_environment(settings, rank=15, world_size=16)


def test_fresh_restart_rotates_deleted_id_and_preserves_cache(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
    settings["run"]["resume_checkpoint"] = None
    record = tmp_path / "metadata/wandb_run.json"
    record.parent.mkdir()
    original = json.dumps(
        {**{key: settings["wandb"][key] for key in ("base_url", "entity", "project")}, "id": "64d9fc3d17613e99"}
    )
    record.write_text(original)
    cache = tmp_path / "dataset_cache"
    cache.mkdir()
    sentinel = cache / "existing-index.npy"
    sentinel.write_bytes(b"unchanged cache")
    # An inherited ID/resume setting must not override the pipeline's fresh-start policy.
    monkeypatch.setenv("WANDB_RUN_ID", "64d9fc3d17613e99")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    tracking.prepare_wandb_environment(settings, rank=0, world_size=16)
    assert record.read_text() == original
    assert not list(record.parent.glob("wandb_run.previous.*.json"))
    tracking.prepare_wandb_environment(settings, rank=15, world_size=16)
    first_id = os.environ["WANDB_RUN_ID"]
    assert first_id != "64d9fc3d17613e99"
    assert os.environ["WANDB_RESUME"] == "never"
    assert json.loads(record.read_text())["id"] == first_id
    archived = list(record.parent.glob("wandb_run.previous.*.json"))
    assert len(archived) == 1
    assert archived[0].read_text() == original
    tracking.prepare_wandb_environment(settings, rank=15, world_size=16)
    assert os.environ["WANDB_RUN_ID"] != first_id
    assert len(list(record.parent.glob("wandb_run.previous.*.json"))) == 2
    assert sentinel.read_bytes() == b"unchanged cache"


def test_checkpoint_resume_without_identity_starts_new_tracking_history(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"].update(directory=str(tmp_path), resume_checkpoint=str(tmp_path / "checkpoints"))
    tracking.prepare_wandb_environment(settings, rank=15, world_size=16)
    assert os.environ["WANDB_RESUME"] == "never"
    assert json.loads((tmp_path / "metadata/wandb_run.json").read_text())["id"] == os.environ["WANDB_RUN_ID"]
    assert "checkpoint resume will start new W&B history" in caplog.text


@pytest.mark.parametrize("run_id", ["", "invalid/id", None])
def test_checkpoint_resume_rejects_invalid_saved_id(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path, run_id: str | None
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"].update(directory=str(tmp_path), resume_checkpoint=str(tmp_path / "checkpoints"))
    record = tmp_path / "metadata/wandb_run.json"
    record.parent.mkdir()
    record.write_text(
        json.dumps({**{key: settings["wandb"][key] for key in ("base_url", "entity", "project")}, "id": run_id})
    )
    with pytest.raises(ValueError, match="Invalid saved W&B run identity"):
        tracking.prepare_wandb_environment(settings, rank=15, world_size=16)


def test_disabled_tracking_does_not_rotate_existing_identity(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
    settings["wandb"]["enabled"] = False
    record = tmp_path / "metadata/wandb_run.json"
    record.parent.mkdir()
    record.write_text("unchanged identity")
    tracking.prepare_wandb_environment(settings, rank=15, world_size=16)
    assert record.read_text() == "unchanged identity"
    assert "WANDB_RUN_ID" not in os.environ


def test_authentication_check_is_read_only(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
    api = Mock(return_value=SimpleNamespace(viewer={"id": "test-user"}))
    init = Mock()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=api, init=init))
    tracking.check_wandb_access(settings)
    api.assert_called_once_with(timeout=15, overrides={"base_url": "https://cerebras.wandb.io"})
    init.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_launcher_sends_synthetic_secret_via_stdin_not_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # These executables replace SSH and Docker entirely, so no actual node is contacted.
    secret = "synthetic-test-key-123456"
    ssh_log = tmp_path / "ssh.jsonl"
    docker_log = tmp_path / "docker.json"
    ssh_stub = tmp_path / "ssh"
    ssh_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shlex, subprocess, sys\n"
        "with open(os.environ['SSH_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "command = shlex.split(sys.argv[-1])\n"
        "if command == ['id', '-u']: sys.stdout.write('2087\\n')\n"
        "elif command == ['id', '-g']: sys.stdout.write('9004\\n')\n"
        "else: sys.exit(subprocess.run(command, input=sys.stdin.buffer.read()).returncode)\n"
    )
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, os, sys\n"
        "with open(os.environ['DOCKER_TEST_LOG'], 'w') as handle:\n"
        "    json.dump({'args': sys.argv[1:], 'key_digest': hashlib.sha256(os.environ['WANDB_API_KEY'].encode()).hexdigest()}, handle)\n"
    )
    ssh_stub.chmod(0o755)
    docker_stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("SSH_TEST_LOG", str(ssh_log))
    monkeypatch.setenv("DOCKER_TEST_LOG", str(docker_log))
    monkeypatch.setenv("WANDB_API_KEY", secret)
    monkeypatch.setenv("MELLUM_ENV_FILE", "")  # Never read the user's real credential file in tests.
    monkeypatch.setenv("MELLUM_NODES", "test-node")
    subprocess.run(["bash", str(SCRIPTS / "launch.sh"), "--wandb-check"], check=True, capture_output=True)
    assert secret not in ssh_log.read_text()
    assert secret not in docker_log.read_text()
    calls = [shlex.split(json.loads(line)[-1]) for line in ssh_log.read_text().splitlines()]
    assert calls[-1][:2] == ["bash", "-c"]
    result = json.loads(docker_log.read_text())
    assert result["key_digest"] == hashlib.sha256(secret.encode()).hexdigest()
    assert "WANDB_API_KEY" in result["args"]
    assert "wandb-check" in result["args"]


@pytest.fixture
def dotenv_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    # Copy only the public launcher. Never read or copy the real .env file.
    launch = tmp_path / "scripts/launch.sh"
    launch.parent.mkdir()
    launch.write_text((SCRIPTS / "launch.sh").read_text())
    ssh_log = tmp_path / "ssh.jsonl"
    docker_log = tmp_path / "docker.json"
    ssh_stub = tmp_path / "ssh"
    ssh_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shlex, subprocess, sys\n"
        "with open(os.environ['SSH_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "command = shlex.split(sys.argv[-1])\n"
        "if command == ['id', '-u']: sys.stdout.write('2087\\n')\n"
        "elif command == ['id', '-g']: sys.stdout.write('9004\\n')\n"
        "else:\n"
        "    remote_env = dict(os.environ)\n"
        "    for key in ('WANDB_API_KEY', 'WANDB_BASE_URL', 'WANDB_HOST'):\n"
        "        remote_env.pop(key, None)\n"
        "    sys.exit(subprocess.run(command, input=sys.stdin.buffer.read(), env=remote_env).returncode)\n"
    )
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, os, sys\n"
        "key = os.environ.get('WANDB_API_KEY')\n"
        "with open(os.environ['DOCKER_TEST_LOG'], 'w') as handle:\n"
        "    json.dump({'args': sys.argv[1:], 'key_digest': hashlib.sha256(key.encode()).hexdigest() if key else None,\n"
        "               'base_url': os.environ.get('WANDB_BASE_URL'), 'host': os.environ.get('WANDB_HOST')}, handle)\n"
    )
    ssh_stub.chmod(0o755)
    docker_stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("SSH_TEST_LOG", str(ssh_log))
    monkeypatch.setenv("DOCKER_TEST_LOG", str(docker_log))
    monkeypatch.setenv("MELLUM_NODES", "test-node")
    for key in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_HOST", "MELLUM_ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    return launch, ssh_log, docker_log


@pytest.mark.parametrize("syntax", ["plain", "quoted", "export", "crlf"])
def test_dotenv_default_path_and_literal_syntax(
    dotenv_launcher: tuple[Path, Path, Path], tmp_path: Path, syntax: str
) -> None:
    launch, ssh_log, docker_log = dotenv_launcher
    secret = "synthetic-dotenv-key"
    url = "https://private.example.com"
    lines = ["# private credentials", "", f"WANDB_API_KEY={secret}", f"WANDB_BASE_URL={url}", f"WANDB_HOST={url}"]
    if syntax == "quoted":
        lines = [f" WANDB_API_KEY = '{secret}' ", f'WANDB_BASE_URL="{url}"', f"WANDB_HOST='{url}'"]
    elif syntax == "export":
        lines = [f"export WANDB_API_KEY={secret}", f"export WANDB_BASE_URL={url}", f"export WANDB_HOST={url}"]
    separator = "\r\n" if syntax == "crlf" else "\n"
    launch.with_name(".env").write_bytes(separator.join(lines).encode())  # No final newline, intentionally.
    result = subprocess.run(["bash", "-x", str(launch), "--wandb-check"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout + result.stderr + ssh_log.read_text() + docker_log.read_text()
    received = json.loads(docker_log.read_text())
    assert received["key_digest"] == hashlib.sha256(secret.encode()).hexdigest()
    assert received["base_url"] == received["host"] == url
    assert url not in ssh_log.read_text()  # URLs travel over stdin too.


def test_dotenv_environment_overrides_file(
    dotenv_launcher: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    launch, _, docker_log = dotenv_launcher
    launch.with_name(".env").write_text("WANDB_API_KEY=synthetic-file-key\nWANDB_BASE_URL=https://file.example.com\n")
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-shell-key")
    monkeypatch.setenv("WANDB_BASE_URL", "https://shell.example.com")
    subprocess.run(["bash", str(launch), "--wandb-check"], check=True, capture_output=True)
    received = json.loads(docker_log.read_text())
    assert received["key_digest"] == hashlib.sha256(b"synthetic-shell-key").hexdigest()
    assert received["base_url"] == "https://shell.example.com"


def test_dotenv_values_never_execute_shell_code(dotenv_launcher: tuple[Path, Path, Path], tmp_path: Path) -> None:
    launch, ssh_log, docker_log = dotenv_launcher
    marker = tmp_path / "must-not-exist"
    secret = f"$(touch {marker})"
    launch.with_name(".env").write_text(f"WANDB_API_KEY='{secret}'\nOTHER=`touch {marker}`\n")
    result = subprocess.run(["bash", str(launch), "--wandb-check"], check=True, capture_output=True, text=True)
    assert not marker.exists()
    assert json.loads(docker_log.read_text())["key_digest"] == hashlib.sha256(secret.encode()).hexdigest()
    assert secret not in result.stdout + result.stderr + ssh_log.read_text() + docker_log.read_text()


@pytest.mark.parametrize("action", ["--wandb-check", "--dry-run"])
def test_dotenv_can_be_disabled_and_secrets_not_forwarded_for_dry_run(
    dotenv_launcher: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    launch, _, docker_log = dotenv_launcher
    launch.with_name(".env").write_text("WANDB_API_KEY=synthetic-file-key\n")
    if action == "--wandb-check":
        monkeypatch.setenv("MELLUM_ENV_FILE", "")
    subprocess.run(["bash", str(launch), action], check=True, capture_output=True)
    assert json.loads(docker_log.read_text())["key_digest"] is None


@pytest.mark.parametrize(
    "contents",
    [
        "WANDB_API_KEY synthetic-hidden-key",
        "WANDB_API_KEY='synthetic-hidden-key",
        "WANDB_API_KEY=synthetic-hidden-key\nWANDB_API_KEY=duplicate",
    ],
)
def test_dotenv_errors_hide_values_before_contacting_nodes(
    dotenv_launcher: tuple[Path, Path, Path], contents: str
) -> None:
    launch, ssh_log, _ = dotenv_launcher
    launch.with_name(".env").write_text(contents)
    result = subprocess.run(["bash", "-x", str(launch), "--wandb-check"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "contents hidden" in result.stderr
    assert "synthetic-hidden-key" not in result.stdout + result.stderr
    assert not ssh_log.exists()


def test_dotenv_explicit_missing_path_fails_before_ssh(
    dotenv_launcher: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch, ssh_log, _ = dotenv_launcher
    monkeypatch.setenv("MELLUM_ENV_FILE", str(tmp_path / "missing.env"))
    result = subprocess.run(["bash", str(launch), "--wandb-check"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not exist" in result.stderr
    assert not ssh_log.exists()


@pytest.mark.parametrize("name", ["WANDB_BASE_URL", "WANDB_HOST"])
def test_environment_destination_mismatch_fails_without_exposing_value(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    monkeypatch.setenv(name, "https://synthetic-hidden-url.example.com")
    with pytest.raises(ValueError, match="differs from YAML") as exc:
        tracking.check_wandb_access(settings)
    assert "synthetic-hidden-url" not in str(exc.value)


def test_matching_environment_destination_allows_trailing_slash(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, tracking = modules
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    url = settings["wandb"]["base_url"]
    monkeypatch.setenv("WANDB_BASE_URL", url + "/")
    monkeypatch.setenv("WANDB_HOST", url + "/")
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Api=Mock(return_value=SimpleNamespace(viewer={"id": "test"})))
    )
    tracking.check_wandb_access(settings)
    assert os.environ["WANDB_BASE_URL"] == url
