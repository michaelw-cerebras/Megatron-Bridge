"""CPU memory telemetry parsing uses only fixture files, never real credentials."""

import importlib
import subprocess
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[4] / "mw_scripts/mellum2_3xh200"
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("limit", ["max", "1099511627776"])
def test_monitor_preserves_memory_categories_and_host_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monitor = importlib.import_module("memory_monitor")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    for name, value in {
        "memory.current": "123456",
        "memory.peak": "654321",
        "memory.max": limit,
        "memory.stat": "anon 100000\nfile 23456\nshmem 1000\nfile_mapped 2000\npagetables 3000\n",
        "memory.events": "oom 1\noom_kill 2\nmax 3\n",
    }.items():
        (cgroup / name).write_text(value)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 200000 kB\nMemAvailable: 150000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
    result = monitor.read_memory_snapshot(cgroup=cgroup, meminfo=meminfo)
    assert result["memory.max"] == ("max" if limit == "max" else int(limit))
    assert result["memory.current"] == 123456
    assert result["memory.peak"] == 654321
    assert result["memory.stat.anon"] == 100000
    assert result["memory.stat.file"] == 23456
    assert result["memory.stat.shmem"] == 1000
    assert result["memory.stat.file_mapped"] == 2000
    assert result["memory.stat.pagetables"] == 3000
    assert result["memory.events.oom_kill"] == 2
    assert result["host.MemAvailable_bytes"] == 150000 * 1024
    assert result["time_utc"].endswith("+00:00")


@pytest.mark.parametrize("limit", [None, "max", "0", "invalid", "1099511627776"])
def test_gpu_launch_memory_guard(tmp_path: Path, limit: str | None) -> None:
    if limit is not None:
        (tmp_path / "memory.max").write_text(limit)
    script = (SCRIPTS / "node.sh").read_text()
    # Exercise the actual shell function, without sourcing node.sh (which would
    # contact datasets/import training packages or potentially launch GPUs).
    function = script[script.index("require_memory_limit() {") : script.index('\ncase "$action" in')]
    result = subprocess.run(
        ["bash", "-c", function + '\nrank=0; require_memory_limit "$1"', "test", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if limit == "1099511627776" else 1)
    if result.returncode:
        assert "memory" in result.stderr
