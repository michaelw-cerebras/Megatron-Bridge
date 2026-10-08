# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused single/two-GPU Mellum conversion and hybrid-attention parity."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 2)])
def test_toy_conversion_and_forward(tp: int, ep: int, tmp_path: Path) -> None:
    """Round-trip every tensor and cross the sliding-window boundary on a toy model."""
    if torch.cuda.device_count() < tp:
        pytest.skip(f"Requires {tp} GPUs")
    repo = Path(__file__).resolve().parents[4]
    result = tmp_path / "parity.json"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={tp}",
            str(repo / "mw_scripts/mellum2_3xh200/validate.py"),
            "--toy-dir",
            str(tmp_path / "hf"),
            "--tp",
            str(tp),
            "--ep",
            str(ep),
            "--output",
            str(result),
        ],
        cwd=repo,
        check=True,
        timeout=600,
    )
    metrics = json.loads(result.read_text())
    assert metrics["weights_exact"] == 87
    assert metrics["cosine"] >= 0.9999
    assert abs(metrics["loss"] - metrics["hf_loss"]) < 0.02
