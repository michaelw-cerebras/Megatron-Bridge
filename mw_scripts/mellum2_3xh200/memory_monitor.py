# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record container CPU memory categories without importing training libraries."""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def read_memory_snapshot(*, cgroup: Path, meminfo: Path) -> dict[str, int | str]:
    """Read cgroup-v2 usage and host headroom, preserving overlapping categories.

    Args:
        cgroup: Directory containing the container's cgroup-v2 memory files.
        meminfo: Host-visible proc meminfo file.

    Returns:
        Byte-valued counters, event counters, and the UTC observation time.
        File, shmem, and file_mapped counters overlap; do not sum them.
    """
    result: dict[str, int | str] = {"time_utc": datetime.now(timezone.utc).isoformat()}
    for filename in ("memory.current", "memory.peak", "memory.max"):
        path = cgroup / filename
        if path.exists():
            value = path.read_text().strip()
            result[filename] = "max" if value == "max" else int(value)
    for filename in ("memory.stat", "memory.events"):
        for line in (cgroup / filename).read_text().splitlines():
            key, value = line.split()
            result[f"{filename}.{key}"] = int(value)
    for line in meminfo.read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            result[f"host.{key}_bytes"] = int(value.split()[0]) * 1024
    return result


def main() -> None:
    """Append memory samples to a per-node JSONL log until the launcher stops us."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--interval", type=float, default=30.0, help="Memory sampling period in seconds")
    parser.add_argument("--cgroup", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Refuse host/ancestor cgroups: the Docker launcher uses a private cgroup namespace.
    if not (args.cgroup / "memory.current").is_file():
        parser.error("A container-private cgroup-v2 directory is required")
    with args.output.open("a", buffering=1) as output:
        while True:
            sample = read_memory_snapshot(cgroup=args.cgroup, meminfo=Path("/proc/meminfo"))
            sample.update(node_rank=args.node_rank, launch_id=args.launch_id)
            output.write(json.dumps(sample, sort_keys=True) + "\n")
            limit = sample.get("memory.max")
            current = sample.get("memory.current")
            if isinstance(limit, int) and isinstance(current, int) and current >= limit * 0.85:
                LOGGER.warning(
                    "CPU memory above 85%% of container limit: %.1f / %.1f GiB", current / 2**30, limit / 2**30
                )
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
