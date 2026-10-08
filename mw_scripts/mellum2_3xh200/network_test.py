# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify 24-rank NCCL collectives before loading the model."""

import argparse
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    """Check correctness and report bandwidth for repeated cross-node all-reduces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)  # identifies this owned launch in stop/status
    parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        tensor = torch.full((16 * 1024 * 1024,), float(rank + 1), device="cuda")
        dist.all_reduce(tensor)
        expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
        if not torch.all(tensor == expected):
            raise RuntimeError("NCCL all-reduce correctness failure")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            tensor.zero_()
            dist.all_reduce(tensor)
        dist.barrier()
        start.record()
        for _ in range(20):
            tensor.zero_()
            dist.all_reduce(tensor)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000
        if rank == 0:
            logging.getLogger(__name__).info(
                "NCCL PASS: world=%d payload=64MiB average=%.3fs algorithm_bw=%.2fGB/s",
                dist.get_world_size(),
                seconds / 20,
                20 * tensor.numel() * 4 / seconds / 1e9,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
