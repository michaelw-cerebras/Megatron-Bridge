# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure W&B without storing credentials in training metadata."""

import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING

from configuration import TrainingSettings


if TYPE_CHECKING:
    from megatron.bridge.training.config import LoggerConfig


LOGGER = logging.getLogger(__name__)
DEFAULT_NETRC = Path(__file__).parent / ".cache" / "wandb-auth.netrc"


def apply_wandb_config(logger: "LoggerConfig", settings: TrainingSettings) -> None:
    """Map an optional, non-secret YAML section onto the Bridge logger."""
    tracking = settings.get("wandb")
    if tracking is None or not tracking["enabled"]:
        logger.wandb_project = None
        logger.wandb_exp_name = ""
        logger.wandb_entity = None
        logger.wandb_save_dir = None
        return
    logger.wandb_project = tracking["project"]
    logger.wandb_exp_name = tracking["run_name"]
    logger.wandb_entity = tracking["entity"]
    logger.wandb_save_dir = str(Path(settings["run"]["directory"]) / "wandb")


def _set_destination(settings: TrainingSettings) -> None:
    # YAML is authoritative, avoiding accidental uploads to the public W&B cloud.
    destination = settings["wandb"]["base_url"].rstrip("/")
    for name in ("WANDB_BASE_URL", "WANDB_HOST"):
        requested = os.environ.get(name)
        if requested and requested.rstrip("/") != destination:
            raise ValueError(f"{name} differs from YAML wandb.base_url; use the same W&B destination")
    os.environ["WANDB_BASE_URL"] = destination
    os.environ.setdefault("NETRC", str(DEFAULT_NETRC))
    os.environ["WANDB_MODE"] = "online"


def check_wandb_access(settings: TrainingSettings) -> None:
    """Check SDK authentication without creating a project/run or loading a model."""
    tracking = settings.get("wandb")
    if tracking is None or not tracking["enabled"]:
        LOGGER.info("W&B is disabled")
        return
    _set_destination(settings)
    import wandb

    api = wandb.Api(timeout=15, overrides={"base_url": os.environ["WANDB_BASE_URL"]})
    if api.viewer is None:
        raise RuntimeError("W&B authentication failed; log in inside the container or export WANDB_API_KEY")
    LOGGER.info(
        "W&B authentication OK: %s; destination %s/%s", tracking["base_url"], tracking["entity"], tracking["project"]
    )
    LOGGER.info("No project/run created. Project write permission is checked by the first online training run.")


def prepare_wandb_environment(settings: TrainingSettings, *, rank: int, world_size: int) -> None:
    """Create a fresh tracking identity, or reuse it for explicit checkpoint resume."""
    tracking = settings.get("wandb")
    if tracking is None or not tracking["enabled"] or rank != world_size - 1:
        return
    _set_destination(settings)
    directory = Path(settings["run"]["directory"])
    (directory / "wandb").mkdir(parents=True, exist_ok=True)
    record_path = directory / "metadata" / "wandb_run.json"
    destination = {
        "base_url": tracking["base_url"].rstrip("/"),
        "entity": tracking["entity"],
        "project": tracking["project"],
    }
    resume = settings["run"]["resume_checkpoint"] is not None
    reuse_identity = resume and record_path.exists()
    if reuse_identity:
        record = json.loads(record_path.read_text())
        if not isinstance(record, dict) or any(record.get(name) != value for name, value in destination.items()):
            raise ValueError("W&B destination differs from saved run identity; use a new run.directory")
        run_id = record.get("id")
        if not isinstance(run_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id) is None:
            raise ValueError("Invalid saved W&B run identity")
    else:
        run_id = secrets.token_hex(8)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.exists():
            # Preserve the old identity without retaining it for a fresh HF start.
            archived = record_path.with_name(f"wandb_run.previous.{time.time_ns()}.{run_id}.json")
            with archived.open("x") as handle:
                handle.write(record_path.read_text())
            LOGGER.info("Archived previous W&B identity to %s", archived)
        elif resume:
            LOGGER.warning("No saved W&B identity at %s; checkpoint resume will start new W&B history", record_path)
        temporary = record_path.with_name(f".wandb_run.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({**destination, "id": run_id}, indent=2) + "\n")
        temporary.replace(record_path)
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow" if reuse_identity else "never"
    LOGGER.info(
        "W&B run identity (%s): %s/%s/%s",
        "resume" if reuse_identity else "new",
        tracking["entity"],
        tracking["project"],
        run_id,
    )
