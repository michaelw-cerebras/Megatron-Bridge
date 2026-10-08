# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated YAML and immutable data-mixture snapshots for Mellum training."""

import hashlib
import json
import logging
import math
import os
import re
import struct
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from urllib.parse import urlsplit

import numpy as np

# The runtime image lacks optional PyYAML typing stubs; no new dependency is needed.
import yaml  # type: ignore[import-untyped]


LOGGER = logging.getLogger(__name__)


class ModelSettings(TypedDict):
    """Checkpoint and architecture selection."""

    hf_path: str
    architecture: str


class TrainSettings(TypedDict):
    """Training duration, sequence and batch settings."""

    train_tokens: int
    sequence_length: int
    micro_batch_size: int
    global_batch_size: int
    seed: int


class OptimizerSettings(TypedDict):
    """Explicit Adam and clipping settings."""

    lr: float
    min_lr: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    clip_grad: float


class SchedulerSettings(TypedDict):
    """Learning-rate schedule settings."""

    lr_decay_style: str
    lr_warmup_iters: int


class ParallelSettings(TypedDict):
    """Dense and expert model-parallel meshes."""

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int


class DatasetSettings(TypedDict):
    """Immutable indexed-data mixture and document-boundary ID."""

    eod_id: int
    num_workers: int
    subsets: dict[str, float]
    reserved_token_check: NotRequired[str]


class RunSettings(TypedDict):
    """Output, checkpoint and short-run controls."""

    directory: str
    save_interval: int
    log_interval: int
    resume_checkpoint: str | None
    exit_interval: int | None


class WandbSettings(TypedDict):
    """Non-secret W&B destination and experiment settings."""

    enabled: bool
    base_url: str
    entity: str
    project: str
    run_name: str


class TrainingSettings(TypedDict):
    """The complete validated YAML schema."""

    schema_version: int
    model: ModelSettings
    train: TrainSettings
    optimizer: OptimizerSettings
    scheduler: SchedulerSettings
    parallelism: ParallelSettings
    dataset: DatasetSettings
    run: RunSettings
    wandb: NotRequired[WandbSettings]


class FileIdentity(TypedDict):
    """Cheap immutable-file identity used before training and resume."""

    size: int
    mtime_ns: int


class Shard(TypedDict):
    """A resolved, token-weighted indexed-dataset prefix."""

    prefix: str
    tokens: int
    files: dict[str, FileIdentity]
    weight: float
    subset: str


class MixtureSnapshot(TypedDict):
    """Frozen model, config, data and tokenizer identities."""

    config_digest: str
    config: TrainingSettings
    hf_snapshot: str
    shards: list[Shard]
    tokenizers: dict[str, str]
    reserved_ids_checked_absent: dict[str, list[int]]
    reserved_token_check: NotRequired[str]
    reserved_ids_not_checked: NotRequired[dict[str, list[int]]]


# SafeLoader's base type is unavailable without the optional PyYAML stubs.
class _UniqueLoader(yaml.SafeLoader):  # type: ignore[misc]
    pass


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[object, object]:
    pairs = loader.construct_pairs(node, deep=True)
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = value
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_config(path: Path) -> TrainingSettings:
    """Read YAML, reject unknown fields and validate batch/scheduler semantics."""
    config = yaml.load(path.read_text(), Loader=_UniqueLoader)
    fields = {
        "model": {"hf_path", "architecture"},
        "train": {"train_tokens", "sequence_length", "micro_batch_size", "global_batch_size", "seed"},
        "optimizer": {"lr", "min_lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "clip_grad"},
        "scheduler": {"lr_decay_style", "lr_warmup_iters"},
        "parallelism": {
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        },
        "dataset": {"eod_id", "num_workers", "subsets"},
        "run": {"directory", "save_interval", "log_interval", "resume_checkpoint", "exit_interval"},
    }
    required = set(fields) | {"schema_version"}
    if not isinstance(config, dict) or set(config) - {"wandb"} != required:
        raise ValueError("Expected schema_version and exactly: " + ", ".join(fields))
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    for section, names in fields.items():
        optional = {"reserved_token_check"} if section == "dataset" else set()
        if not isinstance(config[section], dict) or set(config[section]) - optional != names:
            raise ValueError(f"{section} requires exactly these fields: {sorted(names)}")
    policy = config["dataset"].get("reserved_token_check", "strict")
    if not isinstance(policy, str) or policy not in {"strict", "skip"}:
        raise ValueError("dataset.reserved_token_check must be 'strict' or 'skip'")
    if "wandb" in config:
        tracking = config["wandb"]
        names = {"enabled", "base_url", "entity", "project", "run_name"}
        if not isinstance(tracking, dict) or set(tracking) != names:
            raise ValueError(
                f"wandb requires exactly {sorted(names)}; credentials belong in the environment, not YAML"
            )
        if type(tracking["enabled"]) is not bool:
            raise ValueError("wandb.enabled must be a boolean")
        for name in names - {"enabled"}:
            if not isinstance(tracking[name], str) or not tracking[name].strip():
                raise ValueError(f"wandb.{name} must be a nonempty string")
        url = urlsplit(tracking["base_url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("wandb.base_url must be an HTTPS URL without credentials, query, or fragment")
    if config["model"]["architecture"] != "original":
        raise ValueError("Only original Mellum is implemented; surgery variants require a new layer spec")
    if not isinstance(config["model"]["hf_path"], str) or not config["model"]["hf_path"].strip():
        raise ValueError("model.hf_path must name a Hugging Face repository or local checkpoint")
    for section, names in (
        ("train", fields["train"]),
        ("parallelism", fields["parallelism"]),
        ("run", {"save_interval", "log_interval"}),
    ):
        for name in names:
            value = config[section][name]
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{section}.{name} must be an integer in its valid range")
    for name in ("eod_id", "num_workers"):
        if type(config["dataset"][name]) is not int or config["dataset"][name] < 0:
            raise ValueError(f"dataset.{name} must be a nonnegative integer")
    for name, value in config["optimizer"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"optimizer.{name} must be finite")
    opt = config["optimizer"]
    if not 0 <= opt["min_lr"] <= opt["lr"] or opt["lr"] <= 0 or opt["weight_decay"] < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if opt["adam_eps"] <= 0 or opt["clip_grad"] <= 0 or not all(0 <= opt[k] < 1 for k in ("adam_beta1", "adam_beta2")):
        raise ValueError("Invalid Adam or gradient clipping parameters")
    schedule = config["scheduler"]
    if schedule["lr_decay_style"] not in {"constant", "linear", "cosine"}:
        raise ValueError("Unsupported LR schedule")
    train = config["train"]
    iters = math.ceil(train["train_tokens"] / (train["sequence_length"] * train["global_batch_size"]))
    if type(schedule["lr_warmup_iters"]) is not int or not 0 <= schedule["lr_warmup_iters"] < iters:
        raise ValueError("Warmup iterations must be nonnegative and smaller than train_iters")
    for key in ("directory", "resume_checkpoint"):
        value = config["run"][key]
        if value is not None and (not isinstance(value, str) or not Path(value).is_absolute()):
            raise ValueError(f"run.{key} must be an absolute path")
    if config["run"]["directory"] is None:
        raise ValueError("run.directory is required")
    limit = config["run"]["exit_interval"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("exit_interval must be null or a positive integer")
    subsets = config["dataset"]["subsets"]
    if not isinstance(subsets, dict) or not subsets:
        raise ValueError("dataset.subsets must be a nonempty full-path -> weight mapping")
    for prefix, weight in subsets.items():
        if not isinstance(prefix, str) or not Path(prefix).is_absolute():
            raise ValueError("Dataset paths must be absolute")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (float, int))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"Invalid positive weight for {prefix}")
    # The checks above validate the complete external schema before this boundary.
    return cast(TrainingSettings, config)


def config_digest(config: TrainingSettings) -> str:
    """Identify training/data settings, excluding tracking and resume/exit controls."""
    semantic = json.loads(json.dumps(config))
    # Adding tracking must not invalidate an existing prepared mixture/checkpoint.
    semantic.pop("wandb", None)
    # Explicit strict and older configurations have the same validation semantics.
    # Keep skip in the identity so an existing run cannot silently weaken its policy.
    if semantic["dataset"].get("reserved_token_check", "strict") == "strict":
        semantic["dataset"].pop("reserved_token_check", None)
    semantic["run"].pop("resume_checkpoint")
    semantic["run"].pop("exit_interval")
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def validate_parallelism(config: TrainingSettings, *, world_size: int, experts: int) -> int:
    """Validate dense and expert meshes; return the gradient accumulation count."""
    if world_size < 1 or experts < 1:
        raise ValueError("World size and expert count must be positive")
    parallel = config["parallelism"]
    tp = parallel["tensor_model_parallel_size"]
    pp = parallel["pipeline_model_parallel_size"]
    cp = parallel["context_parallel_size"]
    ep = parallel["expert_model_parallel_size"]
    etp = parallel["expert_tensor_parallel_size"]
    if world_size % (tp * pp * cp) or world_size % (ep * etp * pp) or experts % ep:
        raise ValueError("GPU count must divide both parallel meshes, and EP must divide the expert count")
    dp = world_size // (tp * pp * cp)
    denominator = config["train"]["micro_batch_size"] * dp
    if config["train"]["global_batch_size"] % denominator:
        raise ValueError(f"GBS must be divisible by MBS*DP={denominator}")
    if cp > 1 and config["train"]["sequence_length"] % (2 * cp):
        raise ValueError("Sequence length must be divisible by 2*CP")
    return config["train"]["global_batch_size"] // denominator


def index_token_count(prefix: Path) -> int:
    """Read token counts from a Megatron index and reject truncated bin/idx pairs."""
    index = Path(f"{prefix}.idx")
    binary = Path(f"{prefix}.bin")
    with index.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise ValueError(f"Not a Megatron mmap index: {index}")
        version, dtype, sequences, documents = struct.unpack("<QBQQ", stream.read(25))
        if version != 1 or dtype not in {1, 2, 3, 4, 5, 8}:
            raise ValueError(f"Unsupported index format: {index}")
        if index.stat().st_size != 34 + sequences * 12 + documents * 8:
            raise ValueError(f"Truncated or multimodal index: {index}")
        lengths = np.fromfile(stream, dtype="<i4", count=sequences)
        if not sequences or np.any(lengths < 0):
            raise ValueError(f"Empty or invalid dataset: {prefix}")
        tokens = int(lengths.sum(dtype=np.int64))
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 8: 2}
    if tokens <= 0 or binary.stat().st_size != tokens * sizes[dtype]:
        raise ValueError(f"Empty/truncated bin file or incorrect dtype: {binary}")
    return tokens


def check_reserved_ids_absent(prefix: Path, ids: set[int]) -> None:
    """Reject data using reserved IDs whose meanings changed in the target tokenizer."""
    with Path(f"{prefix}.idx").open("rb") as stream:
        stream.seek(17)
        dtype_code = stream.read(1)[0]
    dtypes = {1: "u1", 2: "i1", 3: "<i2", 4: "<i4", 5: "<i8", 8: "<u2"}
    # Numpy cannot infer a fixed dtype from the validated index's dynamic dtype code.
    data = np.memmap(f"{prefix}.bin", mode="r", dtype=dtypes[dtype_code])  # type: ignore[var-annotated]
    LOGGER.info("Checking renamed reserved IDs %s in %s", sorted(ids), prefix)
    for start in range(0, len(data), 16 * 1024 * 1024):
        chunk = data[start : start + 16 * 1024 * 1024]
        # First narrow the range to avoid one full comparison per renamed ID.
        candidates = chunk[(chunk >= min(ids)) & (chunk <= max(ids))]
        if len(candidates) and np.isin(candidates, list(ids)).any():
            raise ValueError(f"Data contains renamed reserved token IDs {sorted(ids)}: {prefix}; retokenize it")


def prepare_mixture(config: TrainingSettings, *, hf_snapshot: str) -> MixtureSnapshot:
    """Validate mappings, apply the reserved-ID policy, and resolve shard weights."""
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_snapshot)
    vocab_size = AutoConfig.from_pretrained(hf_snapshot).vocab_size
    target_vocab = tokenizer.get_vocab()
    records: list[Shard] = []
    manifests: dict[Path, str] = {}
    renamed_by_manifest: dict[Path, set[int]] = {}
    # Missing policy preserves the original strict behavior for older YAML files.
    policy = config["dataset"].get("reserved_token_check", "strict")
    seen = set()
    total_weight = sum(config["dataset"]["subsets"].values())
    for path, weight in config["dataset"]["subsets"].items():
        root = Path(path)
        prefixes = sorted(index.with_suffix("") for index in root.rglob("*.idx")) if root.is_dir() else [root]
        if not prefixes:
            raise ValueError(f"No completed .idx/.bin shards in {root}")
        subset: list[Shard] = []
        for prefix in prefixes:
            identity = str(prefix.resolve())
            if identity in seen:
                raise ValueError(f"Overlapping subset/shard: {prefix}")
            seen.add(identity)
            manifest_path = next(
                (p / "manifest.json" for p in prefix.parents if (p / "manifest.json").is_file()), None
            )
            if manifest_path is None:
                raise ValueError(f"No tokenizer manifest for {prefix}")
            if manifest_path not in manifests:
                manifest = json.loads(manifest_path.read_text())
                metadata = manifest["tokenizer"]
                if metadata["eos_id"] != config["dataset"]["eod_id"]:
                    raise ValueError(f"Dataset EOD differs from config: {manifest_path}")
                token_file = manifest_path.parent / metadata["json_path"]
                digest = hashlib.sha256(token_file.read_bytes()).hexdigest()
                if digest != metadata["json_sha256"]:
                    raise ValueError(f"Tokenizer checksum mismatch: {token_file}")
                source_vocab = Tokenizer.from_file(str(token_file)).get_vocab()
                if max(source_vocab.values()) >= vocab_size:
                    raise ValueError(f"Dataset tokenizer contains IDs outside vocab_size={vocab_size}: {token_file}")
                mismatches = {
                    token: index for token, index in source_vocab.items() if target_vocab.get(token) != index
                }
                # Base -> Thinking renames unused reserved slots, not ordinary vocabulary.
                # Never accept these aliases without proving they are absent in the data.
                target_ids = set(target_vocab.values())
                if any(
                    re.fullmatch(r"<\|extra_token_[0-9]+\|>", token) is None or index not in target_ids
                    for token, index in mismatches.items()
                ):
                    raise ValueError(f"Token ID mapping is incompatible with the HF checkpoint: {token_file}")
                renamed_by_manifest[manifest_path] = set(mismatches.values())
                manifests[manifest_path] = digest
                if policy == "skip" and mismatches:
                    LOGGER.warning(
                        "Skipping full bin scan for renamed reserved IDs %s in %s; absence is UNVERIFIED",
                        sorted(mismatches.values()),
                        manifest_path,
                    )
            tokens = index_token_count(prefix)
            renamed = renamed_by_manifest[manifest_path]
            if renamed and policy == "strict":
                check_reserved_ids_absent(prefix, renamed)
            subset.append(
                {
                    "prefix": str(prefix),
                    "tokens": tokens,
                    "weight": 0.0,
                    "subset": path,
                    "files": {
                        suffix: {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                        for suffix in ("idx", "bin")
                        for stat in [Path(f"{prefix}.{suffix}").stat()]
                    },
                }
            )
        total_tokens = sum(record["tokens"] for record in subset)
        for record in subset:
            record["weight"] = weight / total_weight * record["tokens"] / total_tokens
            record["subset"] = path
            records.append(record)
    return {
        "config_digest": config_digest(config),
        "config": config,
        "hf_snapshot": hf_snapshot,
        "shards": records,
        "tokenizers": {str(path): digest for path, digest in manifests.items()},
        "reserved_token_check": policy,
        "reserved_ids_checked_absent": {
            str(path): sorted(ids) for path, ids in renamed_by_manifest.items() if policy == "strict"
        },
        "reserved_ids_not_checked": {
            str(path): sorted(ids) for path, ids in renamed_by_manifest.items() if policy == "skip" and ids
        },
    }


def validate_snapshot_files(mixture: MixtureSnapshot) -> None:
    """Fail before training if a frozen shard or tokenizer has been replaced."""
    for record in mixture["shards"]:
        for suffix, expected in record["files"].items():
            path = Path(f"{record['prefix']}.{suffix}")
            stat = path.stat()
            if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(f"Frozen dataset file changed: {path}; use a new run directory")
    for manifest_name, expected_digest in mixture["tokenizers"].items():
        metadata = json.loads(Path(manifest_name).read_text())["tokenizer"]
        tokenizer_path = Path(manifest_name).parent / metadata["json_path"]
        if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f"Frozen tokenizer changed: {tokenizer_path}")


def write_json_atomic(path: Path, value: MixtureSnapshot) -> None:
    """Publish a metadata snapshot without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
