# Bring-up 验证记录：2026-10-07

launcher 目前默认只使用 GPU 31、32（16 卡）。下面的 24 卡结果是此前三节点 bring-up 的历史实测，
不是新两节点布局的实测，也不证明 24 → 16 卡 checkpoint 恢复已经验证。
本次拓扑调整的配置/launcher 单元测试为 11 passed，覆盖 16/24 卡 batch 整除与 mock SSH 节点选择；
shell 语法检查通过，未连接 GPU 33，也未启动新的 16 卡训练或网络测试。

实现保留原始 Mellum2 的串行 attention → MoE。没有修改 Megatron-LM submodule，
也没有启动 50B token 或真实 12B 的 24 卡长跑。

## 已完成

| 检查 | 结果 |
|---|---|
| 新增单元测试 | 16 passed |
| 小模型单卡、TP=2 / EP=2 功能测试 | 2 passed；87 个权重精确往返 |
| 3 节点 / 24 GPU NCCL | all-reduce 正确；日志确认 AWS OFI / EFA |
| EFA smoke 的 64 MiB all-reduce | algorithm bandwidth 约 26.61 GB/s；不是全面性能 benchmark |
| 24 卡小模型训练 | 0 → 2 updates，完整 checkpoint 保存 |
| 24 卡恢复 | 2 → 4 → 6 updates；加载 optimizer/RNG/scheduler，HF_load=False |
| 真实 Math 数据 prepare | 201,836,617 tokens；完整检查 bin/idx、词表与保留 ID |
| Ruff / 格式 / shell 语法 | 通过 |
| configuration.py strict mypy | 通过；未新增依赖 |
| 定向 pre-commit | 通过；按用户要求保留 user_manual.md，跳过文件名 underscore hook |

## 真实 Thinking checkpoint

HF revision：`a7311550557e93cc706ab5dd3d879c1a11703ab4`。
最终实现的结果保存在 `runs/validation/thinking-hf-routing.json`：

- 5,631 个权重精确往返。
- 1152-token 序列，跨过真实 sliding window=1024 的边界。
- logits cosine：`0.9999158978462219`，高于脚本默认门槛 `0.9999`。
- next-token argmax agreement：`1.0`。
- Megatron CE：`0.030233437195420265`；HF CE：`0.030501065775752068`。
- max absolute logit difference：`9.0`。这是 BF16 数值检查，不是逐值或 bitwise logits 等价。
- 64-token full-prefix greedy generation 产生连贯的 `<think>` 思考文本；不是完整代码质量评测。

镜像的 Transformers 未内置 Mellum；HF 对照使用本实现的 compatibility reference，
其 attention/rotary/routing 定义参考
[官方 Mellum 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mellum/modeling_mellum.py)，
RoPE 用 HF 原有的 YaRN 函数计算。尚未与安装官方原生 Mellum 的独立环境交叉验证。

小模型训练与恢复使用独立 synthetic data，不应当作真实 12B / seq=8192 吞吐或显存验证。
生产前仍需按 user_manual.md 对真实数据、真实模型执行短训练 smoke，并确定 mixture/LR/budget。

## 数据兼容性与保留资产

Base 与 Thinking 普通词表的 token IDs 一致；6 个 reserved IDs（27–32）被重命名。
仅当所选所有 shards 中这些 ID 均未出现，pipeline 才允许使用这些 Base-tokenized 数据。
目前完成了上述 Math shard 的完整检查；**没有扫描或批准全部 mixture**。

三台节点保留 `mellum2_3xh200` 容器。完整 HF checkpoint 保存在共享 `.cache/huggingface/`。
checkpoint、日志、数据 snapshot 和 toy fixture 保留在忽略版本控制的 `runs/validation/`。
恢复时会归档旧节点日志，当前 `train_node_*.log` 对应最后一段训练。

不支持的 Mellum 变体会显式报错；parallel attention/MoE model surgery 尚未实现。

## W&B integration checks

- Configured the private instance for `team-inferenceml/draft_pretrain`.
- Configuration, logger mapping, persistent run identity, and mock SSH credential transport: 26 tests passed.
- Both selected containers (GPU nodes 31 and 32) have W&B SDK 0.28.1 and can reach the instance (HTTP 200).
- Python formatting/lint hooks and shell syntax checks passed.
- No credentials were embedded in code/YAML, and no real authentication, project creation, or online upload was performed.
- Use `launch.sh --wandb-check` after container login or environment authentication; project write access still needs an online run.

## Writable output mount checks: 2026-10-07

- Created new `mellum2_3xh200_rw` containers on GPU 31 and 32 only. Original containers and the active preparation process on GPU 31 were preserved.
- Verified `/cb/ml-olympus` is mounted read-only, with a separate read-write bind mount for `/cb/ml-olympus/codesign_pretrain/runs`.
- Both nodes passed preflight, including actual temporary-file creation/removal in the configured run directory, visibility of eight H200 GPUs, EFA provider discovery, and Mellum Bridge imports.
- Temporary files created as the training UID on each node were readable/writable from the other node; both probes were removed afterward.
- New and original containers use the same image. Checked runtime versions match: Transformers 5.8.1, W&B 0.28.1, and PyTorch `2.12.0a0+0291f960b6.nv26.4.48445190`.
- Targeted configuration/launcher/W&B tests: 69 passed. Targeted pre-commit hooks and shell syntax passed; the user-requested Markdown filename exception remains in place.
- Image imports emit existing optional Triton-kernel warnings; preflight exits successfully. This check does not establish real-model training success or repeat the multi-GPU network test.
- No training, additional mixture preparation, real W&B authentication, or checkpoint migration was started during this storage fix.
- Earlier reserved-ID compatibility results above describe strict scans. The current experiment explicitly selects `reserved_token_check: skip`; absence of IDs 27–32 in the full mixture remains unverified.

## Container name replacement: 2026-10-07

- At the user's explicit request, stopped and deleted the old `mellum2_3xh200` containers on GPU 31 and 32, terminating the remaining remote preparation process on GPU 31.
- Renamed the writable-output containers from `mellum2_3xh200_rw` to `mellum2_3xh200`; their container IDs, runtime environments, and mounts were retained. Launcher and manual now use the original name by default.
- Only container writable layers were removed. No volumes, bind-mounted training data, HF cache, or run directories were deleted. GPU 33 was not contacted.
- Documented that coordinator Ctrl+C is not guaranteed to terminate remote preparation, and that the current PID-based `--stop` operation covers training/network tests, not preparation.
- Verified the old preparation PID is gone, both renamed containers are running with the expected mounts, and the configured run does not yet have a published mixture snapshot. Preparation must be rerun before dry-run/training.
- Re-ran the targeted configuration/launcher/W&B tests: 69 passed. Targeted lint/format hooks and shell syntax checks passed.
