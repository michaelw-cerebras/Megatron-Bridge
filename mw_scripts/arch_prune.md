# Architecture Pruning Proposal

## Goal

The goal is to prune or modify existing small-to-medium models into shapes that provide the best latency and throughput on Cerebras hardware, while retaining as much value as possible from pretrained weights.

Instead of initializing each target model from scratch, we propose to transform pretrained weights into the desired architecture and then continue pretraining (CPT):

- Slice, clone, merge, or initialize rows and columns when changing weight-matrix dimensions.
- Prune, clone, or merge experts when changing the number of MoE experts.
- Apply existing methods for vocabulary expansion, layer pruning, and KV-head pruning.
- Use local reconstruction objectives and short CPT stages to recover from less invasive transformations before applying the most invasive changes.

## System Boundary and Checkpoint Flow

Weight surgery should be performed offline, before Megatron training. The intended handoff is:

```text
Original Hugging Face checkpoint
    -> offline pruning / weight surgery / reconstruction
    -> canonical target HF-style checkpoint (BF16 safetensors + config)
    -> one-time Megatron-Bridge import
    -> Megatron distributed checkpoint
    -> Megatron pretraining and resume
```

Megatron should not rerun pruning logic or repeatedly import safetensors when resuming training. After the first successful Bridge import, the Megatron distributed checkpoint becomes the training checkpoint of record.

The offline output must be a complete HF-style checkpoint directory, not only a standalone safetensors file:

```text
pruned-model/
├── config.json
├── model.safetensors
│   # or sharded model-*.safetensors plus model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Prefer BF16 tensors at this boundary. If the offline output uses FP8, FP4, or another quantized representation, its scale tensors and quantization metadata must also be preserved, and the Bridge import path must explicitly dequantize or interpret them.

## Architecture and Code-Reuse Strategy

The target models should be treated as a dedicated, configurable canonical pruned-MoE family rather than labeling every output as `gpt_oss`.

The source models share a desired hardware-oriented shape, but they intentionally retain different computation semantics:

- Mellum retains Mellum SwiGLU and QK norm.
- Ling/Ring retains its SwiGLU, sigmoid router, QK norm, and shared expert.
- Trinity retains its SwiGLU, sigmoid router, QK norm, and shared expert.
- GPT-OSS has its own activation, bias, router, attention-sink, expert-layout, and YaRN conventions.

Therefore, reusing the existing GPT-OSS recipe is acceptable for early random-initialized throughput experiments, but the stock `GPTOSSBridge` must not become the long-term checkpoint contract for every source family.

The recommended ownership boundary is:

```text
Source-specific offline adapters
├── GPT-OSS -> canonical pruned checkpoint
├── Mellum2 -> canonical pruned checkpoint
├── Ling/Ring -> canonical pruned checkpoint
└── Trinity -> canonical pruned checkpoint

Canonical pruned-MoE model family
├── one explicit HF config schema and tensor-key layout
├── one Megatron Bridge
├── stock GPTModelProvider and Megatron-Core modules where possible
└── narrowly scoped model-specific ModuleSpecs only when required
```

Most runtime components should be reused from Megatron-Core: GQA, dense/sliding-window attention, QK norm, learnable attention sinks, shared experts, sigmoid/softmax routing, expert bias, grouped MoE, and RoPE/YaRN. Source-specific tensor surgery belongs in the offline adapters, not in the training loop.

## Canonical Target Checkpoint Contract

Before implementing the offline pruners, define one versioned target checkpoint schema. Its `config.json` must capture computation semantics as well as tensor shapes.

At minimum, the schema must specify:

### Core architecture

- `model_type` and `architectures` for the canonical pruned family.
- Vocabulary size and special-token IDs.
- Number of layers, hidden size, FFN/expert width, and normalization epsilon.
- Number of Q heads, number of KV heads, and head dimension.
- Whether input embeddings and the output layer are tied.
- Whether MTP or any other auxiliary prediction heads are present.
- Which layers are dense MLP versus MoE.

### Attention

- An explicit per-layer dense/SWA pattern rather than an implicit frequency when possible.
- Sliding-window size.
- Whether QK norm is enabled, including the exact tensor shape/layout.
- Whether attention sinks are enabled, their shape, which layers use them, and their initialization rule.
- Bias policy for Q, K, V, and output projections.

### Positional encoding

- RoPE type and base/theta.
- Scaling factor, original context length, and target context length.
- YaRN beta parameters and any truncation behavior.
- Whether dense and SWA layers use different RoPE/scaling rules.

### Activation and MLP

- Exact activation formula; the name `SwiGLU` alone is insufficient.
- Whether activation clamp or linear offset is used.
- Gate/up projection packing or interleaving convention.
- Bias policy for routed experts, shared experts, and dense MLPs.

### MoE router and experts

- Total experts, active experts per token, and shared-expert width.
- Router score function (`softmax`, `sigmoid`, or another function).
- Whether scores are computed pre-softmax and whether selected scores are renormalized.
- Routed scaling factor.
- Group-limited top-k settings, if used.
- Expert correction bias, its initialization, and its update rate.
- Router auxiliary loss, z-loss, and coefficients.
- Canonical expert tensor layout: per-expert versus stacked, and FC1/FC2 orientation.

The schema should be explicit enough that two independent implementations construct the same computation graph from `config.json` alone.

## Proposed Sequence of Model Modifications

1. **Vocabulary expansion.** This is relatively non-invasive. Apply the vocabulary transformation and adapt with a short CPT stage.
2. **Changes that do not alter hidden size.** This includes KV-head pruning, dense/SWA changes, total/active expert-count changes, and expert-width changes. Apply local layer-wise reconstruction followed by short CPT.
3. **Layer pruning and hidden-dimension expansion.** These are the most invasive transformations. Start the main pretraining stage from this checkpoint and compare candidates based on training dynamics.

Each stage should produce a recoverable checkpoint and a transformation manifest recording the source checkpoint revision, transformation parameters, selected layers/heads/experts, initialization methods, and validation metrics.

## Proposal 1: GPT-OSS-20B Prune

- Source: `openai/gpt-oss-20b`.
- Keep the existing hidden dimension.
- Prune attention+MoE blocks from 24 to 16 layers.
- Prune KV heads from 8 to 1.
- Change Q heads from 64 to 32 and head dimension from 64 to 128.
- Apply an iso-FLOP MoE transformation: double total/active experts and halve expert width.
- Further double the active experts to reach 16 active experts out of 64 total experts.
- Optionally add one shared expert.

Open decisions specific to this proposal:

- Select the exact 16 source layers.
- Define how 8 KV heads are reduced to 1 and how 64 Q heads are converted to 32 wider heads.
- Define the expert cloning/initialization method and router initialization after expansion.
- Decide whether the shared expert is part of the canonical target or an ablation.
- Decide whether the target attention pattern is GPT-OSS alternating dense/SWA, all-SWA, or another explicit pattern.

## Proposal 2: JetBrains/Mellum2-12B-A2.5B Prune

Candidate checkpoints include Base-Pretrain, Base, Instruct, Instruct-SFT, Thinking, and Thinking-SFT.

- Expand vocabulary from 98,304 to 201,088.
- Prune layers from 28 to 16.
- Prune KV heads from 4 to 1.
- Expand hidden dimension from 2,304 to 2,880.
- Expand expert width from 896 to 1,440.
- Increase active experts from 8 to 16.
- Keep 64 total experts.
- Keep Mellum's SwiGLU rather than converting to GPT-OSS activation semantics.
- Update RoPE for the new dense/SWA pattern.
- Do not add a shared expert.
- Add attention sinks and retain Mellum's QK norm.

Open decisions specific to this proposal:

- Select the source checkpoint and exact 16 source layers.
- Define hidden-dimension and expert-width expansion initialization.
- Define the new dense/SWA pattern and exact RoPE parameters.
- Define QK-norm expansion/mapping and attention-sink initialization.
- Specify Mellum's exact activation, projection packing, and bias semantics in the canonical config.

## Proposal 3: inclusionAI Ling/Ring-mini-2.0 Prune

The source checkpoint name must be resolved before implementation. The proposal title currently refers to Ling, while the retained architecture description refers to Ring. Record the exact Hugging Face repository and revision in the transformation manifest.

- Expand vocabulary from 157,184 to 201,088.
- Prune layers from 20 to 16.
- Increase Q heads from 16 to 32.
- Prune KV heads from 4 to 1.
- Expand hidden dimension from 2,048 to 2,880.
- Expand expert width from 512 to 1,440.
- Increase active experts from 8 to 16.
- Reduce total experts from 256 to 64.
- Update RoPE for the new dense/SWA pattern.
- Keep the shared expert.
- Keep Ling/Ring SwiGLU rather than converting to GPT-OSS activation semantics.
- Add attention sinks and retain sigmoid routing and QK norm.

Open decisions specific to this proposal:

- Decide whether MTP and any initially dense layers are retained or removed.
- Select the exact 16 source layers and the 64 experts retained or merged from 256.
- Define how Q heads expand from 16 to 32 while hidden size and head layout also change.
- Preserve or redefine group-limited top-k routing, selected-score normalization, routed scaling, expert correction bias, and bias update rate.
- Define shared-expert width, gating behavior, and weight-expansion method.
- Define the new dense/SWA pattern, RoPE parameters, QK-norm mapping, and attention-sink initialization.

## Proposal 4: arcee-ai/Trinity-Mini Prune

- Expand vocabulary from 98,304 to 201,088.
- Prune layers from 28 to 16.
- Prune KV heads from 4 to 1.
- Expand hidden dimension from 2,048 to 2,880.
- Expand expert width from 1,024 to 1,440.
- Increase active experts from 8 to 16.
- Reduce total experts from 128 to 64.
- Keep Trinity's SwiGLU rather than converting to GPT-OSS activation semantics.
- Update RoPE for the new dense/SWA pattern.
- Keep the shared expert.
- Add attention sinks and retain QK norm and sigmoid routing.

Open decisions specific to this proposal:

- Select the exact source checkpoint revision and 16 source layers.
- Select or merge 64 experts from the original 128 and define the corresponding router transformation.
- Define hidden-dimension, expert-width, shared-expert, and QK-norm expansion methods.
- Preserve or redefine the full sigmoid-router semantics, not only its score-function name.
- Define the new dense/SWA pattern, RoPE parameters, and attention-sink initialization.

## Offline Weight-Surgery Requirements

Each source adapter must explicitly define and record:

- Vocabulary mapping and initialization of newly added tokens.
- Layer-selection indices.
- Q/K/V head selection, grouping, cloning, averaging, or reconstruction method.
- Hidden-dimension slice/clone/expansion method for every affected tensor, including norms and embeddings.
- Expert selection metric and whether experts are selected, cloned, merged, or reconstructed.
- Router transformation corresponding to the expert transformation.
- Expert-width and shared-expert transformation.
- Attention-sink initialization.
- Dense/SWA and positional-encoding conversion.
- Tensor names, layouts, orientations, and dtypes in the canonical output.

Do not silently pad or truncate mismatched tensors. Every transformation should be deterministic from a versioned specification and should fail on unexpected source shapes.

## Validation Gates

### Offline checkpoint validation

- All tensors declared by the target config exist and have the expected shape and dtype.
- There are no unexpected or unused source/target tensors unless explicitly documented.
- The tokenizer vocabulary and embedding/output dimensions agree.
- Tensor statistics are finite and reasonable after every transformation.

### Numerical and reconstruction validation

- Identity/no-op transformations preserve logits within the expected BF16 tolerance.
- Each locally modified layer is evaluated with a reconstruction objective against its source behavior.
- Router load distribution, dropped-token rate, expert utilization, and shared-expert contribution are measured.
- A short CPT run verifies finite loss/gradients and compares initial loss and convergence speed across candidates.

Exact end-to-end logit parity is not expected after architecture-changing pruning; the relevant acceptance criteria are reconstruction quality, stable training, and convergence relative to random initialization and other candidates.

### Bridge validation

- Import the canonical HF checkpoint into Megatron without missing or unexpected mappings.
- Save and resume from a Megatron distributed checkpoint.
- Perform an HF-to-Megatron-to-HF round trip for the canonical format.
- Verify inference or loss consistency before and after the Bridge round trip.
- Verify that quantized inputs, if ever used, are correctly dequantized rather than loaded as raw integer/FP8 values.

### Performance validation

- Measure Cerebras compile success, latency, throughput, and memory for every target shape.
- Measure Megatron training throughput separately from Cerebras inference performance.
- Record sequence length, batch size, precision, parallelism, attention pattern, and active-expert count with every result.
- Update FLOPs accounting if the final model introduces a computational block not represented by the standard attention/MLP/MoE formulas.

## Implementation Decision

Use an independent canonical pruned-MoE Bridge and config schema, backed by the stock `GPTModelProvider` wherever existing Megatron-Core components reproduce the required computation. Add custom model-family modules only for semantics that Megatron-Core cannot express correctly.

Keep source-specific pruning and tensor-layout conversion in offline adapters. Do not place source-family conditionals in shared conversion infrastructure unless the behavior is genuinely reusable by multiple model families.

The current GPT-OSS-based `mini_gptoss.py` remains useful as an early random-initialized throughput prototype. It should not define the long-term model identity or safetensors format.
