# TriAttention MLX Calibration — Apple Silicon

Contributed by [@cmfontes](https://github.com/fattchris) (Zero) — June 2026

## What This Is

A calibration pipeline for generating TriAttention frequency statistics on Apple Silicon using MLX. Produces per-head RoPE-aware trig scoring stats for full-attention layers, enabling the same `kv_budget` compression quality on Metal API that the CUDA pipeline achieves on NVIDIA GPUs.

## Usage

```bash
python triattention/mlx/calibrate_mlx.py \
    --model froggeric/qwen3.6-27b-mlx-4bit \
    --output triattention/calibration/my_model_stats.npz \
    --samples 128
```

## Architecture Support

| Architecture | Attention Type | Status |
|---|---|---|
| Qwen3NextAttention (Qwen3.5/6) | M‑RoPE, partial rotary (0.25), gate | ✅ Supported |
| GemmaAttention (Gemma 4) | Standard RoPE | 🟡 Detected, hooks TBD |

## Key Design Decisions

1. **Class-level dispatch with global table** — MLX has no `register_forward_pre_hook`. We install a single class-level interceptor on `Qwen3NextAttention.__call__` and dispatch to per-instance hooks via `_HOOK_TABLE[id(module)]`. A singleton pattern ensures the original `__call__` is saved exactly once on first hook installation, avoiding overwrites across multiple layers.

2. **Q capture at projection point** — We tap `q_proj()` output (split into queries/gate) before `q_norm` and RoPE, capturing the full `[nheads, head_dim]` activation for all 16 full-attention layers per forward pass.

3. **Stats format** — Output `.npz` uses the same `l_{layer}_h_{head}_q_mean_real|imag|_abs_mean` keys as the existing MLX `triattention_mlx.py` `_load_frequency_stats` function, ensuring drop-in compatibility.

## Known Limitation: Qwen3.6 DeltaNet Layers

Qwen3.6 uses GatedDeltaNet linear attention for 48 of 64 layers (layers 0–2, 4–6, 8–10, etc.) with only 16 full-attention layers (3, 7, 11, …, 63). TriAttention compresses only the full-attention layers, so the majority of the model's KV budget is unaffected. Benchmarks at 4.8× compression showed identical output between norm‑only and trig‑scoring — the DeltaNet layers dominate the signal path. For models with standard attention across all layers (Llama, Mistral, Gemma), the trig‑vs‑norm divergence is expected to be more pronounced.

## PR Contents

- `triattention/mlx/calibrate_mlx.py` — Full calibration pipeline with Qwen3NextAttention hooks
- `triattention/calibration/qwen3.6_27b_stats.npz` — Pre‑generated 384‑head stats for Qwen3.6-27B MLX 4‑bit
- `docs/mlx-calibration.md` — This document

## Attempted: Qwable-3.6-27b

We attempted to calibrate [Qwable-3.6-27b](https://huggingface.co/Mia-AiLab/Qwable-3.6-27b) — a full fine‑tune of Qwen3.6‑27B on Fable‑5 reasoning traces. The model uses the same `Qwen3NextAttention` architecture, so the hooks apply. The blocker was **not** the architecture — it was the MLX conversion step.

### What Worked

- `froggeric/qwen3.6-27b-mlx-4bit` (pre‑converted 4‑bit MLX) loaded and calibrated in 12 minutes on an M4 Pro (24 GB)
- Calibration generated valid 384‑head stats matching the existing `_load_frequency_stats` format
- Full trig‑scoring generation succeeded at 4.8× KV compression on Apple Silicon

### What Failed

- Converting Qwable from HF safetensors to MLX format requires loading the full fp16 model (~54 GB), which won't fit in 24 GB RAM
- `mlx_lm.convert` loads the entire model into memory before quantizing — no streaming conversion path exists
- The `HF_HUB_ENABLE_HF_TRANSFER=1` workaround only helps with download speed, not memory
- Cloud conversion (128 GB instance) would work but wasn't scoped for this PR

### Recommendation

Future work should either:
1. Request MLX‑native quantized uploads from model publishers (like froggeric's pre‑converted versions)
2. Use a cloud instance for the single‑pass conversion → upload the resulting MLX model to HF
3. Add streaming quantization to `mlx_lm.convert` (out of scope for TriAttention)
