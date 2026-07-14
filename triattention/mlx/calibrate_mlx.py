"""
TriAttention MLX Calibration — Production-Grade
================================================
Hooks into mlx-lm attention layers to capture POST-RoPE Q activations,
inverts RoPE (half-split / NeoX convention, partial-rotary aware), and
computes per-head complex frequency statistics.

The math mirrors the canonical CUDA pipeline in ``scripts/calibrate.py``:
  1. Hook computes q_proj → (optional) q_norm → apply RoPE → capture q_rot.
  2. ``compute_stats`` inverts RoPE on q_rot, forms half-split complex pairs
     over the rotated dimensions only, and reduces to per-frequency vectors:
       - q_mean_real / q_mean_imag : [freq_count]  (complex mean)
       - q_abs_mean                : [freq_count]  (|q|.mean, NOT scalar)

``freq_count = int(partial_rotary_factor * head_dim) // 2``.  For Qwen3.5/6
with partial_rotary_factor=0.25 and head_dim=256 → 32 frequencies.

Output: .npz with per-head keys ``l_{layer}_h_{head}_{q_mean_real,q_mean_imag,q_abs_mean}``
plus global metadata (partial_rotary_factor, rope_theta, rope_style, ...).

Architecture-aware hooks for:
  - Qwen3NextAttention (Qwen3.5/6, M-RoPE, partial rotary, gate)
  - GemmaAttention (Gemma 4, standard RoPE)

Usage:
    python calibrate_mlx.py \\
        --model froggeric/qwen3.6-27b-mlx-4bit \\
        --output triattention/calibration/qwen3.6_27b_stats.npz \\
        --samples 128

Contributed by: @cmfontes / Zero — June 2026
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# RoPE helpers (half-split / NeoX convention, partial-rotary aware)
# ═══════════════════════════════════════════════════════════════════════════════

ROPE_STYLE = "half"
"""Half-split (NeoX) convention — matches the consumer's ``invert_rope_mlx``."""


def _rotated_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """Number of leading dimensions rotated by RoPE."""
    return max(2, int(partial_rotary_factor * head_dim))


def build_inv_freq(
    head_dim: int, rope_theta: float, partial_rotary_factor: float
) -> mx.array:
    """Build inverse frequencies for RoPE [freq_count].

    ``freq_count = rotated_dim // 2`` where
    ``rotated_dim = int(partial_rotary_factor * head_dim)``.
    """
    rotated_dim = _rotated_dim(head_dim, partial_rotary_factor)
    freq_count = rotated_dim // 2
    i = mx.arange(0, freq_count, dtype=mx.float32)
    return 1.0 / (rope_theta ** (2 * i / rotated_dim))


def _apply_rope(
    q: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    rotated_dim: int,
) -> mx.array:
    """Apply half-split RoPE to the first ``rotated_dim`` dims of ``q``.

    q: [seq_len, ..., head_dim]  (positions broadcast on axis 0)
    positions: [seq_len]
    inv_freq: [freq_count]

    Returns: q_rot [seq_len, ..., head_dim] (non-rotated tail passes through)
    """
    freq_count = rotated_dim // 2
    theta = mx.outer(positions.astype(mx.float32), inv_freq.astype(mx.float32))
    cos_t = mx.cos(theta)  # [seq_len, freq_count]
    sin_t = mx.sin(theta)

    # Broadcast cos/sin to q's shape: [seq_len, 1, ..., 1, freq_count]
    cos_b = cos_t
    sin_b = sin_t
    for _ in range(q.ndim - 2):
        cos_b = cos_b[:, None]
        sin_b = sin_b[:, None]

    q1 = q[..., :freq_count]               # [..., freq_count]
    q2 = q[..., freq_count:rotated_dim]     # [..., freq_count]

    q_rot1 = q1 * cos_b - q2 * sin_b
    q_rot2 = q1 * sin_b + q2 * cos_b

    if rotated_dim < q.shape[-1]:
        return mx.concatenate([q_rot1, q_rot2, q[..., rotated_dim:]], axis=-1)
    return mx.concatenate([q_rot1, q_rot2], axis=-1)


def _invert_rope(
    q_rot: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    rotated_dim: int,
) -> mx.array:
    """Invert half-split RoPE on the first ``rotated_dim`` dims.

    Inverse of ``_apply_rope``: q_base = invert(apply(q)) == q.
    Non-rotated tail passes through unchanged.
    """
    freq_count = rotated_dim // 2
    theta = mx.outer(positions.astype(mx.float32), inv_freq.astype(mx.float32))
    cos_t = mx.cos(theta)  # [seq_len, freq_count]
    sin_t = mx.sin(theta)

    cos_b = cos_t
    sin_b = sin_t
    for _ in range(q_rot.ndim - 2):
        cos_b = cos_b[:, None]
        sin_b = sin_b[:, None]

    q1 = q_rot[..., :freq_count]
    q2 = q_rot[..., freq_count:rotated_dim]

    # R^-1 = R^T → [cos, sin; -sin, cos]
    q1_orig = q1 * cos_b + q2 * sin_b
    q2_orig = -q1 * sin_b + q2 * cos_b

    if rotated_dim < q_rot.shape[-1]:
        return mx.concatenate([q1_orig, q2_orig, q_rot[..., rotated_dim:]], axis=-1)
    return mx.concatenate([q1_orig, q2_orig], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

LayerHeadKey = Tuple[int, int]  # (layer_idx, head_idx)


class CapturedState:
    """Per-layer accumulator for POST-RoPE Q states and their positions."""
    __slots__ = ("layer_idx", "q_rot_samples", "positions_samples")

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.q_rot_samples: List[mx.array] = []    # list of [seq_len, nheads, head_dim]
        self.positions_samples: List[mx.array] = []  # list of [seq_len]


class StatsAccumulator:
    """Accumulates POST-RoPE Q states across calibration samples and computes
    rotation-removed per-frequency statistics."""

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        rope_theta: float,
        partial_rotary_factor: float,
        rope_style: str = ROPE_STYLE,
    ):
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_style = rope_style
        self.rotated_dim = _rotated_dim(head_dim, partial_rotary_factor)
        self.freq_count = self.rotated_dim // 2
        self.inv_freq = build_inv_freq(head_dim, rope_theta, partial_rotary_factor)
        self._captures: Dict[int, CapturedState] = {
            i: CapturedState(i) for i in range(num_layers)
        }

    def capture(self, layer_idx: int, q_rot: mx.array, positions: mx.array):
        """Capture POST-RoPE Q activations for one attention call.

        q_rot: [seq_len, num_heads, head_dim]
        positions: [seq_len]
        """
        cs = self._captures[layer_idx]
        cs.q_rot_samples.append(q_rot.astype(mx.float32))
        cs.positions_samples.append(positions)

    def compute_stats(self) -> Dict[LayerHeadKey, Dict[str, np.ndarray]]:
        """Invert RoPE and compute per-frequency statistics.

        Returns dict mapping (layer, head) → {
            "q_mean_real": np.float32[freq_count],
            "q_mean_imag": np.float32[freq_count],
            "q_abs_mean":  np.float32[freq_count],   # per-frequency, NOT scalar
        }
        """
        stats: Dict[LayerHeadKey, Dict[str, np.ndarray]] = {}

        for layer_idx, cs in self._captures.items():
            if not cs.q_rot_samples:
                continue

            # Invert RoPE per-sample (each has its own positions), then concat.
            q_base_chunks: List[mx.array] = []
            for q_rot, positions in zip(cs.q_rot_samples, cs.positions_samples):
                q_base = _invert_rope(
                    q_rot, positions, self.inv_freq, self.rotated_dim
                )
                q_base_chunks.append(q_base)

            all_q = mx.concatenate(q_base_chunks, axis=0)  # [T, nheads, head_dim]
            T, num_heads, head_dim = all_q.shape

            for head_idx in range(num_heads):
                q_head = all_q[:, head_idx, :]  # [T, head_dim]

                # Half-split complex pairs on rotated dims only.
                q_rotated = q_head[:, :self.rotated_dim]  # [T, rotated_dim]
                real = q_rotated[:, :self.freq_count].astype(mx.float32)  # [T, freq_count]
                imag = q_rotated[:, self.freq_count:].astype(mx.float32)  # [T, freq_count]

                # Per-frequency complex mean: [freq_count]
                q_mean_real = real.mean(axis=0)
                q_mean_imag = imag.mean(axis=0)

                # Per-frequency |q| mean: [freq_count]  (NOT a scalar!)
                q_abs = mx.sqrt(real ** 2 + imag ** 2 + 1e-12)  # [T, freq_count]
                q_abs_mean = q_abs.mean(axis=0)  # [freq_count]

                stats[(layer_idx, head_idx)] = {
                    "q_mean_real": np.array(q_mean_real, dtype=np.float32),
                    "q_mean_imag": np.array(q_mean_imag, dtype=np.float32),
                    "q_abs_mean": np.array(q_abs_mean, dtype=np.float32),
                }

            # Free memory
            cs.q_rot_samples.clear()
            cs.positions_samples.clear()

        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Attention hooks
# ═══════════════════════════════════════════════════════════════════════════════

_HOOK_TABLE: Dict[int, "AttentionCalibrationHook"] = {}
"""Global dispatch: id(module) → hook instance."""

# Saved original __call__ for the patched class (set once by first hook).
_SAVED_CLASS_CALL = None
_PATCHED_CLASS = None


class AttentionCalibrationHook:
    """
    Hooks Qwen3NextAttention to capture POST-RoPE Q activations.

    Uses a class-level interceptor installed once, dispatching to
    per-instance hooks via _HOOK_TABLE[id(module)].  Safe to unhook
    individually without affecting sibling hooks.

    In the hook we replicate the model's Q path: q_proj → (optional) q_norm
    → apply RoPE (half-split, partial-rotary aware) → capture q_rot.
    ``compute_stats`` later inverts RoPE to recover rotation-removed stats,
    matching the canonical CUDA pipeline and the MLX consumer.
    """

    def __init__(
        self,
        module,
        layer_idx: int,
        accumulator: StatsAccumulator,
        head_dim: int,
        num_heads: int,
    ):
        self._module = module
        self._module_id = id(module)
        self._layer_idx = layer_idx
        self._accumulator = accumulator
        self._head_dim = head_dim
        self._num_heads = num_heads
        self._hooked = False

    def hook(self):
        """Register in dispatch table. Patches class __call__ on first use."""
        global _SAVED_CLASS_CALL, _PATCHED_CLASS
        if self._hooked:
            return
        cls = type(self._module)
        if _PATCHED_CLASS is None:
            _SAVED_CLASS_CALL = cls.__call__
            _PATCHED_CLASS = cls
            cls.__call__ = _shared_attention_interceptor
        elif cls is not _PATCHED_CLASS:
            print(f"  [Warning] Hook type mismatch: expected {_PATCHED_CLASS.__name__}, got {cls.__name__}")
            return
        _HOOK_TABLE[self._module_id] = self
        self._hooked = True

    def unhook(self):
        """Remove from dispatch table. Restores class __call__ when table empty."""
        global _SAVED_CLASS_CALL, _PATCHED_CLASS
        if not self._hooked:
            return
        _HOOK_TABLE.pop(self._module_id, None)
        self._hooked = False
        # Restore original __call__ when no hooks remain.
        if not _HOOK_TABLE and _PATCHED_CLASS is not None and _SAVED_CLASS_CALL is not None:
            _PATCHED_CLASS.__call__ = _SAVED_CLASS_CALL
            _PATCHED_CLASS = None
            _SAVED_CLASS_CALL = None


def _shared_attention_interceptor(self_attn, x: mx.array, mask=None, cache=None):
    """Class-level interceptor — dispatches by Python object id.

    Captures POST-RoPE Q by replicating the model's Q path:
      q_proj → split (gate) → reshape → q_norm (if present) → RoPE → capture.
    """
    hook = _HOOK_TABLE.get(id(self_attn))
    if hook is not None:
        acc = hook._accumulator
        bsz, q_len, _ = x.shape
        nheads = hook._num_heads
        hdim = hook._head_dim

        q_proj_output = self_attn.q_proj(x)
        # Qwen3Next splits q_proj into queries + gate.
        queries, _gate = mx.split(
            q_proj_output.reshape(bsz, q_len, nheads, -1), 2, axis=-1
        )
        # queries: [bsz, q_len, nheads, hdim]

        # Apply q_norm if the attention layer has one (Qwen3 RMSNorm per-head).
        q_norm = getattr(self_attn, "q_norm", None)
        if q_norm is not None:
            try:
                queries = q_norm(queries)
            except Exception:
                # q_norm may expect a different shape; try per-head flatten.
                b, s, h, d = queries.shape
                queries = q_norm(queries.reshape(b * h, s, d)).reshape(b, s, h, d)

        # Build positions from cache offset (or zero).
        offset = cache.offset if cache is not None else 0
        positions = mx.arange(offset, offset + q_len, dtype=mx.int32)

        # Reshape to [seq_len, nheads, head_dim] for RoPE (bsz=1 in calibration).
        q_flat = queries.reshape(bsz, q_len, nheads, hdim)
        if bsz != 1:
            # Replicate positions for each batch row and flatten.
            q_flat = q_flat.reshape(bsz * q_len, nheads, hdim)
            positions = mx.broadcast_to(
                positions[None, :], (bsz, q_len)
            ).reshape(-1)
        else:
            q_flat = q_flat.reshape(q_len, nheads, hdim)

        # Apply RoPE (half-split, partial-rotary aware) → capture POST-RoPE Q.
        q_rot = _apply_rope(q_flat, positions, acc.inv_freq, acc.rotated_dim)
        hook._accumulator.capture(hook._layer_idx, q_rot, positions)

    return _SAVED_CLASS_CALL(self_attn, x, mask=mask, cache=cache)


# ═══════════════════════════════════════════════════════════════════════════════
# Model introspection
# ═══════════════════════════════════════════════════════════════════════════════

def _find_attention_layers(model) -> List[Tuple[int, object, str]]:
    """
    Discover full-attention layers across supported architectures.

    Returns: List[(layer_idx, module, architecture_tag)]
        architecture_tag ∈ {"qwen3next", "gemma"}
    """
    results: List[Tuple[int, object, str]] = []

    for name, mod in model.named_modules():
        type_name = type(mod).__name__
        if type_name not in ("Qwen3NextAttention", "GemmaAttention"):
            continue
        # Extract layer index from e.g. "language_model.model.layers.42.self_attn"
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    continue
                break
        if layer_idx is None:
            continue
        tag = "qwen3next" if type_name == "Qwen3NextAttention" else "gemma"
        results.append((layer_idx, mod, tag))

    results.sort(key=lambda x: x[0])
    return results


def _detect_arch_config(model) -> dict:
    """Extract architecture-specific configuration."""
    cfg = {}
    try:
        args = model.args
        tc = getattr(args, "text_config", {}) or {}
        get = lambda key, default: (
            tc.get(key, getattr(args, key, default))
            if isinstance(tc, dict)
            else getattr(args, key, default)
        )
        cfg["model_type"] = str(get("model_type", "unknown"))
        cfg["head_dim"] = int(get("head_dim", 128))
        cfg["num_attention_heads"] = int(get("num_attention_heads", 8))
        cfg["num_hidden_layers"] = int(get("num_hidden_layers", 32))
        rope_params = get("rope_parameters", {}) or {}
        cfg["rope_theta"] = float(
            rope_params.get("rope_theta", get("rope_theta", 10000.0))
            if isinstance(rope_params, dict)
            else get("rope_theta", 10000.0)
        )
        cfg["partial_rotary_factor"] = float(get("partial_rotary_factor", 1.0))
        cfg["mrope_interleaved"] = bool(
            rope_params.get("mrope_interleaved", False)
            if isinstance(rope_params, dict)
            else False
        )
        if isinstance(rope_params, dict) and rope_params.get("mrope_section"):
            cfg["mrope_section"] = rope_params["mrope_section"]
    except Exception as e:
        print(f"[Warning] Config detection incomplete: {e}")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration prompts
# ═══════════════════════════════════════════════════════════════════════════════

CALIBRATION_PROMPTS: List[str] = [
    "Solve step by step: What is the integral of x^2 * sin(x) dx?",
    "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.",
    "Explain the difference between TCP and UDP protocols with concrete examples.",
    "A train leaves City A at 9 AM traveling at 60 mph. Another train leaves City B (300 miles away) at 11 AM traveling at 80 mph toward City A. When do they meet?",
    "Write a comprehensive security audit checklist for a REST API.",
    "Explain how transformer attention works mathematically, step by step.",
    "Debug this code: def fib(n): return fib(n-1)+fib(n-2) if n>1 else n",
    "What are the key differences between Proof of Work and Proof of Stake?",
    "Design a distributed system for handling 1M concurrent WebSocket connections.",
    "Prove that the square root of 2 is irrational.",
    "Implement a thread-safe LRU cache in Python with O(1) operations.",
    "Explain the CAP theorem and its implications for real-world database choices.",
    "What is backpropagation? Derive the gradient update rules for a 2-layer neural network.",
    "Compare and contrast B-trees, LSM trees, and their use in modern databases.",
    "Write a SQL query to find the top 3 employees by salary in each department.",
    "Explain how HTTPS works: TLS handshake, certificate validation, session keys.",
    "What is a Bloom filter? Derive the false positive probability formula.",
    "Design a rate limiter that handles 10M requests per second across 1000 servers.",
    "Explain the Linux kernel's OOM killer: how it selects victims and what triggers it.",
    "Write a simple regex engine that supports ., *, and ^. Explain each part.",
    "What is the halting problem and why is it undecidable? Give an intuitive proof.",
    "Compare gRPC vs REST vs GraphQL for microservice communication.",
    "Explain SIMD vectorization and how it speeds up numerical computation.",
    "Design a chat application backend that supports 100K concurrent users.",
    "What is a Merkle tree? Explain its role in Bitcoin and Git.",
    "Write a lock-free concurrent queue in Python. What guarantees does it provide?",
    "Explain Kubernetes architecture: control plane, nodes, pods, and the reconciliation loop.",
    "What is the P vs NP problem? Explain using the traveling salesman example.",
    "Design a recommendation system for a streaming service with 50M users.",
    "Explain how garbage collection works in V8 (JavaScript engine).",
    "What is a double-spend attack on a blockchain? How is it prevented?",
    "Implement a minimal HTTP/1.1 server in Python that handles concurrent connections.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main calibration pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(
    model,
    tokenizer,
    prompts: List[str],
    output_path: str,
    num_samples: int = 128,
    max_tokens: int = 64,
) -> bool:
    """
    Run the full calibration pipeline.

    Args:
        model: mlx-lm model instance
        tokenizer: corresponding tokenizer
        prompts: calibration text prompts
        output_path: where to save the .npz file
        num_samples: number of prompts to process
        max_tokens: max generation tokens per sample (smaller = faster calibration)

    Returns:
        True on success, False on failure
    """
    import mlx_lm

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Detect architecture ──
    arch = _detect_arch_config(model)
    head_dim = arch.get("head_dim", 256)
    num_heads = arch.get("num_attention_heads", 24)
    num_layers = arch.get("num_hidden_layers", 64)
    rope_theta = arch.get("rope_theta", 10000.0)
    partial_rotary_factor = arch.get("partial_rotary_factor", 1.0)

    rotated_dim = _rotated_dim(head_dim, partial_rotary_factor)
    freq_count = rotated_dim // 2

    print(f"[Calibrate] Model: {arch.get('model_type', 'unknown')}")
    print(f"[Calibrate] Heads: {num_heads} × head_dim={head_dim}")
    print(f"[Calibrate] RoPE θ: {rope_theta}")
    print(f"[Calibrate] Partial rotary: {partial_rotary_factor} "
          f"→ rotated_dim={rotated_dim}, freq_count={freq_count}")
    print(f"[Calibrate] RoPE style: {ROPE_STYLE}")
    print(f"[Calibrate] M-RoPE: {arch.get('mrope_interleaved', False)}")

    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        print("[Calibrate] ERROR: No full-attention layers found. Aborting.")
        return False

    print(f"[Calibrate] Found {len(attn_layers)} full-attention layers: "
          f"{[l for l, _, _ in attn_layers]}")
    print(f"[Calibrate] Samples: {num_samples}")

    # ── Setup accumulator ──
    accumulator = StatsAccumulator(
        num_layers=num_layers,
        head_dim=head_dim,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        rope_style=ROPE_STYLE,
    )

    # ── Install hooks ──
    hooks: List[AttentionCalibrationHook] = []
    for layer_idx, module, tag in attn_layers:
        if tag == "qwen3next":
            hook = AttentionCalibrationHook(
                module,
                layer_idx=layer_idx,
                accumulator=accumulator,
                head_dim=head_dim,
                num_heads=num_heads,
            )
            hook.hook()
            hooks.append(hook)
        else:
            print(f"[Calibrate] WARNING: Unsupported attention type '{tag}' "
                  f"in layer {layer_idx} — skipping")

    if not hooks:
        print("[Calibrate] ERROR: No attention hooks could be installed. Aborting.")
        return False

    print(f"[Calibrate] Installed {len(hooks)} hooks.")

    # ── Run calibration passes ──
    try:
        sample_count = 0
        cycles = max(1, (num_samples + len(prompts) - 1) // len(prompts))

        for cycle in range(cycles):
            for i, prompt in enumerate(prompts):
                if sample_count >= num_samples:
                    break

                try:
                    tokens = tokenizer.encode(prompt)
                    if len(tokens) > 2048:
                        tokens = tokens[:2048]

                    _ = mlx_lm.generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=min(max_tokens, 64),
                        verbose=False,
                    )
                    sample_count += 1
                    if sample_count % 10 == 0:
                        print(f"  [{sample_count}/{num_samples}] samples processed")
                except Exception as e:
                    print(f"  [Warning] Sample {sample_count + 1} failed: {e}")

    finally:
        # ── Always unhook ──
        for hook in hooks:
            hook.unhook()
        print("[Calibrate] Hooks removed.")

    print(f"[Calibrate] Processed {sample_count} samples.")

    # ── Compute and save stats ──
    stats = accumulator.compute_stats()
    if not stats:
        print("[Calibrate] ERROR: No stats computed. Hooks may not have fired.")
        return False

    n_entries = len(stats)
    print(f"[Calibrate] Computed stats for {n_entries} head-layer pairs.")

    # ── Build npz payload ──
    # Metadata (global)
    np_stats: dict = {
        "model_type": np.array(arch.get("model_type", "unknown")),
        "calibrated": np.array([True], dtype=np.bool_),
        "num_attention_heads": np.array([num_heads], dtype=np.int32),
        "head_dim": np.array([head_dim], dtype=np.int32),
        "rope_theta": np.array([rope_theta], dtype=np.float32),
        "partial_rotary_factor": np.array([partial_rotary_factor], dtype=np.float32),
        "rotated_dim": np.array([rotated_dim], dtype=np.int32),
        "freq_count": np.array([freq_count], dtype=np.int32),
        "rope_style": np.array(ROPE_STYLE),
        "num_samples": np.array([sample_count], dtype=np.int32),
    }

    # Per-head stats — each array is [freq_count], NOT scalar.
    for (layer_idx, head_idx), head_stats in stats.items():
        prefix = f"l_{layer_idx}_h_{head_idx}"
        np_stats[f"{prefix}_q_mean_real"] = head_stats["q_mean_real"]
        np_stats[f"{prefix}_q_mean_imag"] = head_stats["q_mean_imag"]
        np_stats[f"{prefix}_q_abs_mean"] = head_stats["q_abs_mean"]

    np.savez_compressed(str(output), **np_stats)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"[Calibrate] Saved {n_entries} entries → {output} ({size_mb:.1f} MB)")
    print(f"[Calibrate]   freq_count={freq_count}, "
          f"partial_rotary_factor={partial_rotary_factor}, "
          f"rope_style={ROPE_STYLE}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TriAttention MLX Calibration — Generate frequency stats for any mlx-lm model",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model ID or local path (e.g., froggeric/qwen3.6-27b-mlx-4bit)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz stats file path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max generation tokens per sample (default: 64)",
    )
    args = parser.parse_args()

    try:
        import mlx_lm
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        return 1

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)

    success = calibrate(
        model,
        tokenizer,
        CALIBRATION_PROMPTS,
        args.output,
        num_samples=args.samples,
        max_tokens=args.max_tokens,
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
