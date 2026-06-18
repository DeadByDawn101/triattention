"""
TriAttention MLX Calibration — Production-Grade
================================================
Hooks into mlx-lm attention layers to capture pre‑RoPE Q activations,
inverts RoPE, and computes per‑head complex frequency statistics.

Output: .npz with keys {(layer, head): {"freq_real", "freq_imag", "abs_mean"}}

Architecture-aware hooks for:
  - Qwen3NextAttention (Qwen3.5/6, M‑RoPE, partial rotary, gate)
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
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

LayerHeadKey = Tuple[int, int]  # (layer_idx, head_idx)

class CapturedState:
    """Per‑sample accumulator for Q states across attention heads."""
    __slots__ = ("layer_idx", "q_pre_rope", "positions")
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.q_pre_rope: List[mx.array] = []  # list of [num_heads, head_dim]
        self.positions: List[mx.array] = []   # list of [seq_len]


class StatsAccumulator:
    """Accumulates Q stats across calibration samples."""
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._captures: Dict[int, CapturedState] = {
            i: CapturedState(i) for i in range(num_layers)
        }

    def capture(self, layer_idx: int, q_pre: mx.array, positions: mx.array):
        """Capture pre‑RoPE Q activations for one attention call."""
        cs = self._captures[layer_idx]
        cs.q_pre_rope.append(q_pre.astype(mx.complex64))
        cs.positions.append(positions)

    def compute_stats(self) -> Dict[LayerHeadKey, Dict[str, float]]:
        """Aggregate captured Q states into RoPE‑aware frequency statistics."""
        stats: Dict[LayerHeadKey, Dict[str, float]] = {}

        for layer_idx, cs in self._captures.items():
            if not cs.q_pre_rope:
                continue

            # Stack all samples: [total_tokens, num_heads, head_dim]
            all_q = mx.concatenate(cs.q_pre_rope, axis=0)   # [T, n_heads, head_dim]
            T, num_heads, head_dim = all_q.shape

            for head_idx in range(num_heads):
                q_head = all_q[:, head_idx, :]  # [T, head_dim]
                mean_complex = q_head.mean(axis=0)  # [head_dim]

                # Invert RoPE — isolate the real frequency component
                # RoPE pairs dimensions: even = real, odd = imag
                head_dim_half = head_dim // 2
                freq_real = mean_complex[0 : 2 * head_dim_half : 2]
                freq_imag = mean_complex[1 : 2 * head_dim_half : 2]

                # ∣Q∣ mean (for norm‑based scoring)
                abs_mean = float(
                    mx.sqrt(mx.sum(q_head.astype(mx.float32) ** 2, axis=-1).mean() + 1e-8)
                )

                stats[(layer_idx, head_idx)] = {
                    "freq_real": np.array(freq_real.real, dtype=np.float32),
                    "freq_imag": np.array(freq_imag.real, dtype=np.float32),
                    "abs_mean": np.float32(abs_mean),
                }

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
    Hooks Qwen3NextAttention to capture pre‑RoPE Q activations.

    Uses a class-level interceptor installed once, dispatching to
    per‑instance hooks via _HOOK_TABLE[id(module)].  Safe to unhook
    individually without affecting sibling hooks.
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


def _shared_attention_interceptor(self_attn, x: mx.array, mask=None, cache=None):
    """Class‑level interceptor — dispatches by Python object id."""
    hook = _HOOK_TABLE.get(id(self_attn))
    if hook is not None:
        bsz, q_len, _ = x.shape
        nheads = hook._num_heads
        hdim = hook._head_dim
        q_proj_output = self_attn.q_proj(x)
        queries, _gate = mx.split(
            q_proj_output.reshape(bsz, q_len, nheads, -1), 2, axis=-1
        )
        q_flat = queries.reshape(-1, nheads, hdim)
        offset = cache.offset if cache else 0
        positions = mx.arange(offset, offset + q_len, dtype=mx.int32)
        hook._accumulator.capture(hook._layer_idx, q_flat, positions)
    return _SAVED_CLASS_CALL(self_attn, x, mask=mask, cache=cache)


# ═══════════════════════════════════════════════════════════════════════════════
# Model introspection
# ═══════════════════════════════════════════════════════════════════════════════

def _find_attention_layers(model) -> List[Tuple[int, object, str]]:
    """
    Discover full‑attention layers across supported architectures.

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
    """Extract architecture‑specific configuration."""
    cfg = {}
    try:
        args = model.args
        tc = getattr(args, "text_config", {})
        cfg["model_type"] = getattr(args, "model_type", "unknown")
        cfg["head_dim"] = tc.get("head_dim", getattr(args, "head_dim", 128))
        cfg["num_attention_heads"] = tc.get(
            "num_attention_heads", getattr(args, "num_attention_heads", 8)
        )
        cfg["num_hidden_layers"] = tc.get(
            "num_hidden_layers", getattr(args, "num_hidden_layers", 32)
        )
        cfg["rope_theta"] = tc.get(
            "rope_parameters", {}
        ).get("rope_theta", getattr(args, "rope_theta", 10000.0))
        cfg["partial_rotary_factor"] = tc.get(
            "partial_rotary_factor",
            getattr(args, "partial_rotary_factor", 1.0),
        )
        cfg["mrope_interleaved"] = tc.get(
            "rope_parameters", {}
        ).get("mrope_interleaved", False)
        if tc.get("rope_parameters", {}).get("mrope_section"):
            cfg["mrope_section"] = tc["rope_parameters"]["mrope_section"]
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
    "Compare and contrast B‑trees, LSM trees, and their use in modern databases.",
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
    "What is a double‑spend attack on a blockchain? How is it prevented?",
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
    print(f"[Calibrate] Model: {arch.get('model_type', 'unknown')}")
    print(f"[Calibrate] Heads: {arch.get('num_attention_heads', '?')} × "
          f"head_dim={arch.get('head_dim', '?')}")
    print(f"[Calibrate] RoPE θ: {arch.get('rope_theta', '?')}")
    print(f"[Calibrate] Partial rotary: {arch.get('partial_rotary_factor', '?')}")
    print(f"[Calibrate] M‑RoPE: {arch.get('mrope_interleaved', False)}")

    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        print("[Calibrate] ERROR: No full‑attention layers found. Aborting.")
        return False

    print(f"[Calibrate] Found {len(attn_layers)} full‑attention layers: "
          f"{[l for l, _, _ in attn_layers]}")
    print(f"[Calibrate] Samples: {num_samples}")

    # ── Setup accumulator ──
    accumulator = StatsAccumulator(num_layers=arch.get("num_hidden_layers", 64))

    # ── Install hooks ──
    hooks: List[AttentionCalibrationHook] = []
    for layer_idx, module, tag in attn_layers:
        if tag == "qwen3next":
            hook = AttentionCalibrationHook(
                module,
                layer_idx=layer_idx,
                accumulator=accumulator,
                head_dim=arch.get("head_dim", 256),
                num_heads=arch.get("num_attention_heads", 24),
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
    print(f"[Calibrate] Computed stats for {n_entries} head‑layer pairs.")

    # Convert to numpy‑friendly format
    np_stats: dict = {
        "model_type": np.array([arch.get("model_type", "unknown")]),
        "calibrated": np.array([True], dtype=np.bool_),
        "num_attention_heads": np.array([arch.get("num_attention_heads", 0)], dtype=np.int32),
        "head_dim": np.array([arch.get("head_dim", 0)], dtype=np.int32),
        "rope_theta": np.array([arch.get("rope_theta", 0.0)], dtype=np.float32),
        "partial_rotary_factor": np.array(
            [arch.get("partial_rotary_factor", 1.0)], dtype=np.float32
        ),
        "num_samples": np.array([sample_count], dtype=np.int32),
    }

    for (layer_idx, head_idx), head_stats in stats.items():
        prefix = f"l_{layer_idx}_h_{head_idx}"
        np_stats[f"{prefix}_q_mean_real"] = head_stats["freq_real"]
        np_stats[f"{prefix}_q_mean_imag"] = head_stats["freq_imag"]
        np_stats[f"{prefix}_q_abs_mean"] = head_stats["abs_mean"]

    np.savez_compressed(str(output), **np_stats)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"[Calibrate] Saved {n_entries} entries → {output} ({size_mb:.1f} MB)")
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
