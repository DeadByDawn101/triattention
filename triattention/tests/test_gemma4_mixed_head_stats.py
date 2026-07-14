from pathlib import Path

import pytest
import torch

from triattention.sglang.stats_loader import (
    load_stats,
    validate_stats_against_model,
)
from triattention.vllm.core.compressor import TriAttentionCompressor
from triattention.vllm.core.config import TriAttentionConfig
from triattention.vllm.core.scoring import compute_scores_pytorch
from triattention.vllm.core.utils import compute_rope_frequencies, load_frequency_stats


def _gemma4_like_payload() -> dict:
    stats = {}
    for layer_idx, freq_count in [(0, 128), (1, 256)]:
        for head_idx in range(4):
            value = float(layer_idx + head_idx + 1)
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.full((freq_count,), value),
                "q_mean_imag": torch.full((freq_count,), value + 0.5),
                "q_abs_mean": torch.full((freq_count,), value + 1.0),
            }
    return {
        "metadata": {
            "head_dim": 512,
            "layer_head_dims": [256, 512],
            "num_key_value_heads": 2,
            "rope_style": "half",
            "rope_theta": 10000.0,
        },
        "stats": stats,
    }


def test_vllm_rkv_loader_handles_gemma4_mixed_head_dims(tmp_path: Path) -> None:
    path = tmp_path / "gemma4_stats.pt"
    torch.save(_gemma4_like_payload(), path)

    metadata, head_stats = load_frequency_stats(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert metadata["head_dim"] == 512
    assert metadata["num_kv_heads"] == 2
    assert metadata["layer_head_dims"] == [256, 512]
    assert metadata["layer_freq_counts"] == [128, 256]

    assert head_stats[0]["q_mean_complex"].shape == (2, 128, 2)
    assert head_stats[1]["q_mean_complex"].shape == (2, 256, 2)
    assert head_stats[0]["q_abs_mean"].shape == (2, 128)
    assert head_stats[1]["q_abs_mean"].shape == (2, 256)


def test_sglang_rkv_loader_handles_gemma4_mixed_head_dims(tmp_path: Path) -> None:
    path = tmp_path / "gemma4_stats.pt"
    torch.save(_gemma4_like_payload(), path)

    bundle = load_stats(
        str(path),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=2,
    )

    assert bundle.head_dim == 512
    assert bundle.num_kv_heads == 2
    assert bundle.num_attention_heads == 4
    assert bundle.gqa_group_size == 2
    assert bundle.layer_freq_counts == [128, 256]

    assert bundle.head_stats[0]["q_mean_complex"].shape == (4, 128, 2)
    assert bundle.head_stats[1]["q_mean_complex"].shape == (4, 256, 2)
    assert bundle.head_stats[0]["q_abs_mean"].shape == (4, 128)
    assert bundle.head_stats[1]["q_abs_mean"].shape == (4, 256)

    validate_stats_against_model(
        bundle,
        model_num_layers=2,
        model_num_kv_heads=2,
        model_head_dim=256,
    )
    validate_stats_against_model(
        bundle,
        model_num_layers=2,
        model_num_kv_heads=2,
        model_head_dim=512,
    )


def test_vllm_mixed_head_scoring_uses_native_layer_frequency_basis(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gemma4_stats.pt"
    torch.save(_gemma4_like_payload(), path)

    config = TriAttentionConfig(
        stats_path=path,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        use_triton_scoring=False,
        use_trig_cache=False,
        disable_mlr=True,
        kv_budget=2,
        window_size=0,
        offset_max_length=2,
    )
    compressor = TriAttentionCompressor(config)
    compressor._lazy_init()

    native_256 = compute_rope_frequencies(256, 10000.0, device=torch.device("cpu"))
    sliced_512 = compute_rope_frequencies(512, 10000.0, device=torch.device("cpu"))[
        :128
    ]
    assert torch.allclose(compressor.get_layer_omega(0), native_256)
    assert not torch.allclose(native_256[64:], sliced_512[64:])

    torch.manual_seed(0)
    key_states = torch.randn(1, 2, 5, 256)
    scores = compressor._compute_scores(key_states=key_states, layer_idx=0)
    reference = compute_scores_pytorch(
        key_states=key_states,
        cache_positions=None,
        head_stats=compressor.head_stats[0],
        omega=native_256,
        offsets=compressor.offsets,
        freq_scale_sq=compressor.get_layer_freq_scale_sq(0),
        config=compressor.config,
        round_start=compressor.state.get_round_start(),
    )
    wrong_sliced_basis = compute_scores_pytorch(
        key_states=key_states,
        cache_positions=None,
        head_stats=compressor.head_stats[0],
        omega=sliced_512,
        offsets=compressor.offsets,
        freq_scale_sq=compressor.get_layer_freq_scale_sq(0),
        config=compressor.config,
        round_start=compressor.state.get_round_start(),
    )

    assert torch.allclose(scores, reference)
    assert not torch.allclose(scores, wrong_sliced_basis)


def test_sglang_validation_checks_all_mixed_head_layers(tmp_path: Path) -> None:
    path = tmp_path / "gemma4_stats.pt"
    torch.save(_gemma4_like_payload(), path)

    bundle = load_stats(
        str(path),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=2,
    )
    bundle.head_stats[1]["freq_scale_sq"] = torch.ones(128)

    with pytest.raises(ValueError, match="layer 1 freq_scale_sq"):
        validate_stats_against_model(
            bundle,
            model_num_layers=2,
            model_num_kv_heads=2,
            model_head_dim=512,
        )
