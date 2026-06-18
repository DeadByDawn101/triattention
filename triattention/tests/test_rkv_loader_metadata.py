from pathlib import Path

import torch

from triattention.vllm.core.utils import load_frequency_stats


def test_vllm_rkv_loader_uses_num_key_value_heads_metadata(tmp_path: Path) -> None:
    stats = {}
    for layer_idx in range(1):
        for head_idx in range(4):
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.ones(4),
                "q_mean_imag": torch.zeros(4),
                "q_abs_mean": torch.ones(4),
            }
    path = tmp_path / "rkv_stats.pt"
    torch.save(
        {
            "metadata": {
                "head_dim": 8,
                "num_key_value_heads": 2,
                "rope_style": "half",
                "rope_theta": 10000.0,
            },
            "stats": stats,
        },
        path,
    )

    metadata, head_stats = load_frequency_stats(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert metadata["num_attention_heads"] == 4
    assert metadata["num_kv_heads"] == 2
    assert metadata["gqa_ratio"] == 2
    assert head_stats[0]["q_mean_complex"].shape == (2, 4, 2)
    assert head_stats[0]["q_abs_mean"].shape == (2, 4)
