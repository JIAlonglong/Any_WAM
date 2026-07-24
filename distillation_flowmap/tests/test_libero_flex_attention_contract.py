import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WAN_VA_ROOT = ROOT / "wan_va"


def test_flex_attention_exposes_packed_length_and_padding_contract():
    sys.path.insert(0, str(WAN_VA_ROOT))
    try:
        from modules.model import FlexAttnFunc

        total = FlexAttnFunc._packed_sequence_length(
            (1, 48, 16, 8, 8),
            (1, 30, 16, 4, 1),
            (1, 2, 2),
        )
        assert total == 2 * (16 * 4 * 4 + 16 * 4)
        assert FlexAttnFunc.resolve_padded_length(total, attn_mode="torch") == (
            128 - total % 128
        ) % 128
    finally:
        sys.path.pop(0)


def test_mask_cache_key_distinguishes_action_geometries():
    sys.path.insert(0, str(WAN_VA_ROOT))
    try:
        from modules.model import FlexAttnFunc

        common = dict(
            latent_shape=(1, 48, 16, 8, 8),
            padded_length=0,
            chunk_size=2,
            window_size=72,
            patch_size=(1, 2, 2),
            device="cpu",
            fdm=False,
        )
        full = FlexAttnFunc._mask_cache_key(
            action_shape=(1, 30, 16, 4, 1), **common
        )
        downsampled = FlexAttnFunc._mask_cache_key(
            action_shape=(1, 30, 4, 4, 1), **common
        )
        assert full != downsampled
    finally:
        sys.path.pop(0)
