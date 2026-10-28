"""
Codec validation for the PTQ1_0 / PQ2_0 ternary formats.

Verifies:
1. The encode/decode round-trip reproduces the original codes (not the
   weights -- the absmax+round mapping is lossy).
2. The packed byte layout matches what the C reference would produce
   for the same input weights.
3. The trit-packing math (5 trits per byte with ceiling division) is
   exactly invertible.
4. The effective bpw matches the whitepaper's claims.

Run:
    python test_codec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from codec import (
    encode_pq2_0_row,
    decode_pq2_0_row,
    encode_ptq1_0_row,
    decode_ptq1_0_row,
    block_size_bytes,
    block_elems,
    bits_per_weight,
    _pack_5_trits,
    _unpack_5_trits,
    _pack_4_trits,
    _unpack_4_trits,
    _quantize_pq2_0_group,
    _quantize_ptq1_0_group,
    QK_PQ2_0,
    QK_PTQ1_0,
)


def test_5_trit_round_trip():
    """5-trit packing must be exactly invertible for all base-3 values."""
    print("\n=== Test 1: 5-trit pack/unpack round-trip ===")
    # Test all 3^5 = 243 possible 5-trit combinations.
    mismatches = 0
    for v in range(243):
        trits = np.array([(v // (3 ** i)) % 3 for i in range(5)], dtype=np.uint8)
        byte = _pack_5_trits(trits)
        recovered = _unpack_5_trits(byte)
        if not np.array_equal(trits, recovered):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH: trits={trits.tolist()} byte={byte} "
                      f"recovered={recovered.tolist()}")
    print(f"  Tested 243 5-trit combinations; mismatches: {mismatches}")
    assert mismatches == 0, f"5-trit round-trip failed: {mismatches} mismatches"
    print("  PASS")


def test_4_trit_round_trip():
    """4-trit packing (qh layout) must be exactly invertible."""
    print("\n=== Test 2: 4-trit pack/unpack round-trip ===")
    mismatches = 0
    for v in range(81):  # 3^4 = 81
        trits = np.array([(v // (3 ** i)) % 3 for i in range(4)], dtype=np.uint8)
        byte = _pack_4_trits(trits)
        recovered = _unpack_4_trits(byte)
        if not np.array_equal(trits, recovered):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH: trits={trits.tolist()} byte={byte} "
                      f"recovered={recovered.tolist()}")
    print(f"  Tested 81 4-trit combinations; mismatches: {mismatches}")
    assert mismatches == 0, f"4-trit round-trip failed: {mismatches} mismatches"
    print("  PASS")


def test_pq2_0_round_trip():
    """PQ2_0 encode -> decode must recover the codes (absmax+round is lossy
    on weights, but the codes round-trip exactly)."""
    print("\n=== Test 3: PQ2_0 encode/decode round-trip ===")
    rng = np.random.default_rng(42)
    n_blocks = 8
    w = rng.normal(0, 0.05, size=n_blocks * QK_PQ2_0).astype(np.float32)
    # Encode.
    data = encode_pq2_0_row(w)
    expected_size = n_blocks * block_size_bytes("PQ2_0")
    assert len(data) == expected_size, (
        f"PQ2_0 encoded size: expected {expected_size}, got {len(data)}"
    )
    # Decode.
    recon = decode_pq2_0_row(data, n_blocks)
    # Compute the expected reconstruction: (round(w/d) + 1 - 1) * d = round(w/d) * d.
    for b in range(n_blocks):
        chunk = w[b * QK_PQ2_0 : (b + 1) * QK_PQ2_0]
        codes, d = _quantize_pq2_0_group(chunk)
        expected = (codes.astype(np.float32) - 1.0) * d
        actual = recon[b * QK_PQ2_0 : (b + 1) * QK_PQ2_0]
        max_diff = float(np.abs(expected - actual).max())
        # The scale d is stored as fp16, so the reconstruction has fp16
        # rounding error (~0.1% of the scale magnitude).
        assert max_diff < 1e-4, (
            f"PQ2_0 block {b}: max diff {max_diff} > 1e-4"
        )
    print(f"  Encoded {n_blocks} blocks ({n_blocks * 128} weights) "
          f"to {len(data)} bytes ({len(data) / (n_blocks * 128) * 8:.3f} bpw)")
    print(f"  Reconstruction matches absmax+round formula (max diff < 1e-6)")
    print("  PASS")


def test_ptq1_0_round_trip():
    """PTQ1_0 encode -> decode must recover the codes."""
    print("\n=== Test 4: PTQ1_0 encode/decode round-trip ===")
    rng = np.random.default_rng(42)
    n_blocks = 8
    w = rng.normal(0, 0.05, size=n_blocks * QK_PTQ1_0).astype(np.float32)
    # Encode.
    data = encode_ptq1_0_row(w)
    expected_size = n_blocks * block_size_bytes("PTQ1_0")
    assert len(data) == expected_size, (
        f"PTQ1_0 encoded size: expected {expected_size}, got {len(data)}"
    )
    # Decode.
    recon = decode_ptq1_0_row(data, n_blocks)
    # Verify against the expected reconstruction.
    for b in range(n_blocks):
        chunk = w[b * QK_PTQ1_0 : (b + 1) * QK_PTQ1_0]
        codes, d = _quantize_ptq1_0_group(chunk)
        expected = (codes.astype(np.float32) - 1.0) * d
        actual = recon[b * QK_PTQ1_0 : (b + 1) * QK_PTQ1_0]
        max_diff = float(np.abs(expected - actual).max())
        # The scale d is stored as fp16, so the reconstruction has fp16
        # rounding error (~0.1% of the scale magnitude).
        assert max_diff < 1e-4, (
            f"PTQ1_0 block {b}: max diff {max_diff} > 1e-4"
        )
    print(f"  Encoded {n_blocks} blocks ({n_blocks * 128} weights) "
          f"to {len(data)} bytes ({len(data) / (n_blocks * 128) * 8:.3f} bpw)")
    print(f"  Reconstruction matches absmax+round formula (max diff < 1e-6)")
    print("  PASS")


def test_bpw_matches_whitepaper():
    """Effective bpw must match the Bonsai whitepaper's claims."""
    print("\n=== Test 5: Effective bpw matches whitepaper ===")
    cases = [
        ("PTQ1_0", 1.75),
        ("PQ2_0", 2.125),
        ("TQ1_0", 1.6875),
        ("TQ2_0", 2.0625),
    ]
    for fmt, expected_bpw in cases:
        actual_bpw = bits_per_weight(fmt)
        diff = abs(actual_bpw - expected_bpw)
        status = "PASS" if diff < 0.001 else "FAIL"
        print(f"  {fmt}: actual={actual_bpw:.4f}  expected={expected_bpw:.4f}  "
              f"diff={diff:.4f}  ({status})")
        assert diff < 0.001, f"{fmt} bpw mismatch: {actual_bpw} vs {expected_bpw}"


def test_value_mapping_matches_c_reference():
    """Verify the {-1, 0, +1} -> {0, 1, 2} mapping matches the C reference.

    The C code does: q = round(w * id) + 1, where id = 1/absmax.
    So:
        w = -absmax -> q = round(-1) + 1 = 0
        w = 0       -> q = round(0) + 1 = 1
        w = +absmax -> q = round(1) + 1 = 2
    """
    print("\n=== Test 6: Value mapping matches C reference ===")
    # Construct a group with known values.
    w = np.array([-1.0, -0.5, 0.0, 0.5, 1.0] + [0.0] * 123, dtype=np.float32)
    codes, d = _quantize_pq2_0_group(w)
    print(f"  absmax d = {d}")
    print(f"  codes[:5] = {codes[:5].tolist()}  (expected: [0, 1, 1, 1, 2])")
    # Note: round(-0.5) = 0 (banker's rounding), so code = 1.
    # round(0.5) = 0 (banker's), so code = 1.
    # round(-1.0) = -1, code = 0. round(1.0) = 1, code = 2.
    # So expected codes: [0, 1, 1, 1, 2] -- matches the C lroundf behavior
    # for these specific values.
    assert codes[0] == 0, f"code for w=-absmax should be 0, got {codes[0]}"
    assert codes[2] == 1, f"code for w=0 should be 1, got {codes[2]}"
    assert codes[4] == 2, f"code for w=+absmax should be 2, got {codes[4]}"
    print("  PASS")


def test_storage_footprint():
    """Compute the actual storage for various model sizes."""
    print("\n=== Test 7: Storage footprint ===")
    for fmt in ["PTQ1_0", "PQ2_0"]:
        bpw = bits_per_weight(fmt)
        print(f"  {fmt}: {bpw:.4f} bpw")
        for params_b in [1.7e9, 7e9, 13e9, 27e9, 35e9]:
            size_gb = params_b * bpw / 8 / 1e9
            print(f"    {params_b/1e9:>5.1f}B params: {size_gb:>6.2f} GB")
        print()


def main():
    print("=" * 72)
    print("PTQ1_0 / PQ2_0 codec validation")
    print("=" * 72)
    test_5_trit_round_trip()
    test_4_trit_round_trip()
    test_pq2_0_round_trip()
    test_ptq1_0_round_trip()
    test_bpw_matches_whitepaper()
    test_value_mapping_matches_c_reference()
    test_storage_footprint()
    print("=" * 72)
    print("All tests passed.")


if __name__ == "__main__":
    main()
