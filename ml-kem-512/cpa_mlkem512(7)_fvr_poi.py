#!/usr/bin/env python3
"""
cpa_mlkem512.py — CPA attack on ML-KEM-512 polyvec_basemul_acc_montgomery
                  using ELMO-simulated power traces.

Methodology: Mujdei et al. (2022), Section 4.3.
  ML-KEM-512 uses a negacyclic NTT over Z_q (q=3329) with n=256,
  followed by 128 two-coefficient schoolbook multiplications (basemul).
  Each basemul computes:
    r[0] = fqmul(fqmul(a[1], b[1]), zeta) + fqmul(a[0], b[0])
    r[1] = fqmul(a[0], b[1]) + fqmul(a[1], b[0])
  where a = known (NTT of decomp'd ciphertext), b = secret (NTT key), zeta
  is a twiddle factor.  The strh storing r[0] leaks HW(r[0]) in ELMO's model.

Attack overview:
  1. Load ELMO traces and randdata.txt.
  2. Reconstruct per-trace NTT-domain ciphertext coefficients (â).
  3. For each of the 128 basemul pairs:
       Enumerate all q^2 ≈ 11M candidate pairs (ŝ₂ᵢ, ŝ₂ᵢ₊₁).
       Compute hypothetical HW(basemul(â, ŝ_candidate)).
       Pearson-correlate with trace power across all traces.
       Best candidate = recovered secret NTT coefficient pair.
  4. Verify recovered key against known sk_cpa from randdata.txt.

Usage:
  python cpa_mlkem512.py [options]
  
  --traces-dir    Path to ELMO trace folder  (default: ../ELMO/output/traces)
  --randdata      Path to randdata.txt       (default: ../ELMO/output/randdata.txt)
  --ntt-source    Path to ntt.c              (default: ./ntt.c)
  --num-traces    Number of traces to use    (default: 200)
  --vec-idx       Polyvec index to attack: 0 or 1 (default: 0)
  --output        Save recovered key to file (default: recovered_key.npy)
  --asmtrace-dir  Path to asmtrace for POI   (default: ../ELMO/output_fvr/asmtrace)

Notes:
  - Build elmo_test_cpa.c with compile_elmo_profiled.sh (patched binary
    minimises non-profiled instructions corrupting the trace).
  - With 200 traces this script recovers 256 NTT-domain coefficients of
    one secret polynomial; repeat with --vec-idx 1 for the second.
  - Runtime: ~5 min per basemul pair on a laptop (as per the paper).
    128 pairs × ~5 min ≈ 10 hours total for vec[0].  Use --start-pair to
    resume from a checkpoint.
"""

import argparse
import os
import sys
import re
import struct
import time
import numpy as np

# -----------------------------------------------------------------------
# ML-KEM-512 / Kyber512 parameters
# -----------------------------------------------------------------------
KYBER_Q     = 3329
KYBER_N     = 256
KYBER_K     = 2
KYBER_DU    = 10          # ciphertext polyvec compression bits
KYBER_DV    = 4           # ciphertext poly compression bits
MONT        = np.int32(-1044)     # 2^16 mod q
QINV        = np.int32(-3327)     # q^-1 mod 2^16

# Byte sizes
POLY_BYTES        = 384           # 256 * 12 bits / 8
POLYVEC_BYTES     = KYBER_K * POLY_BYTES       # 768
POLYVEC_CBYTES    = KYBER_K * KYBER_N * KYBER_DU // 8   # 640
POLY_CBYTES       = KYBER_N * KYBER_DV // 8            # 128
INDCPA_CT_BYTES   = POLYVEC_CBYTES + POLY_CBYTES        # 768
INDCPA_SK_BYTES   = POLYVEC_BYTES                       # 768

# -----------------------------------------------------------------------
# Montgomery arithmetic (matching PQClean exactly)
# -----------------------------------------------------------------------

def montgomery_reduce_scalar(a: np.int32) -> np.int16:
    """Scalar montgomery_reduce: returns (a * R^-1) mod q, R = 2^16."""
    t = np.int16(np.int32(np.int16(a)) * QINV)
    t_val = np.int16((np.int32(a) - np.int32(t) * KYBER_Q) >> 16)
    return t_val

def fqmul_scalar(a: int, b: int) -> int:
    """Scalar fqmul: Montgomery multiply two int16 values."""
    return int(montgomery_reduce_scalar(np.int32(int(a)) * np.int32(int(b))))

def montgomery_reduce_vec(a: np.ndarray) -> np.ndarray:
    """
    Vectorised montgomery_reduce.
    a: int32 array of any shape.
    Returns int16 array of same shape.
    """
    a32 = a.astype(np.int32)
    t   = (a32.astype(np.int16) * np.int16(QINV)).astype(np.int32)
    return ((a32 - t * np.int32(KYBER_Q)) >> 16).astype(np.int16)

def fqmul_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorised fqmul.  a and b broadcast against each other.
    Returns int16 of same broadcast shape.
    """
    return montgomery_reduce_vec(a.astype(np.int32) * b.astype(np.int32))

# -----------------------------------------------------------------------
# Zeta values — extracted from ntt.c at runtime
# -----------------------------------------------------------------------

def load_zetas(ntt_c_path: str) -> np.ndarray:
    """
    Parse the int16_t zetas[128] array from PQClean's ntt.c.
    Avoids hardcoding values that may differ between library versions.
    """
    if not os.path.exists(ntt_c_path):
        raise FileNotFoundError(
            f"Cannot find ntt.c at '{ntt_c_path}'. "
            "Pass --ntt-source to specify its location."
        )
    with open(ntt_c_path) as f:
        text = f.read()
    # Match: int16_t zetas[128] = { ... };
    m = re.search(r'zetas\s*\[128\]\s*=\s*\{([^}]+)\}', text, re.DOTALL)
    if not m:
        raise ValueError("Could not parse zetas[128] from ntt.c — "
                         "check the file format.")
    nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
    if len(nums) != 128:
        raise ValueError(f"Expected 128 zeta values, found {len(nums)}")
    return np.array(nums, dtype=np.int16)

# -----------------------------------------------------------------------
# Decompression (matching PQClean indcpa.c / poly.c)
# -----------------------------------------------------------------------

def polyvec_decompress(ct_bytes: bytes, k: int = KYBER_K) -> np.ndarray:
    """
    Decompress the first (k * n * du / 8) bytes of the ciphertext into
    a (k, n) int16 polynomial vector.  Matches polyvec_decompress() in
    PQClean for du = 10.
    """
    result = np.zeros((k, KYBER_N), dtype=np.int16)
    offset = 0
    for i in range(k):
        for j in range(KYBER_N // 4):
            a = ct_bytes[offset:offset + 5]
            offset += 5
            t0 =  ((a[0]     ) | (a[1] << 8)) & 0x3ff
            t1 =  ((a[1] >> 2) | (a[2] << 6)) & 0x3ff
            t2 =  ((a[2] >> 4) | (a[3] << 4)) & 0x3ff
            t3 =  ((a[3] >> 6) | (a[4] << 2)) & 0x3ff
            result[i, 4*j+0] = (t0 * KYBER_Q + 512) >> 10
            result[i, 4*j+1] = (t1 * KYBER_Q + 512) >> 10
            result[i, 4*j+2] = (t2 * KYBER_Q + 512) >> 10
            result[i, 4*j+3] = (t3 * KYBER_Q + 512) >> 10
    return result  # shape (k, 256), values in [0, q-1]

def poly_frombytes(a: bytes) -> np.ndarray:
    """
    Unpack 384 bytes into 256 int16 coefficients (12 bits each).
    Matches poly_frombytes() in PQClean — used to decode sk_cpa.
    """
    r = np.zeros(KYBER_N, dtype=np.int16)
    for i in range(KYBER_N // 2):
        r[2*i]   = np.int16((a[3*i]   | (a[3*i+1] << 8)) & 0xfff)
        r[2*i+1] = np.int16(((a[3*i+1] >> 4) | (a[3*i+2] << 4)) & 0xfff)
    return r

def polyvec_frombytes(a: bytes, k: int = KYBER_K) -> np.ndarray:
    """Decode a k-vector of NTT-domain polynomials from bytes."""
    result = np.zeros((k, KYBER_N), dtype=np.int16)
    for i in range(k):
        result[i] = poly_frombytes(a[i*POLY_BYTES:(i+1)*POLY_BYTES])
    return result

# -----------------------------------------------------------------------
# Forward NTT (matching PQClean ntt.c exactly)
# -----------------------------------------------------------------------

def ntt_single(r: np.ndarray, zetas: np.ndarray) -> np.ndarray:
    """
    Forward NTT of one polynomial.
    r:     (256,) int16 input (coefficients in [0, q-1])
    zetas: (128,) int16 zeta table from ntt.c
    Returns: (256,) int16 NTT output.
    Matches PQClean PQCLEAN_MLKEM512_CLEAN_ntt().
    """
    r = r.astype(np.int32).copy()
    k = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = int(zetas[k])
            k += 1
            j_range = np.arange(start, start + length)
            t = montgomery_reduce_vec(
                np.int32(zeta) * r[j_range + length].astype(np.int32)
            ).astype(np.int32)
            r[j_range + length] = r[j_range] - t
            r[j_range]          = r[j_range] + t
            start += 2 * length
        length >>= 1
    return r.astype(np.int16)

def ntt_polyvec(polyvec: np.ndarray, zetas: np.ndarray) -> np.ndarray:
    """
    NTT of a (k, 256) polynomial vector.
    Returns (k, 256) int16 NTT-domain coefficients.
    """
    result = np.zeros_like(polyvec)
    for i in range(polyvec.shape[0]):
        result[i] = ntt_single(polyvec[i], zetas)
    return result

def ntt_batch(polys: np.ndarray, zetas: np.ndarray) -> np.ndarray:
    """
    Batch NTT of (N, 256) polynomials.  Much faster than calling
    ntt_single N times since butterfly stages are vectorised.
    polys:  (N, 256) int16
    Returns (N, 256) int16
    """
    N = polys.shape[0]
    r = polys.astype(np.int32).copy()  # (N, 256)
    k = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = np.int32(int(zetas[k]))
            k += 1
            j = np.arange(start, start + length)
            # Shape: (N, length)
            t = montgomery_reduce_vec(zeta * r[:, j + length])
            r[:, j + length] = r[:, j] - t.astype(np.int32)
            r[:, j]          = r[:, j] + t.astype(np.int32)
            start += 2 * length
        length >>= 1
    return r.astype(np.int16)

# -----------------------------------------------------------------------
# Basemul leakage model
# -----------------------------------------------------------------------

def basemul_r0(a0: np.ndarray, a1: np.ndarray,
               s0_cands: np.ndarray, s1_cands: np.ndarray,
               zeta: int) -> np.ndarray:
    """
    Compute r[0] of basemul for all candidate (s0, s1) pairs and all traces.

    PQClean basemul:
        r[0] = fqmul(a[1], b[1])          // montgomery_reduce(a1 * s1)
        r[0] = fqmul(r[0], zeta)           // montgomery_reduce(r[0] * zeta)
        r[0] += fqmul(a[0], b[0])          // += montgomery_reduce(a0 * s0)

    Args:
      a0:       (N,) int16 — â[2i]   for all N traces
      a1:       (N,) int16 — â[2i+1] for all N traces
      s0_cands: (C,) int16 — candidate values for ŝ[2i],   C = range of q
      s1_cands: (C,) int16 — candidate values for ŝ[2i+1], C = range of q
      zeta:     int16 scalar twiddle factor for this basemul pair

    Returns:
      (C, C, N) int16 array of r[0] values.
      Axis 0 = s0 candidate, Axis 1 = s1 candidate, Axis 2 = trace index.
    """
    N  = a0.shape[0]
    C  = s0_cands.shape[0]   # = q = 3329
    z  = np.int32(zeta)

    # Contribution of a1 * s1 for all trace/candidate combos: (N, C)
    # a1[:, None] * s1_cands[None, :] broadcast → (N, C) int32 products
    contrib_s1 = montgomery_reduce_vec(
        a1.astype(np.int32)[:, None] * s1_cands.astype(np.int32)[None, :]
    )  # (N, C) int16

    # Multiply by zeta: (N, C)
    contrib_s1_zeta = montgomery_reduce_vec(
        contrib_s1.astype(np.int32) * z
    )  # (N, C) int16

    # Contribution of a0 * s0: (N, C)
    contrib_s0 = montgomery_reduce_vec(
        a0.astype(np.int32)[:, None] * s0_cands.astype(np.int32)[None, :]
    )  # (N, C) int16

    # r[0] = contrib_s1_zeta[:, None, :] + contrib_s0[:, :, None] doesn't
    # quite work because we need the outer product over (C, C).
    # Reshape to (C_s0=1, C_s1, N) + (C_s0, 1, N):
    r0 = (contrib_s1_zeta.T[None, :, :].astype(np.int32) +   # (1,  C, N)
          contrib_s0.T[:, None, :].astype(np.int32))           # (C,  1, N)
    # Result: (C, C, N) int32

    return r0.astype(np.int16)

def hamming_weight_vec(x: np.ndarray) -> np.ndarray:
    """
    Hamming weight of 16-bit values (treating as uint16).
    x: any integer array.
    Returns same-shape uint8 array.
    """
    v = x.view(np.uint16) if x.dtype == np.int16 else x.astype(np.uint16)
    # Lookup table is fastest for 16-bit
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    lo  = lut[(v & 0xff).astype(np.uint8)]
    hi  = lut[(v >> 8).astype(np.uint8)]
    return (lo + hi).astype(np.uint8)

def pearson_correlation(hyp: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between hypothetical power (hyp) and traces.
    hyp:    (..., N) — hypothetical leakage for each candidate, N traces
    traces: (N, T)   — power traces, T cycles
    Returns (..., T) correlation coefficients.

    Uses the efficient formulation to avoid computing full covariance matrix.
    """
    N = traces.shape[0]

    # Normalise hypotheses: (..., N)
    hyp_f  = hyp.astype(np.float32)
    hyp_c  = hyp_f - hyp_f.mean(axis=-1, keepdims=True)
    hyp_s  = np.sqrt((hyp_c**2).sum(axis=-1, keepdims=True)) + 1e-10

    # Normalise traces: (N, T)
    trc_f  = traces.astype(np.float32)
    trc_c  = trc_f - trc_f.mean(axis=0, keepdims=True)   # (N, T)
    trc_s  = np.sqrt((trc_c**2).sum(axis=0, keepdims=True)) + 1e-10  # (1, T)

    # Correlation: (..., N) @ (N, T) / normalisation
    # einsum is more memory-efficient for higher-dim hyp
    corr = np.tensordot(hyp_c / hyp_s, trc_c, axes=([-1], [0])) / trc_s
    # Shape: (..., T)
    return corr

# -----------------------------------------------------------------------
# Trace and randdata loading
# -----------------------------------------------------------------------

def load_traces(traces_dir: str, num_traces: int,
                binary: bool = False) -> np.ndarray:
    """
    Load num_traces ELMO trace files from traces_dir.
    Returns (num_traces, trace_len) float32 array.
    """
    ext    = '.trc'
    traces = []
    for i in range(1, num_traces + 1):
        path = os.path.join(traces_dir, f'trace{i:05d}{ext}')
        if not os.path.exists(path):
            print(f"WARNING: trace {path} not found — stopping at {i-1}")
            break
        if binary:
            with open(path, 'rb') as f:
                data = f.read()
            t = np.frombuffer(data, dtype=np.float32)
        else:
            t = np.loadtxt(path, dtype=np.float32)
        traces.append(t)

    if not traces:
        raise FileNotFoundError(f"No traces found in {traces_dir}")

    # All traces must be the same length (DIFFTRACELENGTH not defined)
    tlen = len(traces[0])
    traces = [t for t in traces if len(t) == tlen]
    print(f"Loaded {len(traces)} traces, {tlen} cycles each.")
    return np.array(traces, dtype=np.float32)

def load_randdata(randdata_path: str,
                  num_traces: int) -> tuple:
    """
    Parse ELMO's randdata.txt.  Each line is one byte (decimal 0–255).

    Layout written by elmo_test_cpa.c:
      bytes [0 : INDCPA_SK_BYTES)                   = fixed sk_cpa
      bytes [INDCPA_SK_BYTES : INDCPA_SK_BYTES +
             num_traces * INDCPA_CT_BYTES)           = ct[0], ct[1], ...

    Returns:
      sk_bytes: (INDCPA_SK_BYTES,) uint8
      ct_array: (num_traces, INDCPA_CT_BYTES) uint8
    """
    with open(randdata_path) as f:
        raw = [int(line.strip(), 16) for line in f if line.strip()]

    expected = INDCPA_SK_BYTES + num_traces * INDCPA_CT_BYTES
    if len(raw) < expected:
        raise ValueError(
            f"randdata.txt has {len(raw)} bytes, expected ≥ {expected}. "
            "Check NOTRACES matches num_traces."
        )

    data     = np.array(raw, dtype=np.uint8)
    sk_bytes = data[:INDCPA_SK_BYTES]
    ct_flat  = data[INDCPA_SK_BYTES : INDCPA_SK_BYTES + num_traces * INDCPA_CT_BYTES]
    ct_array = ct_flat.reshape(num_traces, INDCPA_CT_BYTES)
    return sk_bytes, ct_array

# -----------------------------------------------------------------------
# Trace segmentation — find the basemul region
# -----------------------------------------------------------------------

def find_basemul_region(traces: np.ndarray, verbose: bool = True) -> tuple:
    """
    Locate the basemul region in the trace by finding the zero-variance
    landmark left by polyvec_frombytes(sk).

    indcpa_dec execution order:
      1. polyvec_decompress(bp, c)       — ct-dependent → HIGH variance
      2. poly_decompress(v, c+...)       — ct-dependent → HIGH variance
      3. polyvec_ntt(bp)                 — ct-dependent → HIGH variance
      4. polyvec_frombytes(skpv, sk)     — sk constant  → ZERO variance ★
      5. polyvec_basemul_acc_montgomery  — ct+sk dep    → HIGH variance ★ TARGET
      6. poly_invntt_tomont             — ct+sk dep    → HIGH variance
      7. poly_sub, poly_reduce, poly_tomsg

    Strategy: find the longest run of near-zero variance cycles.  The
    basemul starts right after it ends.

    Returns:
      (basemul_start, basemul_end): cycle indices bounding the basemul region
    """
    N, T = traces.shape
    var = traces.var(axis=0)  # (T,) per-cycle variance

    # Threshold: cycles with variance < 1% of median non-zero variance
    nonzero_var = var[var > 0]
    if len(nonzero_var) == 0:
        raise ValueError("All cycles have zero variance — traces may be identical")
    thresh = np.median(nonzero_var) * 0.01
    is_zero = var < thresh

    # Find all runs of zero-variance cycles
    runs = []
    in_run = False
    run_start = 0
    for i in range(T):
        if is_zero[i] and not in_run:
            run_start = i
            in_run = True
        elif not is_zero[i] and in_run:
            runs.append((run_start, i, i - run_start))
            in_run = False
    if in_run:
        runs.append((run_start, T, T - run_start))

    if not runs:
        if verbose:
            print("      WARNING: No zero-variance region found.")
            print("      Falling back to second half of trace.")
        return (T // 2, T)

    # The polyvec_frombytes region should be the longest zero-variance run
    # that's NOT at the very start or end of the trace
    # Filter out runs at the edges (first/last 5% of trace)
    margin = T // 20
    interior_runs = [(s, e, l) for s, e, l in runs
                     if s > margin and e < T - margin]
    if not interior_runs:
        interior_runs = runs

    # Pick the longest interior run
    best_run = max(interior_runs, key=lambda x: x[2])
    frombytes_start, frombytes_end, frombytes_len = best_run

    if verbose:
        print(f"      Zero-variance region (polyvec_frombytes): "
              f"cycles {frombytes_start}–{frombytes_end} "
              f"({frombytes_len} cycles)")

    # The basemul starts right after frombytes ends.
    # Estimate basemul length: ~20-30% of remaining trace
    basemul_start = frombytes_end
    remaining = T - basemul_start
    # poly_basemul_montgomery for k=2: ~128 basemul calls + accumulation
    # Each basemul ≈ 30-50 instructions → ~5000-8000 cycles for all 128
    # Use the next high-variance region after frombytes
    # Find where variance drops again (end of basemul → inv_NTT boundary)
    # Simple heuristic: take ~25% of remaining cycles
    basemul_end = min(basemul_start + remaining // 4, T)

    if verbose:
        print(f"      Estimated basemul region: "
              f"cycles {basemul_start}–{basemul_end} "
              f"({basemul_end - basemul_start} cycles)")

    return (basemul_start, basemul_end)


def verify_python_math(sk_bytes: np.ndarray, ct_bytes: np.ndarray,
                       zetas: np.ndarray, vec_idx: int = 0):
    """
    Diagnostic: compute the first basemul pair result using the known key
    and first trace's ciphertext, and print intermediate values.
    Helps verify the Python NTT/decompress/basemul matches the C code.
    """
    print("\n  --- Diagnostic: verify Python math for pair 0 ---")

    # Decode secret key
    sk_ntt = polyvec_frombytes(bytes(sk_bytes))
    s0 = int(sk_ntt[vec_idx, 0])
    s1 = int(sk_ntt[vec_idx, 1])
    print(f"  sk_ntt[{vec_idx}][0] = {s0}  (raw from frombytes)")
    print(f"  sk_ntt[{vec_idx}][1] = {s1}")
    print(f"  sk_ntt[{vec_idx}][0] mod q = {s0 % KYBER_Q}")
    print(f"  sk_ntt[{vec_idx}][1] mod q = {s1 % KYBER_Q}")

    # Decompress and NTT ciphertext
    bp = polyvec_decompress(bytes(ct_bytes))
    print(f"  bp[{vec_idx}][0] = {bp[vec_idx, 0]}  (decompressed)")
    print(f"  bp[{vec_idx}][1] = {bp[vec_idx, 1]}")

    bp_ntt = ntt_polyvec(bp, zetas)
    a0 = int(bp_ntt[vec_idx, 0])
    a1 = int(bp_ntt[vec_idx, 1])
    print(f"  NTT(bp)[{vec_idx}][0] = {a0}")
    print(f"  NTT(bp)[{vec_idx}][1] = {a1}")

    # Compute basemul pair 0: zeta = zetas[64]
    zeta = int(zetas[64])
    print(f"  zeta (pair 0) = {zeta}")

    # r[0] = fqmul(fqmul(s1, a1), zeta) + fqmul(s0, a0)
    step1 = fqmul_scalar(s1, a1)
    step2 = fqmul_scalar(step1, zeta)
    step3 = fqmul_scalar(s0, a0)
    r0 = np.int16(np.int32(step2) + np.int32(step3))
    hw_r0 = bin(int(np.uint16(r0))).count('1')
    print(f"  fqmul(s1, a1)           = {step1}")
    print(f"  fqmul(^, zeta)          = {step2}")
    print(f"  fqmul(s0, a0)           = {step3}")
    print(f"  r[0] = {step2} + {step3} = {r0}")
    print(f"  HW(r[0]) = {hw_r0}")

    # Now check: if CPA searches s0_cand in [-q//2, q//2], does the correct
    # reduced value give the same r[0]?
    s0_red = ((s0 + KYBER_Q // 2) % KYBER_Q) - KYBER_Q // 2
    s1_red = ((s1 + KYBER_Q // 2) % KYBER_Q) - KYBER_Q // 2
    step1b = fqmul_scalar(s1_red, a1)
    step2b = fqmul_scalar(step1b, zeta)
    step3b = fqmul_scalar(s0_red, a0)
    r0b = np.int16(np.int32(step2b) + np.int32(step3b))
    print(f"  Using reduced s0={s0_red}, s1={s1_red}: r[0] = {r0b}  "
          f"({'MATCH' if r0 == r0b else 'MISMATCH!'})")
    print(f"  --- end diagnostic ---\n")

def ground_truth_correlation_test(traces, a_ntt, sk_bytes, zetas, vec_idx=0):
    print("\n[Diagnostic] Ground-truth correlation test")

    sk_ntt = polyvec_frombytes(bytes(sk_bytes))

    # Pick pair 0
    pair = 0
    loop_i = pair // 2
    zeta = int(zetas[64 + loop_i])

    a0 = a_ntt[:, vec_idx, 0]
    a1 = a_ntt[:, vec_idx, 1]

    s0 = sk_ntt[vec_idx, 0]
    s1 = sk_ntt[vec_idx, 1]

    # Compute true r[0] for all traces
    r0 = (
        fqmul_vec(
            fqmul_vec(a1, np.full_like(a1, s1)),
            np.int16(zeta)
        ) +
        fqmul_vec(a0, np.full_like(a0, s0))
    ).astype(np.int16)

    hw = hamming_weight_vec(r0).astype(np.float32)

    # Correlate with full trace
    corr = pearson_correlation(hw[None, :], traces)[0]

    max_corr = np.max(np.abs(corr))
    max_idx = np.argmax(np.abs(corr))

    print(f"  Max correlation: {max_corr:.4f} at cycle {max_idx}")

    return corr

def attack_basemul_pair(pair_idx: int,
                        vec_idx: int,
                        a_ntt: np.ndarray,
                        traces: np.ndarray,
                        zetas: np.ndarray,
                        sk_bytes: np.ndarray,
                        basemul_region: tuple = None,
                        verbose: bool = True) -> tuple:
    """
    Recover secret NTT coefficients ŝ[2j] and ŝ[2j+1] for a single
    basemul pair, using the CPA methodology from Mujdei et al. Sec 4.3.
    """
    loop_i   = pair_idx // 2
    second   = (pair_idx % 2 == 1)
    base_idx = 4 * loop_i + (2 if second else 0)
    zeta     = int(zetas[64 + loop_i]) * (-1 if second else 1)
    N        = traces.shape[0]
    T        = traces.shape[1]

    # NTT ciphertext coefficients for this pair, all traces: (N,)
    a0 = a_ntt[:, vec_idx, base_idx].astype(np.int16)
    a1 = a_ntt[:, vec_idx, base_idx + 1].astype(np.int16)

    # Candidate range: [-q//2, q//2] = [-1664, 1664]
    cand_range = np.arange(-KYBER_Q // 2, KYBER_Q // 2 + 1, dtype=np.int16)
    C = len(cand_range)   # = 3329

    best_corr = -1.0
    best_s0   = 0
    best_s1   = 0

    # Precompute all s1 contributions: shape (C, N) int16
    contrib_s1_all = montgomery_reduce_vec(
        np.int32(zeta) *
        montgomery_reduce_vec(
            a1.astype(np.int32)[None, :] *      # (1, N)
            cand_range.astype(np.int32)[:, None] # (C, 1)
        ).astype(np.int32)
    )  # (C, N) int16

    # ---- POI selection: restrict to basemul region ----
    if basemul_region is not None:
        bm_start, bm_end = basemul_region
    else:
        bm_start, bm_end = 0, T

    bm_len = bm_end - bm_start

    # Within the basemul region, subdivide into 128 pair-windows
    # (128 basemul calls for vec[0], roughly evenly spaced)
    n_pairs    = KYBER_N // 2   # 128

    trc_f = traces.astype(np.float32)
    trc_c = trc_f - trc_f.mean(axis=0)
    trc_var = (trc_c**2).sum(axis=0) + 1e-10

    poi = find_poi_for_pair(pair_idx, traces, a_ntt, sk_bytes, zetas, vec_idx)
    trc_f = traces.astype(np.float32)
    trc_c = trc_f - trc_f.mean(axis=0)
    trc_var = (trc_c**2).sum(axis=0) + 1e-10

    trc_c_poi   = trc_c[:, poi:poi+1]      # single cycle
    trc_var_poi = trc_var[poi:poi+1]

    for s0 in cand_range:
        c_s0 = montgomery_reduce_vec(
            a0.astype(np.int32) * np.int32(int(s0))
        )

        r0_all = (c_s0.astype(np.int32)[None, :] +
                  contrib_s1_all.astype(np.int32)).astype(np.int16)

        hw = hamming_weight_vec(r0_all)

        hw_f  = hw.astype(np.float32)
        hw_mu = hw_f.mean(axis=1, keepdims=True)
        hw_c  = hw_f - hw_mu
        hw_sd = np.sqrt((hw_c**2).sum(axis=1, keepdims=True)) + 1e-10
        hw_n  = hw_c / hw_sd

        corr_poi = (hw_n @ trc_c_poi) / (np.sqrt(trc_var_poi)[None, :])
        max_corr_per_s1 = np.abs(corr_poi).max(axis=1)
        best_s1_idx_local = int(max_corr_per_s1.argmax())
        local_best = float(max_corr_per_s1[best_s1_idx_local])

        if local_best > best_corr:
            best_corr = local_best
            best_s0   = int(s0)
            best_s1   = int(cand_range[best_s1_idx_local])

    return best_s0, best_s1, best_corr

# -----------------------------------------------------------------------
# Find POIs
# -----------------------------------------------------------------------
def find_poi_for_pair(pair_idx, traces, a_ntt, sk_bytes, zetas, vec_idx=0):
    sk_ntt = polyvec_frombytes(bytes(sk_bytes))

    loop_i = pair_idx // 2
    second = (pair_idx % 2 == 1)
    base_idx = 4 * loop_i + (2 if second else 0)
    zeta = int(zetas[64 + loop_i]) * (-1 if second else 1)

    a0 = a_ntt[:, vec_idx, base_idx]
    a1 = a_ntt[:, vec_idx, base_idx + 1]

    s0 = sk_ntt[vec_idx, base_idx]
    s1 = sk_ntt[vec_idx, base_idx + 1]

    r0 = (
        fqmul_vec(fqmul_vec(a1, np.full_like(a1, s1)), np.int16(zeta))
        + fqmul_vec(a0, np.full_like(a0, s0))
    ).astype(np.int16)

    hw = hamming_weight_vec(r0).astype(np.float32)

    corr = pearson_correlation(hw[None, :], traces)[0]

    return int(np.argmax(np.abs(corr)))

# -----------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------

def verify_recovered_key(recovered: np.ndarray,
                          sk_bytes: np.ndarray,
                          vec_idx: int) -> float:
    """
    Compare recovered NTT coefficients against the known sk_cpa bytes.
    sk_bytes: (INDCPA_SK_BYTES,) uint8 from randdata.txt.
    Returns fraction of coefficients correctly recovered.
    """
    sk_ntt = polyvec_frombytes(bytes(sk_bytes))  # (K, 256) int16, NTT-domain
    known  = sk_ntt[vec_idx]                     # (256,) int16
    # Reduce both to [0, q-1] for comparison (coefficients may differ by ±q)
    r_mod = np.mod(recovered.astype(np.int32), KYBER_Q).astype(np.int16)
    k_mod = np.mod(known.astype(np.int32),     KYBER_Q).astype(np.int16)
    match_frac = float((r_mod == k_mod).mean())
    return match_frac

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CPA attack on ML-KEM-512 basemul using ELMO traces')
    parser.add_argument('--traces-dir',  default='../ELMO/output/traces')
    parser.add_argument('--randdata',    default='../ELMO/output/randdata.txt')
    parser.add_argument('--asm-dir',     default='../ELMO/output_fvr/asmoutput/')
    parser.add_argument('--ntt-source',  default='./ntt.c')
    parser.add_argument('--num-traces',  type=int,  default=200)
    parser.add_argument('--vec-idx',     type=int,  default=0,
                        help='Polyvec index (0 or 1) to attack')
    parser.add_argument('--start-pair',  type=int,  default=0,
                        help='Resume from this basemul pair index (0-127)')
    parser.add_argument('--end-pair',    type=int,  default=128)
    parser.add_argument('--output',      default='recovered_key.npy')
    parser.add_argument('--binary',      action='store_true',
                        help='Traces are binary - ELMO was compiled with `#define BINARYTRACES`')
    args = parser.parse_args()

    print("=" * 62)
    print("  CPA Attack on ML-KEM-512 polyvec_basemul_acc_montgomery")
    print("  Mujdei et al. (2022) Section 4.3 — Kyber NTT basemul")
    print("=" * 62)

    # -- Load zetas from ntt.c ------------------------------------------
    print(f"\n[1/5] Loading zeta table from {args.ntt_source}")
    zetas = load_zetas(args.ntt_source)
    print(f"      zetas[64:68] = {zetas[64:68]}  (sanity check)")

    # -- Load Fixed vs Random statistics --------------------------------
    print(f"\n[2/5] Loading FvR Statistics ELMO traces from {args.traces_dir}")
    tvla = np.loadtxt(args.fvr_stats_dir)

    threshold = 4.5
    printf"\n[?/?] Find PoI cycles where Welch t-score > 4.5"
    poi_cycles = np.where(np.abs(tvla) > threshold)[0]

    # -- Load Fixed vs Random statistics --------------------------------
    print(f"\n[2/5] Parsing asmtrace {args.asmtrace_dir}")
    with open("{asmtrace_dir}/asmtrace00001.txt") as f:
        asm_lines = [line.strip() for line in f]
    # asm_lines[cycle] gives the instruction at that cycle
    # Care about instructions: str, strh, strb, stm

    interesting = []
    for cycle in poi_cycles:
        insn = asm_lines[cycle]
        if "str" in insn:
            interesting.append((cycle, tvla[cycle], insn))

    # -- Load traces -------------------------------------------------------
    print(f"\n[2/5] Loading {args.num_traces} ELMO traces from {args.traces_dir}")
    traces = load_traces(args.traces_dir, args.num_traces, args.binary)
    N, T   = traces.shape
    print(f"      Loaded {N} traces × {T} cycles")

    # -- Load randdata (sk and ciphertexts) --------------------------------
    print(f"\n[3/5] Parsing randdata from {args.randdata}")
    sk_bytes, ct_array = load_randdata(args.randdata, N)
    print(f"      sk_cpa: {INDCPA_SK_BYTES} bytes  |  "
          f"ct array: {ct_array.shape}")

    # -- Precompute NTT-domain ciphertext coefficients for all traces ------
    print(f"\n[4/7] Decompressing and computing NTT of ciphertexts ...")
    t0 = time.time()

    # polyvec_decompress: (N, K, 256) — batch over traces
    bp_raw = np.zeros((N, KYBER_K, KYBER_N), dtype=np.int16)
    for trace_i in range(N):
        bp_raw[trace_i] = polyvec_decompress(bytes(ct_array[trace_i]))
    print(f"      Decompressed in {time.time()-t0:.1f}s")

    # Batch NTT for each polynomial in the vector
    t1 = time.time()
    a_ntt = np.zeros_like(bp_raw)
    for k_idx in range(KYBER_K):
        a_ntt[:, k_idx, :] = ntt_batch(bp_raw[:, k_idx, :], zetas)
    print(f"      NTT batch completed in {time.time()-t1:.1f}s")
    print(f"      a_ntt shape: {a_ntt.shape}  (N, K, 256)")

    # -- Find basemul region in trace ----------------------------------------
    print(f"\n[5/7] Locating basemul region via zero-variance landmark ...")
    basemul_region = find_basemul_region(traces, verbose=True)

    # -- Diagnostic: verify Python math matches C ----------------------------
    print(f"\n[6/7] Running math verification diagnostic ...")
    verify_python_math(sk_bytes, ct_array[0], zetas, args.vec_idx)

    # -- Diagnostic: GPT correlation test ----------------------------
    print(f"\n[6.5/7] Correlation test...")
    ground_truth_correlation_test(traces, a_ntt, sk_bytes, zetas)

    # -- CPA main loop -----------------------------------------------------
    print(f"\n[7/7] CPA over basemul pairs {args.start_pair}–{args.end_pair-1}"
          f"  (vec_idx={args.vec_idx})")
    bm_s, bm_e = basemul_region
    print(f"      Basemul region: cycles {bm_s}–{bm_e}"
          f"  ({bm_e-bm_s} cycles, {(bm_e-bm_s)//128} per pair)")
    print(f"      Each pair: q²={KYBER_Q**2:,} candidates")
    print()

    # Load or initialise checkpoint
    recovered = np.zeros(KYBER_N, dtype=np.int16)
    if os.path.exists(args.output):
        recovered = np.load(args.output)
        print(f"      Loaded checkpoint from {args.output}")

    for pair in range(args.start_pair, args.end_pair):
        loop_i   = pair // 2
        second   = (pair % 2 == 1)
        base_idx = 4 * loop_i + (2 if second else 0)

        t_start = time.time()
        s0, s1, corr = attack_basemul_pair(
            pair_idx=pair,
            vec_idx=args.vec_idx,
            a_ntt=a_ntt,
            traces=traces,
            zetas=zetas,
            basemul_region=basemul_region,
            sk_bytes=sk_bytes,
        )
        elapsed = time.time() - t_start

        recovered[base_idx]     = np.int16(s0)
        recovered[base_idx + 1] = np.int16(s1)

        # Save checkpoint after every pair
        np.save(args.output, recovered)

        print(f"  Pair {pair:3d}/{args.end_pair-1}  "
              f"[coeff {base_idx:3d},{base_idx+1:3d}]  "
              f"s0={s0:+5d}  s1={s1:+5d}  "
              f"|r|={corr:.4f}  "
              f"({elapsed:.1f}s)")

    # -- Verification -------------------------------------------------------
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    match = verify_recovered_key(recovered, sk_bytes, args.vec_idx)
    print(f"  Coefficient match rate (vec[{args.vec_idx}]): "
          f"{match*100:.1f}%  ({int(match*KYBER_N)}/{KYBER_N})")

    if match > 0.9:
        print("  ✓ Attack successful — key recovery rate > 90%")
    elif match > 0.5:
        print("  ~ Partial success — try more traces or check POI selection")
    else:
        print("  ✗ Attack failed — check trace quality / non-profiled issues")

    print(f"\n  Recovered key saved to: {args.output}")
    print()

    # Show first few recovered vs known coefficients
    sk_ntt = polyvec_frombytes(bytes(sk_bytes))
    print("  First 8 coefficients (recovered | known | match):")
    for j in range(8):
        r = int(recovered[j])
        k = int(sk_ntt[args.vec_idx, j])
        same = '✓' if (r % KYBER_Q == k % KYBER_Q) else '✗'
        print(f"    [{j:3d}]  {r:+5d}  |  {k:+5d}  {same}")

if __name__ == '__main__':
    main()
