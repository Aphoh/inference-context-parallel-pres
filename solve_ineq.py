#!/usr/bin/env python3
"""
Solve for T_max where Pass-Q is faster than Pass-KV.

Ring all-to-all:  (T * D * e) / (4 * BW) < t_kv_comm - t_kv_compute
NVL72 all-to-all: (T * D * e) / (N * 4 * BW) < t_kv_comm - t_kv_compute
"""

from sympy import symbols, Poly, expand, simplify, gcd

T, P, N_KV, N_H, N, BW, C, e, D = symbols(
    'T P N_KV N_H N BW C e D', positive=True, real=True
)

# MLA-specific symbols (DeepSeek V3)
# C_v = 512 (compressed KV dimension)
# C_rope = 576 (key dim with RoPE = 512 + 64)
# H_q = 128 (query heads)
C_v, C_rope, H_q = symbols('C_v C_rope H_q', positive=True, real=True)

# Common terms
tkv_comm = 2 * (P + T) * D * (N_KV / N_H) * (e / BW)
tkv_compute = (2 * (P + T) * T * D) / (N * C)

def solve_for_coeffs(tq_all2all, label):
    print(f"=== {label} ===")
    print()
    expr = expand(tq_all2all - (tkv_comm - tkv_compute))
    poly = Poly(expr, T)
    coeffs = poly.all_coeffs()
    common = gcd(coeffs)
    simplified = [simplify(c / common) for c in coeffs]
    print(f"α = {simplified[0]}")
    print(f"β = {simplified[1]}")
    print(f"γ = {simplified[2]}")
    print()

# Ring all-to-all: T_all2all = (T * D * e) / (4 * BW)
tq_ring = (T * D * e) / (4 * BW)
solve_for_coeffs(tq_ring, "Ring All-to-All: TDe/(4·BW)")

# NVL72 all-to-all: T_all2all = (T * D * e) / (N * 4 * BW)
tq_nvl72 = (T * D * e) / (N * 4 * BW)
solve_for_coeffs(tq_nvl72, "NVL72 All-to-All: TDe/(N·4·BW)")

# =============================================================================
# MLA (Multi-Head Latent Attention) - DeepSeek V3
# =============================================================================
# MLA stores a single compressed vector c_kv of dimension C_v per token
# (no factor of 2 for K and V - they share the same compressed representation)
#
# KV bytes: (P + T) * C_v * e
# FLOPs: 2 * T * (P + T) * H_q * (C_rope + C_v)
#   - QK matmul: 2 * T * (P+T) * H_q * C_rope
#   - AV matmul: 2 * T * (P+T) * H_q * C_v

# MLA terms (after weight absorption)
tkv_comm_mla = (P + T) * C_v * (e / BW)
tkv_compute_mla = (2 * (P + T) * T * H_q * (C_rope + C_v)) / (N * C)

def solve_for_coeffs_mla(tq_all2all, label):
    """Solve for MLA Pass-Q vs Pass-KV crossover."""
    print(f"=== {label} ===")
    print()
    expr = expand(tq_all2all - (tkv_comm_mla - tkv_compute_mla))
    poly = Poly(expr, T)
    coeffs = poly.all_coeffs()
    common = gcd(coeffs)
    simplified = [simplify(c / common) for c in coeffs]
    print(f"α = {simplified[0]}")
    print(f"β = {simplified[1]}")
    print(f"γ = {simplified[2]}")
    print()

# MLA Ring all-to-all: T_all2all = (T * D * e) / (4 * BW)
# Note: D here represents the query dimension that needs to be communicated
tq_ring_mla = (T * D * e) / (4 * BW)
solve_for_coeffs_mla(tq_ring_mla, "MLA Ring All-to-All: TDe/(4·BW)")

# MLA NVL72 all-to-all: T_all2all = (T * D * e) / (N * 4 * BW)
tq_nvl72_mla = (T * D * e) / (N * 4 * BW)
solve_for_coeffs_mla(tq_nvl72_mla, "MLA NVL72 All-to-All: TDe/(N·4·BW)")
