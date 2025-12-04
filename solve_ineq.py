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
