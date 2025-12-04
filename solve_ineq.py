#!/usr/bin/env python3
"""
Solve for T_max where Pass-Q is faster than Pass-KV.

Starting inequality:
    (T * D * e) / (4 * BW) < 2 * (P + T) * D * (N_KV / N_H * e / BW - T / (N * C))
"""

from sympy import symbols, Poly, expand, latex, simplify, gcd

T, P, N_KV, N_H, N, BW, C, e, D = symbols(
    'T P N_KV N_H N BW C e D', positive=True, real=True
)

# Original inequality (line 743):
# t_all2all < t_kv_comm - t_kv_compute
tkv_comm = 2 * (P + T) * D * (N_KV / N_H) * (e / BW)
tkv_compute = (2 * (P + T) * T * D) / (N * C)
tq_all2all = (T * D * e) / (4 * BW)

# Rearrange to: t_all2all - (t_kv_comm - t_kv_compute) < 0
expr = expand(tq_all2all - (tkv_comm - tkv_compute))
poly = Poly(expr, T)
coeffs = poly.all_coeffs()  # [T^2 coeff, T^1 coeff, T^0 coeff]

# Factor out common terms
common = gcd(coeffs)
simplified = [simplify(c / common) for c in coeffs]

print("Quadratic: α·T² + β·T + γ < 0")
print()
print(f"α = {simplified[0]}")
print(f"β = {simplified[1]}")
print(f"γ = {simplified[2]}")
