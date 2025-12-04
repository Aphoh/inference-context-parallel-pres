#!/usr/bin/env python3
"""
Plot TTFT for pass-KV vs pass-Q varying P and T with P + T = 128000
on 4 CP ranks (CP4).

P: length of existing tokens in the KV cache
T: length of new tokens
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 4
data = {
    'P': [126720, 124800, 123840, 121600, 115200, 102400, 89600, 76800, 64000, 51200, 38400, 25600, 12800, 0],
    'T': [1280, 3200, 4160, 6400, 12800, 25600, 38400, 51200, 64000, 76800, 89600, 102400, 115200, 128000],
    'miss_rate': [1.00, 2.50, 3.25, 5.00, 10.00, 20.00, 30.00, 40.00, 50.00, 60.00, 70.00, 80.00, 90.00, 100.00],
    'pass_kv': [1023.39, 1110.18, 1298.92, 1305.56, 2080.67, 3353.02, 4629.23, 5745.08, 6845.21, 7890.35, 8697.27, 10105.78, 11136.4, 11462.15],
    'pass_q': [898.71, 1046.43, 1280.1, 1302.01, 2205.27, 3617.02, 4922.52, 6217.83, 7367.99, 8468.66, 9666.62, 10652.39, 11571.62, 12360.57],
}

miss_rate = np.array(data['miss_rate'])
pass_kv = np.array(data['pass_kv'])
pass_q = np.array(data['pass_q'])

# Compute ratio
ratio = pass_kv / pass_q

# Create figure (wider and shorter)
plt.switch_backend('Agg')  # Non-interactive backend
fig, ax = plt.subplots(figsize=(10, 4))

# Plot ratio
ax.plot(miss_rate, ratio, 'o-', linewidth=2, markersize=8, color='#2563eb')

# Add horizontal line at ratio = 1 (where they are equal)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Equal performance')

# Styling
ax.set_xlabel('Miss Rate (%)', fontsize=12)
ax.set_ylabel('TTFT ratio (pass-KV / pass-Q)', fontsize=12)
ax.set_title('TTFT Ratio: pass-KV / pass-Q (P + T = 128000, CP4)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Set axis limits and log scale for x-axis
ax.set_xscale('log')
ax.set_xlim(0.8, 150)

# Fill regions to show which is better
ax.fill_between(miss_rate, ratio, 1, where=(ratio < 1), alpha=0.3, color='green', label='pass-KV faster')
ax.fill_between(miss_rate, ratio, 1, where=(ratio > 1), alpha=0.3, color='red', label='pass-Q faster')
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('ttft_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('ttft_comparison.pdf', dpi=300, bbox_inches='tight')
print("Saved ttft_comparison.pdf")

