#!/usr/bin/env python3
"""Verify that diff-gaussian-rasterization's backward pass builds J^T.

The kernel in cuda_rasterizer/backward.cu maps the screen-space mean gradient
dL/dmu' (2 components) to the 3D mean gradient dL/dmu (3 components) with a
hand-expanded 3x2 matrix. Those hand-written lines are exactly the transpose
of the projection Jacobian J from the gsplat math supplement, Eq. (3).

Run:  python3 papers/02-公式推导/verify_backward_jacobian.py
"""
import numpy as np

rng = np.random.default_rng(1)
# proj = W @ P flattened, indexed the way the CUDA kernel does it
P = rng.normal(size=16)
m = rng.normal(size=3)


def proj4(P, m):
    """glm-style column-major access, matching transformPoint4x4 in the kernel."""
    return np.array([
        P[0] * m[0] + P[4] * m[1] + P[8] * m[2] + P[12],
        P[1] * m[0] + P[5] * m[1] + P[9] * m[2] + P[13],
        P[2] * m[0] + P[6] * m[1] + P[10] * m[2] + P[14],
        P[3] * m[0] + P[7] * m[1] + P[11] * m[2] + P[15],
    ])


def screen(P, m):
    """NDC screen coords: perspective divide only (ndc2Pix is a linear rescale)."""
    h = proj4(P, m)
    return h[:2] / h[3]


# --- numeric Jacobian  J = d(u,v)/d(x,y,z)  (2x3) ---
J = np.zeros((2, 3))
for i in range(3):
    e = np.zeros(3)
    e[i] = 1e-7
    J[:, i] = (screen(P, m + e) - screen(P, m - e)) / 2e-7

# --- the matrix the CUDA kernel actually builds (backward.cu, dL_dmean.*) ---
m_hom = proj4(P, m)
m_w = 1.0 / (m_hom[3] + 1e-7)
mul1 = (P[0] * m[0] + P[4] * m[1] + P[8] * m[2] + P[12]) * m_w * m_w
mul2 = (P[1] * m[0] + P[5] * m[1] + P[9] * m[2] + P[13]) * m_w * m_w
A = np.array([
    [P[0] * m_w - P[3] * mul1, P[1] * m_w - P[3] * mul2],
    [P[4] * m_w - P[7] * mul1, P[5] * m_w - P[7] * mul2],
    [P[8] * m_w - P[11] * mul1, P[9] * m_w - P[11] * mul2],
])

np.set_printoptions(precision=5, suppress=True)
print("numeric J = d(u,v)/d(x,y,z):\n", J)
print("\nkernel-built matrix A:\n", A)
ok = np.allclose(A, J.T, atol=1e-4)
print(f"\nA == J^T ? -> {ok}   (max abs error {np.abs(A - J.T).max():.2e})")
raise SystemExit(0 if ok else 1)
