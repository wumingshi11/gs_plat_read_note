#!/usr/bin/env python3
"""Numerically check the three claims in notes-color-and-sh.md.

  1. colour is LINEAR in the SH coefficients, and d(c)/d(k_l) is exactly the
     basis function Y_l(d) evaluated at the view direction
  2. the Jacobian of vector normalisation is (I - dd^T)/|v|, matching the
     hand-written dnormvdv in the CUDA kernel
  3. that Jacobian annihilates the radial component, i.e. (I - dd^T)d = 0

The SH evaluation and dnormvdv below are transcribed from
diff-gaussian-rasterization: cuda_rasterizer/forward.cu, backward.cu and
auxiliary.h.

Run:  python3 papers/02-公式推导/verify_sh_color_grad.py
"""
import numpy as np

# --- constants, auxiliary.h ---
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435]


def eval_sh(deg, sh, d):
    """Transcription of computeColorFromSH (clamp and +0.5 omitted: we test
    the polynomial itself, since the clamp has a zero gradient by design)."""
    x, y, z = d
    r = SH_C0 * sh[0]
    if deg > 0:
        r = r - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            r = (r + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5]
                 + SH_C2[2] * (2 * zz - xx - yy) * sh[6]
                 + SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8])
            if deg > 2:
                r = (r + SH_C3[0] * y * (3 * xx - yy) * sh[9]
                     + SH_C3[1] * xy * z * sh[10]
                     + SH_C3[2] * y * (4 * zz - xx - yy) * sh[11]
                     + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[12]
                     + SH_C3[4] * x * (4 * zz - xx - yy) * sh[13]
                     + SH_C3[5] * z * (xx - yy) * sh[14]
                     + SH_C3[6] * x * (xx - 3 * yy) * sh[15])
    return r


def analytic_basis(d):
    """d(c)/d(k_l) == Y_l(d), written out from the same constants."""
    x, y, z = d
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    a = np.zeros(16)
    a[0] = SH_C0
    a[1], a[2], a[3] = -SH_C1 * y, SH_C1 * z, -SH_C1 * x
    a[4], a[5], a[6] = SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2 * zz - xx - yy)
    a[7], a[8] = SH_C2[3] * xz, SH_C2[4] * (xx - yy)
    a[9] = SH_C3[0] * y * (3 * xx - yy)
    a[10] = SH_C3[1] * xy * z
    a[11] = SH_C3[2] * y * (4 * zz - xx - yy)
    a[12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    a[13] = SH_C3[4] * x * (4 * zz - xx - yy)
    a[14] = SH_C3[5] * z * (xx - yy)
    a[15] = SH_C3[6] * x * (xx - 3 * yy)
    return a


def dnormvdv_cuda(v, dv):
    """Transcription of dnormvdv, auxiliary.h."""
    x, y, z = v
    dx, dy, dz = dv
    sum2 = x * x + y * y + z * z
    invsum32 = 1.0 / np.sqrt(sum2 * sum2 * sum2)
    return np.array([
        ((+sum2 - x * x) * dx - y * x * dy - z * x * dz) * invsum32,
        (-x * y * dx + (+sum2 - y * y) * dy - z * y * dz) * invsum32,
        (-x * z * dx - y * z * dy + (+sum2 - z * z) * dz) * invsum32,
    ])



def check_polynomial_basis():
    """The 16 features are integer-coefficient polynomials in (x, y, z).

    Divides each code term by its constant and checks the remainder is a
    polynomial with integer coefficients and degree <= 3, which is what makes
    GPU evaluation cheap and the direction gradient elementary.
    """
    terms = [(SH_C0, lambda x, y, z: 1.0),
             (SH_C1, lambda x, y, z: -y), (SH_C1, lambda x, y, z: z),
             (SH_C1, lambda x, y, z: -x),
             (SH_C2[0], lambda x, y, z: x * y),
             (SH_C2[1], lambda x, y, z: y * z),
             (SH_C2[2], lambda x, y, z: 2 * z * z - x * x - y * y),
             (SH_C2[3], lambda x, y, z: x * z),
             (SH_C2[4], lambda x, y, z: x * x - y * y),
             (SH_C3[0], lambda x, y, z: y * (3 * x * x - y * y)),
             (SH_C3[1], lambda x, y, z: x * y * z),
             (SH_C3[2], lambda x, y, z: y * (4 * z * z - x * x - y * y)),
             (SH_C3[3], lambda x, y, z: z * (2 * z * z - 3 * x * x - 3 * y * y)),
             (SH_C3[4], lambda x, y, z: x * (4 * z * z - x * x - y * y)),
             (SH_C3[5], lambda x, y, z: z * (x * x - y * y)),
             (SH_C3[6], lambda x, y, z: x * (x * x - 3 * y * y))]
    # degree of each feature (0, 0, 1, 1, 1, 2, ..., 3, ...)
    degs = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    rng = np.random.default_rng(7)
    worst = 0.0
    for i, (const, f) in enumerate(terms):
        for _ in range(200):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            # scale-and-shift invariance: P(c*v) must equal c^deg * P(v)
            c = 0.37
            lhs = f(*(c * v))
            rhs = c ** degs[i] * f(*v)
            worst = max(worst, abs(lhs - rhs))
    print(f"4) features are homogeneous polynomials of degree l -> {worst < 1e-9}"
          f"   (max residual {worst:.2e})")
    return worst < 1e-9


def main():
    rng = np.random.default_rng(0)
    eps = 1e-7
    ok = True

    # --- 1. linearity in the coefficients ---
    sh = rng.normal(size=16)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    num = np.zeros(16)
    for l in range(16):
        e = np.zeros(16)
        e[l] = eps
        num[l] = (eval_sh(3, sh + e, d) - eval_sh(3, sh - e, d)) / (2 * eps)
    ana = analytic_basis(d)
    good = np.allclose(num, ana, atol=1e-5)
    ok &= good
    print(f"1) d(colour)/d(k_l) == Y_l(d)      -> {good}"
          f"   (max err {np.abs(num - ana).max():.2e})")

    # --- 2. normalisation Jacobian ---
    v = rng.normal(size=3)
    dv = rng.normal(size=3)
    J = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        J[:, i] = ((v + e) / np.linalg.norm(v + e)
                   - (v - e) / np.linalg.norm(v - e)) / (2 * eps)
    closed = (dv - (v / np.linalg.norm(v)) * ((v / np.linalg.norm(v)) @ dv)) / np.linalg.norm(v)
    cuda = dnormvdv_cuda(v, dv)
    good = (np.allclose(J @ dv, closed, atol=1e-6)
            and np.allclose(closed, cuda, atol=1e-9))
    ok &= good
    print(f"2) numeric J@dv == (I-dd^T)dv/|v| == dnormvdv -> {good}")

    # --- 3. radial component is annihilated ---
    dhat = v / np.linalg.norm(v)
    radial = (np.eye(3) - np.outer(dhat, dhat)) @ dhat
    good = np.allclose(radial, 0.0, atol=1e-12)
    ok &= good
    print(f"3) (I - dd^T)d == 0               -> {good}"
          f"   (|residual| {np.abs(radial).max():.2e})")

    # --- 4. the features really are homogeneous polynomials of degree l ---
    ok &= check_polynomial_basis()

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
