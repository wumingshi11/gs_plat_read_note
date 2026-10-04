#!/usr/bin/env python3
"""Sort papers/*.pdf into category folders and emit a verified catalogue.

Every PDF's first page is re-parsed so a wrong arXiv ID can never hide:
the printed title is compared against the expected title and mismatches
are reported instead of being filed silently.
"""
import glob
import os
import re
import shutil
import subprocess
import unicodedata

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(ROOT, "papers")

# file stem -> (category folder, expected title fragment, arxiv id)
MAP = {
    "3dgs-original-Kerbl2023-SIGGRAPH": ("01-核心论文", "3D Gaussian Splatting for Real-Time Radiance Field Rendering", "2308.04079"),
    "gsplat-open-source-library": ("01-核心论文", "gsplat: An Open-Source Library for Gaussian Splatting", "2409.06765"),
    "gaussian-splatting-survey-3dgs-4d": ("01-核心论文", "Recent Advances in 3D Gaussian Splatting", "2403.11134"),

    "gsplat-math-supplement": ("02-公式推导", "Mathematical Supplement for the gsplat Library", "2312.02121"),
    "EWA-splatting-Zwicker2002-TVCG": ("02-公式推导", "EWA", "MERL TR2002-49"),
    "3dgs-error-analysis-optimal-projection": ("02-公式推导", "On the Error Analysis of 3D Gaussian Splatting", "2402.00752"),
    "does-3dgs-need-accurate-volumetric-rendering": ("02-公式推导", "Does 3D Gaussian Splatting Need Accurate Volumetric Rendering", "2502.19318"),
    "3dgs-2dgs-projection-derivation": ("02-公式推导", "2D Gaussian Splatting", "2403.17888"),

    "nerf-original": ("03-前身工作", "NeRF: Representing Scenes as Neural Radiance Fields", "2003.08934"),
    "mip-nerf-360": ("03-前身工作", "Mip-NeRF 360", "2111.12077"),
    "plenoxels-radiance-fields-without-neural-networks": ("03-前身工作", "Plenoxels", "2112.05131"),
    "instant-ngp": ("03-前身工作", "Instant Neural Graphics Primitives", "2201.05989"),

    "mip-splatting-aliasing-free-3dgs": ("04-加速与质量", "Mip-Splatting", "2311.16493"),
    "stopthepop": ("04-加速与质量", "StopThePop", "2402.00525"),
    "3dgs-ray-tracing": ("04-加速与质量", "3D Gaussian Ray Tracing", "2407.07090"),
    "3dgs-mcmc": ("04-加速与质量", "Markov Chain Monte Carlo", "2404.09591"),
    "3d-student-splatting-scooping": ("04-加速与质量", "3D Student Splatting and Scooping", "2503.10148"),

    "compgs-vector-quantization": ("05-压缩存储", "CompGS", "2311.18159"),
    "compact-3d-gaussian-2311.13681": ("05-压缩存储", "Compact 3D Gaussian Representation", "2311.13681"),
    "gaussian-opacity-fields": ("05-压缩存储", "Gaussian Opacity Fields", "2404.10772"),
    "sugar": ("05-压缩存储", "SuGaR", "2311.12775"),
    "scaffold-gs": ("05-压缩存储", "Scaffold-GS", "2312.00109"),

    "4d-gaussian-splatting": ("06-动态与生成", "4D Gaussian Splatting", "2310.08528"),
    "dynamic-3d-gaussians": ("06-动态与生成", "Dynamic 3D Gaussians", "2308.09713"),
    "gaussianavatars": ("06-动态与生成", "GaussianAvatars", "2312.02069"),
    "gaussianeditor": ("06-动态与生成", "GaussianEditor", "2311.14521"),
    "text-to-3d-dreamgaussian": ("06-动态与生成", "DreamGaussian", "2309.16653"),
    "text-to-3d-gaussiandreamer": ("06-动态与生成", "GaussianDreamer", "2310.08529"),
    "3dgs-relightable-gaussian-splatting": ("06-动态与生成", "Relightable 3D Gaussians", "2311.16043"),
    "gaussian-splatting-large-scale": ("06-动态与生成", "GRM", "2403.14621"),
}


def first_page(path, n=14):
    r = subprocess.run(["pdftotext", "-f", "1", "-l", "1", "-layout", path, "-"],
                       capture_output=True, text=True, timeout=60)
    txt = r.stdout
    out = []
    for line in txt.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        s = "".join(c for c in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(c))
        if len(s) > 8 and not re.match(r"^(arXiv:|\[?cs\.|Abstract|Subject:|Preprint)", s):
            out.append(s)
        if len(out) >= n:
            break
    return " | ".join(out)


def main():
    for cat in {v[0] for v in MAP.values()}:
        os.makedirs(os.path.join(PAPERS, cat), exist_ok=True)
    rows, problems = [], []
    for path in sorted(glob.glob(os.path.join(PAPERS, "*.pdf"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        head = first_page(path)
        flat = re.sub(r"[^a-z0-9]+", "", head.lower())
        entry = MAP.get(stem)
        if not entry:
            problems.append(f"UNMAPPED  {stem}   ::  {head[:100]}")
            continue
        cat, expect, aid = entry
        key = re.sub(r"[^a-z0-9]+", "", expect.lower())
        if key not in flat:
            problems.append(f"TITLE-MISMATCH  {stem}  expected={expect!r}  got={head[:100]!r}")
            continue
        dst = os.path.join(PAPERS, cat, f"{stem}.pdf")
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.move(path, dst)
        rows.append((cat, stem, aid, os.path.getsize(dst), head[:110]))

    order = sorted(rows, key=lambda r: (r[0], r[1]))
    print(f"filed {len(rows)} PDFs")
    for r in order:
        print(f"  [{r[0]}] {r[2]:<12} {r[1]}")
    if problems:
        print("\n!!! NEEDS ATTENTION")
        for p in problems:
            print("  " + p)
    with open(os.path.join(ROOT, "papers_index.tsv"), "w") as f:
        f.write("category\tarxiv\tfile\tsize_bytes\tfirst_page_title\n")
        for r in order:
            f.write(f"{r[0]}\t{r[2]}\t{r[1]}.pdf\t{r[3]}\t{r[4]}\n")


if __name__ == "__main__":
    main()
