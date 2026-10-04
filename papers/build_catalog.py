#!/usr/bin/env python3
"""Emit papers/../CATALOG.md and papers_index.tsv from the real content of every
PDF (first-page text + page count), so the catalogue can never disagree with
the files on disk."""
import glob
import os
import re
import subprocess
import unicodedata

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(ROOT, "papers")

NOTES = {
    "3dgs-original-Kerbl2023-SIGGRAPH": "原始论文, SIGGRAPH 2023 最佳论文。核心: 3D 高斯显式表示 + 可微 tile 光栅化 + 自适应密度控制。**公式推导在补充材料里** (见 02-公式推导)。",
    "gsplat-math-supplement": "★ 你要的「公式推导补充 PDF」。Vickie Ye (UC Berkeley) 为 gsplat 写的数学补充: 第 2 节投影/光栅化前向 (式(1)-(4) 即 3D 协方差 → 2D 协方差 Σ'=J W Σ Wᵀ Jᵀ), 第 3 节完整反向传播梯度推导 (对 μ, Σ, 四元数, 尺度, 不透明度, SH 系数)。6 页纯推导, 是最权威的官方级推导文档。",
    "gsplat-open-source-library": "gsplat 库官方论文 (arXiv 2409.06765), 含 benchmark、实现细节与数学约定。仓库代码见同级 gsplat/ 目录。",
    "3dgs-error-analysis-optimal-projection": "对 3DGS 投影近似的误差分析, 给出最优投影策略 (改进 EWA 近似的推导)。",
    "does-3dgs-need-accurate-volumetric-rendering": "系统梳理 3DGS 相对体渲染理论的各项近似假设, 逐条做数学分析 (Eurographics 2025)。",
    "EWA-splatting-Zwicker2002-TVCG": "EWA Splatting 原始论文 (MERL TR2002-49 / IEEE TVCG 2002)。3DGS 投影公式 Σ'=JWΣWᵀJᵀ 与雅可比 J 的出处, 即式(3)(4) 的源头。注意: arXiv 上的 cs/0108002 实际是另一篇论文, 已核实并弃用。",
    "3dgs-2dgs-projection-derivation": "2DGS: 把 3D 高斯压成 2D 面元, 含投影与射线-面元求交的推导。",
    "gaussian-splatting-survey-3dgs-4d": "《Recent Advances in 3D Gaussian Splatting》综述 (Computational Visual Media), 分类梳理加速/压缩/动态/生成等方向。",
    "compact-3d-gaussian-2311.13681": "Compact 3D Gaussian Representation for Radiance Field (CVPR 2024): 紧凑高斯 + 神经高斯颜色场, 体积减 20x 以上。",
    "compgs-vector-quantization": "CompGS: 用向量量化压缩高斯属性 (CVPR 2024)。",
    "gaussian-opacity-fields": "GOF: 基于不透明度场的自适应表面重建, 含不透明度梯度推导。",
    "sugar": "SuGaR: 表面对齐高斯 + 网格提取。",
    "scaffold-gs": "Scaffold-GS: 锚点 + 视角自适应 MLP 生成高斯。",
    "3dgs-ray-tracing": "3D Gaussian Ray Tracing (TOG 2024)。",
    "3dgs-mcmc-alt": "3DGS as MCMC: 把致密化重解释为随机梯度哈密顿蒙特卡洛采样。",
    "3d-student-splatting-scooping": "3D Student Splatting and Scooping: 用 Student-t 分布替代高斯 (对初始化和噪声更鲁棒)。",
    "stopthepop": "StopThePop: 逐像素深度排序, 解决 tile 级排序的视角不一致。",
    "mip-splatting-aliasing-free-3dgs": "Mip-Splatting: 3D 平滑滤波 + 2D Mip 滤波消除混叠。",
    "gaussianavatars": "GaussianAvatars: 绑定骨架的头部高斯化身。",
    "gaussianeditor": "GaussianEditor: 高斯场景编辑。",
    "text-to-3d-dreamgaussian": "DreamGaussian: SDS + 高斯, 文本/图像到 3D。",
    "text-to-3d-gaussiandreamer": "GaussianDreamer: 桥接 2D/3D 扩散模型生成高斯。",
    "3dgs-relightable-gaussian-splatting": "Relightable 3D Gaussians: BRDF 分解 + 光线追踪阴影。",
    "4d-gaussian-splatting": "4D-GS: 动态场景, 4D 时空高斯 + 可变形场。",
    "dynamic-3d-gaussians": "Dynamic 3D Gaussians: 持久动态视图合成与跟踪。",
    "gaussian-splatting-large-scale": "GRM: 大规模高斯重建/生成模型。",
    "nerf-original": "NeRF 原始论文 (前身工作)。",
    "mip-nerf-360": "Mip-NeRF 360 (无界场景, 3DGS 的主要对比基线)。",
    "plenoxels": "Plenoxels: 无数神经网络的辐射场。",
    "instant-ngp": "Instant-NGP: 多分辨率哈希编码。",
}
ORDER = ["01-核心论文", "02-公式推导", "03-前身工作", "04-加速与质量", "05-压缩存储", "06-动态与生成"]
FALLBACK_ORDER = "99-其他"


def page_text(path, pages="1"):
    r = subprocess.run(["pdftotext", "-f", "1", "-l", "1", "-layout", path, "-"],
                       capture_output=True, text=True, timeout=60)
    out = []
    for line in r.stdout.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        if len(s) > 8:
            out.append(s)
        if len(out) >= 4:
            break
    return " | ".join(out)


def npages(path):
    r = subprocess.run(["pdfinfo", path], capture_output=True, text=True, timeout=30)
    m = re.search(r"^Pages:\s*(\d+)", r.stdout, re.M)
    return int(m.group(1)) if m else 0


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(PAPERS, "*", "*.pdf"))):
        cat = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        if cat == FALLBACK_ORDER:
            continue
        rows.append({"cat": cat, "stem": stem, "path": path, "size": os.path.getsize(path),
                     "pages": npages(path), "head": page_text(path)})
    rows.sort(key=lambda r: (ORDER.index(r["cat"]) if r["cat"] in ORDER else 99, r["stem"]))

    with open(os.path.join(ROOT, "papers_index.tsv"), "w") as f:
        f.write("category\tfile\tpages\tsize_bytes\tfirst_page_title\n")
        for r in rows:
            f.write(f"{r['cat']}\t{r['stem']}.pdf\t{r['pages']}\t{r['size']}\t{r['head']}\n")

    lines = ["# 3DGS 论文资料库", "",
             f"共 {len(rows)} 篇 PDF, 位于 `papers/<分类>/`。索引见 `papers_index.tsv`。", ""]
    for cat in ORDER:
        group = [r for r in rows if r["cat"] == cat]
        if not group:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        for r in group:
            note = NOTES.get(r["stem"], "")
            lines.append(f"- **`{r['stem']}.pdf`** ({r['pages']} 页, {r['size']/1e6:.1f} MB)  ")
            lines.append(f"  {r['head'][:150]}  ")
            if note:
                lines.append(f"  > {note}")
        lines.append("")
    with open(os.path.join(ROOT, "CATALOG.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote CATALOG.md / papers_index.tsv with {len(rows)} papers")
    for r in rows:
        print(f"  [{r['cat']}] {r['stem']:<46} {r['pages']:>2}p {r['size']/1e6:>6.1f}MB")


if __name__ == "__main__":
    main()
