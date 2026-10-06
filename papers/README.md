# 3DGS 论文资料库

**30 篇论文 PDF（305 MB）**，全部按 PDF 正文内容校验过（不是只看文件名或 arXiv 元数据），可直接配合 [`../gsplat`](../gsplat) 源码阅读。

```
papers/
├── CATALOG.md          # 全部论文清单（自动从 PDF 内容生成，含页数/大小/首页标题）
├── papers_index.tsv    # 表格版（分类 / 页数 / 大小 / 首页标题）
├── build_catalog.py    # 重新生成上面两个文件
├── 01-核心论文/         # 原始论文 + gsplat 库论文 + 综述
├── 02-公式推导/         # ★ 公式推导补充 PDF、推导类论文、自撰推导笔记
├── 03-前身工作/         # NeRF 系列、Plenoxels、Instant-NGP
├── 04-加速与质量/
├── 05-压缩存储/
└── 06-动态与生成/
```

> **体积说明**：>30 MB 的 8 篇已用 Ghostscript 压缩（150 DPI / JPEG q88），合计 324.9 MB → 13.4 MB。
> 文字仍是矢量（可搜索、任意缩放清晰），只有内嵌插图被降采样，已逐篇校验页数与文字层。
> 原始 3DGS 论文因此从 35.8 MB 降到 **1.12 MB**（缩小 32 倍），公式页完全清晰。
> 需要原始高清图时，按 `CATALOG.md` 里的 arXiv 编号重新下载即可。

---

## 一、原始论文

`01-核心论文/3dgs-original-Kerbl2023-SIGGRAPH.pdf`
**3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl, Kopanas, Leimkühler, Drettakis; SIGGRAPH 2023 最佳论文; arXiv [2308.04079](https://arxiv.org/abs/2308.04079), 14 页)

三条主线：① 用 3D 高斯椭球做显式场景表示；② 可微 tile 光栅化实现实时渲染；③ 自适应密度控制（克隆/分裂/剪枝）。

## 二、公式推导（重点）★

`02-公式推导/gsplat-math-supplement.pdf`
**Mathematical Supplement for the gsplat Library** (Vickie Ye, Angjoo Kanazawa; UC Berkeley; arXiv [2312.02121](https://arxiv.org/abs/2312.02121), 6 页纯推导)

这是目前最权威、最完整的一份 3DGS 数学推导文档，封面即写明 *"provide a self-contained reference for the computations involved in the forward and backward passes"*。结构：

| 章节 | 内容 |
|---|---|
| 1 Introduction | 符号约定（µ, Σ, c, o） |
| 2 Rasterization Forward Pass | 式(1) 相机内外参 T_cw 与投影矩阵 P |
| 2.1 Projection of Gaussians | 式(2) 均值投影 µ→µ′；**式(3) 雅可比 J**（引自 Zwicker et al. 2002）；**式(4) 协方差投影 Σ′ = J R_cw Σ R_cwᵀ Jᵀ**；由 scale s + 四元数 q 构造 Σ |
| 2.2 Depth Compositing | 深度排序 + 前向 α 混合 |
| 3 Computing Gradients | **反向传播完整推导** |
| 3.1 Depth Compositing Gradients | 对颜色 / 不透明度的梯度 |
| 3.2 Projection Gradients | 对 µ、Σ、s、q 的梯度链式展开 |
| 4 Conclusion | — |

同目录下另外三篇推导类论文可配套看：

- `EWA-splatting-Zwicker2002-TVCG.pdf` — **EWA Splatting** 原文（MERL TR2002-49 / IEEE TVCG 2002），上面式(3)(4) 的原始出处。
  ⚠️ 注意：arXiv 上的 `cs/0108002` 虽然标题登记为 "EWA Splatting"，**实际内容是另一篇论文**（Bounded Concurrent Timestamp Systems），已核实并弃用，改用 MERL 官方 PDF。
- `3dgs-error-analysis-optimal-projection.pdf` — 对 3DGS 投影近似的**误差分析** + 最优投影策略。
- `does-3dgs-need-accurate-volumetric-rendering.pdf` — 逐条分析 3DGS 相对体渲染理论的近似假设（Eurographics 2025）。
- `3dgs-2dgs-projection-derivation.pdf` — 2DGS 的投影与射线求交推导（Surfel 形式）。

> 说明：3DGS **官方**补充材料（Inria 的 `3d-gaussian-splatting-supplemental.pdf`）目前仓库已下线，多个镜像均 404，无法获取。其数学内容已被上面这份 gsplat 补充材料完整覆盖且更详细。

### 自撰推导笔记（补充材料没讲的部分）

补充材料覆盖前向投影与标准反传（式 1–31），但**不涉及高斯的增删，也不解释各参数为何可导**。两篇笔记补上这些缺口：

**① `notes-densification-criterion.md`** —《稠密化判据：为什么是位置梯度》

- 为什么需要稠密化（连续优化管不了离散的"个数"）
- 本质：混合模型的模型选择问题，用"位移的边际价值"当零成本代理
- 为什么判据是**高斯中心的屏幕位置梯度**，而不是"对 (u,v) 的导数"；计算量为何是 $O(n)$ 而非 $O(W \cdot H \cdot n)$
- 该代理的三个内生缺陷（深度失明 / 尺度偏置 / 与图像对比度混淆）与对应补丁
- 与自适应有限元、分裂-合并 EM 的同构；论文 vs 代码差异表

**② `notes-color-and-sh.md`** —《颜色为什么可导：SH 参数化与两条梯度通路》

- 颜色可导的来源与几何不同：SH 是**方向的多项式 × 系数的线性函数**，整张图像是系数的低次多项式
- 唯一没有激活函数的参数；两处不光滑点（clamp、方向单位化）及各自的分段处理
- ⚠️ 方向的梯度**确实回传**到 3D 位置 → 位置有两条梯度通路，而稠密化只取几何那条
- SH 的两类优化问题：高阶需角向信息（渐进升阶）、$Y_{00}$ 是常数导致亮色靠高阶硬撑

**可运行验证脚本**（同目录）：

- `verify_backward_jacobian.py` — 验证 `backward.cu` 手工展开的 3×2 矩阵是补充材料式 (3) 投影雅可比的转置
- `verify_sh_color_grad.py` — 验证 SH 对系数线性（$\partial c/\partial k_l = Y_l$）、单位化雅可比为 $(\mathbf{I}-\hat d\hat d^\top)/\lVert v\rVert$、径向分量被抹掉、16 个特征均为齐次多项式、初始化的 RGB↔SH 精确往返

## 三、对应源码

本仓库根目录的 `gsplat/` 是 [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) 的阅读笔记版（早期版本，未包含后续的 2DGS / 3DGUT / 压缩等模块），对应论文是 `01-核心论文/gsplat-open-source-library.pdf`（arXiv [2409.06765](https://arxiv.org/abs/2409.06765)）。

**读代码路线**（路径相对仓库根目录）：

| 目的 | 文件 |
|---|---|
| 最核心的纯 PyTorch 参考实现，**对着补充材料逐式对照就用这个** | `gsplat/_torch_impl.py` |
| ↳ 由 scale + 四元数构造 Σ | `_torch_impl.py` → `scale_rot_to_cov3d()` |
| ↳ **EWA 协方差投影 Σ′ = J W Σ Wᵀ Jᵀ** | `_torch_impl.py` → `project_cov3d_ewa()` |
| ↳ 投影主流程 | `_torch_impl.py` → `project_gaussians_forward()` |
| ↳ 前向 α 混合 | `_torch_impl.py` → `rasterize_forward()` |
| ↳ 球谐系数求值 | `_torch_impl.py` → `compute_sh_color()` / `eval_sh_bases()` |
| 投影 / 光栅化的 autograd 封装 | `gsplat/project_gaussians.py`、`gsplat/rasterize.py` |
| CUDA kernel（前向 / 反向 / 绑定） | `gsplat/cuda/csrc/forward.cu`、`backward.cu`、`bindings.cu` |
| 训练示例 | `examples/simple_trainer.py` |
| 相机 / 数据约定 | `docs/source/conventions/data_conventions.rst` |

## 四、其它分类

- **03-前身工作**：NeRF、Mip-NeRF 360、Plenoxels、Instant-NGP（理解 3DGS 的对比基线）
- **04-加速与质量**：Mip-Splatting（抗混叠）、StopThePop（逐像素排序）、3D Gaussian Ray Tracing、3DGS-MCMC、3D Student Splatting and Scooping
- **05-压缩存储**：Compact 3D Gaussian (CVPR'24)、CompGS（向量量化）、Gaussian Opacity Fields、Scaffold-GS、SuGaR（网格提取）
- **06-动态与生成**：4D-GS、Dynamic 3D Gaussians、GaussianAvatars、GaussianEditor、DreamGaussian、GaussianDreamer、Relightable 3D Gaussians、GRM

## 五、维护

在仓库根目录运行：

```bash
python3 papers/build_catalog.py                 # 从 PDF 实际内容重新生成 CATALOG.md 与 papers_index.tsv
python3 organize_papers.py                      # 按内容标题把 PDF 归档到分类目录
python3 shrink_papers.py --min-mb 30 --dry-run  # 看哪些 PDF 体积过大
python3 shrink_papers.py --min-mb 30            # 压缩（150 DPI，原文件删除）
python3 shrink_papers.py --restore              # 还原（仅对 --archive 压过的有效）
```

`CATALOG.md` 与 `papers_index.tsv` 中的「首页标题」全部从 PDF 正文提取（而非文件名或 arXiv 元数据），可与 arXiv ID 交叉核对，避免出现编号与内容不符的情况。
