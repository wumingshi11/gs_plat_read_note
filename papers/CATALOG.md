# 3DGS 论文资料库

共 30 篇 PDF, 位于 `papers/<分类>/`。索引见 `papers_index.tsv`。

## 01-核心论文

- **`3dgs-original-Kerbl2023-SIGGRAPH.pdf`** (14 页, 1.2 MB)  
  3D Gaussian Splatting for Real-Time Radiance Field Rendering | BERNHARD KERBL∗ , Inria, Universite Cote d’Azur, France | GEORGIOS KOPANAS∗ , Inria, Un  
  > 原始论文, SIGGRAPH 2023 最佳论文。核心: 3D 高斯显式表示 + 可微 tile 光栅化 + 自适应密度控制。**公式推导在补充材料里** (见 02-公式推导)。
- **`gaussian-splatting-survey-3dgs-4d.pdf`** (28 页, 31.3 MB)  
  Computational Visual Media | https://doi.org/10.1007/s41095-0xx-xxxx-x | Research/Review Article | Recent Advances in 3D Gaussian Splatting  
  > 《Recent Advances in 3D Gaussian Splatting》综述 (Computational Visual Media), 分类梳理加速/压缩/动态/生成等方向。
- **`gsplat-open-source-library.pdf`** (17 页, 0.5 MB)  
  gsplat: An Open-Source Library for Gaussian Splatting | gsplat: An Open-Source Library for Gaussian Splatting | Vickie Ye1,† vye@berkeley.edu | Ruilon  
  > gsplat 库官方论文 (arXiv 2409.06765), 含 benchmark、实现细节与数学约定。仓库代码见同级 gsplat/ 目录。

## 02-公式推导

- **`3dgs-2dgs-projection-derivation.pdf`** (13 页, 24.8 MB)  
  2D Gaussian Splatting for Geometrically Accurate Radiance Fields | BINBIN HUANG, ShanghaiTech University, China | ZEHAO YU, University of Tubingen Tub  
  > 2DGS: 把 3D 高斯压成 2D 面元, 含投影与射线-面元求交的推导。
- **`3dgs-error-analysis-optimal-projection.pdf`** (25 页, 18.8 MB)  
  On the Error Analysis of 3D Gaussian Splatting | and an Optimal Projection Strategy | Letian Huang , Jiayang Bai , Jie Guo⋆ , Yuanqi Li , and Yanwen G  
  > 对 3DGS 投影近似的误差分析, 给出最优投影策略 (改进 EWA 近似的推导)。
- **`EWA-splatting-Zwicker2002-TVCG.pdf`** (18 页, 2.8 MB)  
  MITSUBISHI ELECTRIC RESEARCH LABORATORIES | http://www.merl.com | EWA Splatting | M. Zwicker, H. Pfister, J. van Baar, M. Gross  
  > EWA Splatting 原始论文 (MERL TR2002-49 / IEEE TVCG 2002)。3DGS 投影公式 Σ'=JWΣWᵀJᵀ 与雅可比 J 的出处, 即式(3)(4) 的源头。注意: arXiv 上的 cs/0108002 实际是另一篇论文, 已核实并弃用。
- **`does-3dgs-need-accurate-volumetric-rendering.pdf`** (50 页, 6.1 MB)  
  EUROGRAPHICS 2025 / A. Bousseau and A. Dai COMPUTER GRAPHICS forum | (Guest Editors) Volume 44 (2025), Number 2 | Does 3D Gaussian Splatting Need Accu  
  > 系统梳理 3DGS 相对体渲染理论的各项近似假设, 逐条做数学分析 (Eurographics 2025)。
- **`gsplat-math-supplement.pdf`** (6 页, 0.1 MB)  
  Mathematical Supplement for the gsplat Library | Vickie Ye Angjoo Kanazawa | UC Berkeley | arXiv:2312.02121v1 [cs.MS] 4 Dec 2023  
  > ★ 你要的「公式推导补充 PDF」。Vickie Ye (UC Berkeley) 为 gsplat 写的数学补充: 第 2 节投影/光栅化前向 (式(1)-(4) 即 3D 协方差 → 2D 协方差 Σ'=J W Σ Wᵀ Jᵀ), 第 3 节完整反向传播梯度推导 (对 μ, Σ, 四元数, 尺度, 不透明度, SH 系数)。6 页纯推导, 是最权威的官方级推导文档。

## 03-前身工作

- **`instant-ngp.pdf`** (15 页, 17.8 MB)  
  Instant Neural Graphics Primitives with a Multiresolution Hash Encoding | THOMAS MULLER, NVIDIA, Switzerland | ALEX EVANS, NVIDIA, United Kingdom | CH  
  > Instant-NGP: 多分辨率哈希编码。
- **`mip-nerf-360.pdf`** (18 页, 10.7 MB)  
  Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields | Jonathan T. Barron1 Ben Mildenhall1 Dor Verbin1,2 | Pratul P. Srinivasan1 Peter Hedman1   
  > Mip-NeRF 360 (无界场景, 3DGS 的主要对比基线)。
- **`nerf-original.pdf`** (25 页, 8.3 MB)  
  NeRF: Representing Scenes as | arXiv:2003.08934v2 [cs.CV] 3 Aug 2020 | Neural Radiance Fields for View Synthesis | Ben Mildenhall1? Pratul P. Srinivas  
  > NeRF 原始论文 (前身工作)。
- **`plenoxels-radiance-fields-without-neural-networks.pdf`** (21 页, 3.5 MB)  
  Plenoxels: Radiance Fields without Neural Networks | Alex Yu∗ Sara Fridovich-Keil∗ Matthew Tancik Qinhong Chen | Benjamin Recht Angjoo Kanazawa | UC B  

## 04-加速与质量

- **`3d-student-splatting-scooping.pdf`** (24 页, 22.6 MB)  
  3D Student Splatting and Scooping | Jialin Zhu1 , Jiangbei Yue2 , Feixiang He1 , He Wang 1,3 * | University College London, UK 2 University of Leeds,   
  > 3D Student Splatting and Scooping: 用 Student-t 分布替代高斯 (对初始化和噪声更鲁棒)。
- **`3dgs-mcmc.pdf`** (16 页, 16.6 MB)  
  3D Gaussian Splatting as | Markov Chain Monte Carlo | Shakiba Kheradmand1 , Daniel Rebain1 , Gopal Sharma1 , | Weiwei Sun1 , Yang-Che Tseng1 , Hossam   
- **`3dgs-ray-tracing.pdf`** (19 页, 2.6 MB)  
  3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes | NICOLAS MOENNE-LOCCOZ∗ , NVIDIA, Canada | ASHKAN MIRZAEI∗ , NVIDIA, Canada and University o  
  > 3D Gaussian Ray Tracing (TOG 2024)。
- **`mip-splatting-aliasing-free-3dgs.pdf`** (19 页, 1.5 MB)  
  Mip-Splatting: Alias-free 3D Gaussian Splatting | Zehao Yu1,2 Anpei Chen1,2 Binbin Huang3 Torsten Sattler4 Andreas Geiger1,2 | University of Tubingen   
  > Mip-Splatting: 3D 平滑滤波 + 2D Mip 滤波消除混叠。
- **`stopthepop.pdf`** (17 页, 1.5 MB)  
  StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time | Rendering | LUKAS RADL∗ and MICHAEL STEINER∗ , Graz University of Technology, Au  
  > StopThePop: 逐像素深度排序, 解决 tile 级排序的视角不一致。

## 05-压缩存储

- **`compact-3d-gaussian-2311.13681.pdf`** (14 页, 11.7 MB)  
  Compact 3D Gaussian Representation for Radiance Field | Joo Chan Lee1 Daniel Rho2 Xiangyu Sun1 Jong Hwan Ko1B Eunbyung Park1B | Sungkyunkwan Universit  
  > Compact 3D Gaussian Representation for Radiance Field (CVPR 2024): 紧凑高斯 + 神经高斯颜色场, 体积减 20x 以上。
- **`compgs-vector-quantization.pdf`** (28 页, 20.5 MB)  
  CompGS: Smaller and Faster Gaussian Splatting | with Vector Quantization | K L Navaneet∗ Kossar Pourahmadi Meibodi⋆ | Soroush Abbasi Koohpayegani Hame  
  > CompGS: 用向量量化压缩高斯属性 (CVPR 2024)。
- **`gaussian-opacity-fields.pdf`** (15 页, 1.3 MB)  
  Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in | Unbounded Scenes | ZEHAO YU, University of Tubingen, Tubingen AI Center, Germa  
  > GOF: 基于不透明度场的自适应表面重建, 含不透明度梯度推导。
- **`scaffold-gs.pdf`** (14 页, 22.0 MB)  
  Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering | Tao Lu 1,3 * Mulin Yu1 * Linning Xu2 Yuanbo Xiangli4 | Limin Wang1,3 Dahua Lin1,2 B  
  > Scaffold-GS: 锚点 + 视角自适应 MLP 生成高斯。
- **`sugar.pdf`** (14 页, 26.2 MB)  
  SuGaR: Surface-Aligned Gaussian Splatting for | Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering | Antoine Guedon Vincent Lepetit | LI  
  > SuGaR: 表面对齐高斯 + 网格提取。

## 06-动态与生成

- **`3dgs-relightable-gaussian-splatting.pdf`** (17 页, 8.7 MB)  
  Relightable 3D Gaussians: Realistic Point Cloud | Relighting with BRDF Decomposition and Ray | Jian Gao1∗ , Chun Gu2∗ , Youtian Lin1 , Zhihao Li3 , Ha  
  > Relightable 3D Gaussians: BRDF 分解 + 光线追踪阴影。
- **`4d-gaussian-splatting.pdf`** (15 页, 4.4 MB)  
  4D Gaussian Splatting for Real-Time Dynamic Scene Rendering | Guanjun Wu1 *, Taoran Yi2 *, Jiemin Fang3†, Lingxi Xie3 , Xiaopeng Zhang3 , | Wei Wei1 ,  
  > 4D-GS: 动态场景, 4D 时空高斯 + 可变形场。
- **`dynamic-3d-gaussians.pdf`** (11 页, 10.2 MB)  
  Dynamic 3D Gaussians: | Tracking by Persistent Dynamic View Synthesis | Jonathon Luiten1,2 Georgios Kopanas3 Bastian Leibe2 Deva Ramanan1 | Carnegie M  
  > Dynamic 3D Gaussians: 持久动态视图合成与跟踪。
- **`gaussian-splatting-large-scale.pdf`** (29 页, 29.4 MB)  
  arXiv:2403.14621v1 [cs.CV] 21 Mar 2024 | GRM: Large Gaussian Reconstruction Model for | Efficient 3D Reconstruction and Generation | Yinghao Xu1⋆ , Zi  
  > GRM: 大规模高斯重建/生成模型。
- **`gaussianavatars.pdf`** (13 页, 4.1 MB)  
  GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians | Shenhan Qian1 Tobias Kirschstein1 Liam Schoneveld2 Davide Davoli3 | Simon Gieb  
  > GaussianAvatars: 绑定骨架的头部高斯化身。
- **`gaussianeditor.pdf`** (14 页, 0.9 MB)  
  GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting | Yiwen Chen*1,2 Zilong Chen*3,5 Chi Zhang2 Feng Wang3 Xiaofeng Yang2 | Yika  
  > GaussianEditor: 高斯场景编辑。
- **`text-to-3d-dreamgaussian.pdf`** (18 页, 8.0 MB)  
  Published as a conference paper at ICLR 2024 | D REAM G AUSSIAN : G ENERATIVE G AUSSIAN S PLAT- | TING FOR E FFICIENT 3D C ONTENT C REATION | Jiaxiang  
  > DreamGaussian: SDS + 高斯, 文本/图像到 3D。
- **`text-to-3d-gaussiandreamer.pdf`** (15 页, 1.6 MB)  
  GaussianDreamer: Fast Generation from Text to 3D Gaussians | by Bridging 2D and 3D Diffusion Models | Taoran Yi1 , Jiemin Fang2†, Junjie Wang2 , Guanj  
  > GaussianDreamer: 桥接 2D/3D 扩散模型生成高斯。
