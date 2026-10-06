# 颜色为什么可导：SH 参数化与两条梯度通路

> 配套阅读：
> - `gsplat-math-supplement.pdf` —— 式 (16) 给出 $\partial C_i(k)/\partial c_n(k) = \alpha_n T_n$，是本文 §4 的起点
> - `notes-densification-criterion.md` —— 稠密化判据；本文 §6 解释为什么它只用位置梯度的一条通路
> - `verify_sh_color_grad.py` —— 同目录，可运行脚本，数值验证本文 §3、§4、§5 的三项结论

## 0. 结论速览

- 颜色可导的原因**和位置/协方差完全不同**：后者靠手工推导雅可比（补充材料式 3、§3.4），颜色靠**参数化本身就是初等函数**，不需要任何特殊处理。
- 颜色是 **SH 系数对方向的低次多项式**：$c = 0.5 + \sum_{lm} Y_{lm}(\hat d)\,k_{lm}$。对系数**线性**，对方向是 **3 次多项式**。
- 因此**整张图像是 SH 系数的低次多项式** —— 没有 MLP、没有体渲染积分、没有采样。求导是纯代数操作。
- 颜色是**唯一没有激活函数的参数**（尺度 exp、旋转单位化、不透明度 sigmoid）。
- 两处不光滑，各有一处分段处理：**clamp**（梯度按 PyTorch ReLU 约定置零）与**方向单位化**（雅可比是径向投影算子）。
- ⚠️ 容易猜错：**方向的梯度确实回传到了 3D 位置**（不是当作常数截断）。于是位置有**两条梯度通路**，而稠密化**只取几何那条**。

---

## 1. 可导性来自三个结构性质

| 性质 | 数学 | 后果 |
|---|---|---|
| 基函数是方向的**多项式**（最高 3 次）| $Y_{lm}(\hat d)$ 对 $\hat d$ 各分量是多项式 | 偏导 `dRGBdx/dy/dz` 是初等式 |
| 颜色对系数**线性** | $\partial c/\partial k_{lm} = Y_{lm}(\hat d)$ | 无需推导，直接取基函数值 |
| α 混合对颜色**线性** | $C = \sum_n c_n \alpha_n T_n$ | $\partial C/\partial c_n = \alpha_n T_n$（式 16）|

三条合起来：**从 SH 系数到最终像素，整条链是初等代数**。

这与 NeRF 形成鲜明对比：

| 对比 | NeRF | 3DGS |
|---|---|---|
| 颜色函数 | MLP（多层非线性）| SH 多项式（显式解析）|
| 求导方式 | autograd 反传整张网络 | 手工初等偏导 |
| 可学参数 | 网络权重（百万级，共享）| 每高斯 48 个系数（独立）|
| 不光滑点 | 采样、位置编码 | clamp + 方向单位化（均有分段处理）|

**本质**：3DGS 把可导性**设计进了表示里** —— 用多项式代替神经网络，让求导精确、廉价、且梯度有明确物理含义。代价是表达能力受限（§7）。

---

## 2. 颜色模型

### 2.1 公式

$$c_n = \mathrm{clamp}_{[0,\infty)}\Big(0.5 + \sum_{l=0}^{L}\sum_{m=-l}^{l} Y_{lm}(\hat d_n)\, k_{nlm}\Big),\qquad \hat d_n = \frac{\mu_n - o}{\lVert \mu_n - o\rVert}$$

- $k_{nlm}$：**唯一可学的量**，每高斯每通道 16 个（$L=3$），共 $3 \times 16 = 48$ 个
- $\hat d_n$：从相机指向高斯的单位方向，**每高斯每视角一个**
- $o$：相机位置
- $+0.5$：偏置，让 $k=0$ 时输出中灰
- $\mathrm{clamp}$：保证非负

### 2.2 为什么是 16 项 / 48 个系数

$l$ 阶有 $2l+1$ 项，取到 $L=3$：

$$1 + 3 + 5 + 7 = 16 \text{ 项},\qquad 16 \times 3\ (\mathrm{RGB}) = 48 \text{ 个系数/高斯}$$

代码里展开的就是这 16 项（`forward.cu` 的 `computeColorFromSH`，常量见 `auxiliary.h`）。它们的**多项式形式见 §2.3**。

### 2.3 "多项式"具体长什么样

颜色是**两个向量的内积**：一个是只由视角决定的 **16 维特征向量** $\mathbf{P}(\hat d)$，一个是该高斯学到的 **16 维系数** $\mathbf{k}$：

$$c_n = \mathrm{clamp}\Big(\underbrace{0.5}_{\text{偏置}} + \mathbf{P}(\hat d_n)\cdot \mathbf{k}_n\Big)$$

**特征的每一项都是 $\hat d=(x,y,z)$ 的整系数多项式**（$P_0$ 是常数项 1）：

| $j$ | $\mathbf{P}_j(x,y,z)$ | 次数 | $\mathbf{P}_j(x,y,z)$ | 次数 |
|---|---|---|---|---|
| 0 | $1$ | 0 | 8 | $x^2-y^2$ | 2 |
| 1 | $-y$ | 1 | 9 | $3x^2y-y^3$ | 3 |
| 2 | $z$ | 1 | 10 | $xyz$ | 3 |
| 3 | $-x$ | 1 | 11 | $4yz^2-x^2y-y^3$ | 3 |
| 4 | $xy$ | 2 | 12 | $2z^3-3x^2z-3y^2z$ | 3 |
| 5 | $yz$ | 2 | 13 | $4xz^2-x^3-xy^2$ | 3 |
| 6 | $2z^2-x^2-y^2$ | 2 | 14 | $xz^2-y^2z$ | 3 |
| 7 | $xz$ | 2 | 15 | $x^3-3xy^2$ | 3 |

（已用 sympy 逐项与解析实球谐比对：见 §8 脚本，比值全部是常数，即二者只差一个可被系数吸收的标定。）

**所以"用多项式表示颜色"的完整含义是：**

1. **算 3 个分量**：$\hat d = (\mu-o)/\lVert\mu-o\rVert$，一个单位向量
2. **算 16 个特征**：把上表的整系数多项式在该方向求值 → 16 维向量
3. **内积**：与学到的 16 个系数点乘
4. **加偏置 0.5，clamp 到非负**

**这不是泰勒展开，而是球面上的正交基展开** —— 与傅里叶级数同构，只是基从 $\{e^{ik\theta}\}$ 换成了球谐 $\{Y_{lm}(\theta,\phi)\}$。$l$ 就是"频率"：

| 阶 $l$ | 项数 | 表达什么 | 类比 |
|---|---|---|---|
| 0 | 1 | 一个常数色 | 均值 |
| 1 | 3 | 沿三个轴的线性变化 | 一阶梯度 |
| 2 | 5 | 二次变化（含 $x^2-y^2$ 这类鞍形）| 二阶项 |
| 3 | 7 | 三次变化 | 三阶项 |

**为什么这直接带来可导性：**

- **对系数**：内积对系数是线性的 → $\partial c/\partial k_j = \mathbf{P}_j(\hat d)$，**就是特征本身**。不需要求导运算，把前向算出的特征拿来做梯度即可（§8 脚本验证项①，误差 $4.3\times10^{-10}$）。
- **对方向**：多项式求导是初等式 → 代码里那堆 `dRGBdx` 就是逐项对 $x,y,z$ 求偏导的结果。例如 $j=6$ 项 $2z^2-x^2-y^2$ 贡献 $\partial/\partial x = -2x$、$\partial/\partial z = 4z$。

**为什么用整系数多项式**：GPU 上省浮点乘法。归一化常量（$0.28209$、$0.48860$ …）全部被折进系数里，基函数只剩整数乘加 —— 这也是为什么第 3 阶的常量比标准球谐小一倍、奇数阶差一个符号（这些差异都被可学系数吸收了，函数空间完全相同）。

### 2.4 参数形状

| 张量 | 形状 | 含义 |
|---|---|---|
| `_features_dc` | `(N, 1, 3)` | $l=0$ 的 1 项 × RGB |
| `_features_rest` | `(N, 15, 3)` | $l=1..3$ 的 15 项 × RGB |
| 合计 | $N \times 48$ | 每高斯 48 个可学常数 |

**内存代价**：48 个 float = 192 B/高斯。100 万高斯即 **≈ 192 MB** —— 远超位置（12 B）和不透明度（4 B），是各参数中最大的一项。这也是后续压缩类工作（本目录 `compgs-vector-quantization.pdf`、`compact-3d-gaussian-2311.13681.pdf`）的主要下手对象。

### 2.5 唯一没有激活函数的参数

| 参数 | 激活 | 源码 |
|---|---|---|
| 尺度 $s$ | $\exp$（保证正）| `scaling_activation = torch.exp` |
| 旋转 $q$ | 单位化（保证合法旋转）| `rotation_activation = F.normalize` |
| 不透明度 $\alpha$ | $\sigma$（保证 $[0,1)$）| `opacity_activation = torch.sigmoid` |
| **颜色（SH 系数）** | **无**（恒等）| 存的就是系数本身 |

没有激活 → 系数可取任意实数 → **负值只能靠前向 clamp 兜住**（§5）。

---

## 3. 前向实现

`forward.cu`（逐字对应上面的公式）：

```cuda
glm::vec3 dir = pos - campos;
dir = dir / glm::length(dir);                    // 单位化

glm::vec3 result = SH_C0 * sh[0];
if (deg > 0) {
    result = result - SH_C1*y*sh[1] + SH_C1*z*sh[2] - SH_C1*x*sh[3];
    if (deg > 1) { ... SH_C2 ... }
}
result += 0.5f;

// RGB colors are clamped to positive values. If values are
// clamped, we need to keep track of this for the backward pass.
clamped[3*idx + 0] = (result.x < 0);
clamped[3*idx + 1] = (result.y < 0);
clamped[3*idx + 2] = (result.z < 0);

return glm::max(result, 0.0f);
```

注意：**每个高斯每视角只算一次颜色**（在 `rgb[idx*C + ch]`），然后这个值被它覆盖的所有像素复用。所以后面的 clamp 决定也是**每高斯每视角**一个，不是每像素一个。

---

## 4. α 混合对颜色是线性的

补充材料式 (16)：

$$\frac{\partial C_i(k)}{\partial c_n(k)} = \alpha_n \, T_{n,i}$$

代码里就是那个 `dchannel_dcolor`（`backward.cu`）：

```cuda
const float dchannel_dcolor = alpha * T;
atomicAdd(&dL_dcolors[global_id * C + ch], dchannel_dcolor * dL_dchannel);
```

**因为是线性，颜色的梯度是最"干净"的一条**：不涉及任何几何量（没有协方差、没有雅可比、没有投影），只有可见性权重 $\alpha T$。这也是为什么颜色收敛比几何快得多 —— 它的梯度不受投影近似的误差影响。

汇总方式和位置一样是**对所有覆盖像素求和**（`atomicAdd`）：

$$\frac{\partial L}{\partial c_n} = \sum_i \alpha_n T_{n,i}\,\frac{\partial L}{\partial C_i}$$

同理是 $O(n)$ 的输出（每高斯一个 3 维梯度），见 `notes-densification-criterion.md` §2.1。

---

## 5. 两处不光滑，各有一处分段处理

### 5.1 clamp：梯度按 PyTorch ReLU 约定置零

前向把"是否被夹"记录下来，反传据此切断：

```cuda
glm::vec3 dL_dRGB = dL_dcolor[idx];
dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;
```

**为什么必须这样做**：一旦某通道被夹到 0，输出就**不再随系数变化**，$\partial c/\partial k$ 在数学上就是零。若仍把残差回传，优化器会把系数朝着"让输出更负"的方向推 —— 越推越被夹住，形成**饱和死区**。置零等于承认"这个方向暂时拿不到信息"。

这是 PyTorch 的 `ReLU` 约定（$\mathrm{ReLU}'(0)=0$），注释里也写明了：

```cuda
// Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0.
```

### 5.2 方向单位化：雅可比是径向投影算子

$$\frac{\partial \hat d}{\partial d} = \frac{\mathbf{I} - \hat d\hat d^\top}{\lVert d\rVert}$$

**几何含义**：单位化只关心方向、不关心长度，所以沿 $d$ 的**径向分量必须被完全抹掉** —— $(\mathbf{I}-\hat d\hat d^\top)\hat d = 0$（§8 脚本验证项③）。

实现为 `auxiliary.h` 的 `dnormvdv`：

```cuda
float sum2 = v.x*v.x + v.y*v.y + v.z*v.z;
float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
out.x = ((+sum2 - v.x*v.x)*dv.x - v.y*v.x*dv.y - v.z*v.x*dv.z) * invsum32;
out.y = (-v.x*v.y*dv.x + (+sum2 - v.y*v.y)*dv.y - v.z*v.y*dv.z) * invsum32;
out.z = (-v.x*v.z*dv.x - v.y*v.z*dv.y + (+sum2 - v.z*v.z)*dv.z) * invsum32;
```

---

## 6. ⚠️ 方向的梯度确实回传到了 3D 位置

**这是一个容易猜错的点。** 方向上由相机位置和高斯中心定义，视觉上像"外部输入"，容易以为梯度在此截断。**代码明确否定了这个猜测**（`backward.cu` 的 `computeColorFromSH` 结尾）：

```cuda
// The view direction is an input to the computation. View direction
// is influenced by the Gaussian's mean, so SHs gradients
// must propagate back into 3D position.
glm::vec3 dL_ddir(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));

// Account for normalization of direction
float3 dL_dmean = dnormvdv(dir_orig, dL_ddir);

// Gradients of loss w.r.t. Gaussian means, but only the portion
// that is caused because the mean affects the view-dependent color.
dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
```

所以 3D 位置的**总**梯度有**两条通路**：

| 通路 | 表达式 | 物理含义 | 进稠密化判据？ |
|---|---|---|---|
| 几何（屏幕位置）| $\partial L/\partial\mu' \cdot J^\top$ | **移动改变覆盖哪里** | ✅ 是（`dL_dmean2D`）|
| 外观（视角相关颜色）| $\mathrm{dnormvdv}\cdot\partial L/\partial\hat d$ | **移动改变从哪个角度看** | ❌ 否（`dL_dmeans`）|

代码里两者汇入**不同的累加器**，稠密化只读前者。

**这个区分很关键**：如果判据把第二条通路也算进去，那镜面高光、视角相关反射会不断制造大的位置梯度，让球被反复克隆 —— 而克隆对"从这个角度看颜色不对"毫无帮助。所以**稠密化只听"几何分配"那一支**，是刻意的设计选择，不是疏漏。

---

## 7. SH 的两类优化问题（论文自己承认）

### 7.1 高阶系数需要足够的角向信息

论文 Sec. 7.1：

> SH coefficient optimization is sensitive to the lack of angular information.

高阶 $Y_{lm}$ 的系数只在**多个视角**下才能被约束；视角少时它们会去拟合噪声（尤其是训练视角附近的高频伪影）。

**对策：渐进升阶**（`train.py`）：

```python
# Every 1000 its we increase the levels of SH up to a maximum degree
if iteration % 1000 == 0:
    gaussians.oneupSHdegree()
```

从 0 阶（只有常数项，等价于"处处同色"）开始，每 1000 次迭代加一阶，最多到 3 阶。这让几何先粗略成形，再逐级学习视角相关外观 —— 相当于一种**从低频到高频的课程学习**。

### 7.2 SH 的取值区间受限，亮色要靠高阶项硬撑

$Y_{00}$ 是**固定常数**（$0.28209479$），所以只有 0 阶时：

$$c = 0.5 + 0.28209479\,k_0$$

| 目标输出 | 需要的 $k_0$ |
|---|---|
| 0.5（中灰）| 0 |
| 1.0（饱和）| $\approx 1.772$ |
| 0.0（全黑）| $\approx -1.772$ |
| 不被 clamp 的下界 | $k_0 \ge -1.772$ |

**常数偏置 +0.5 意味着 0 阶项无法表示任意亮度**：要让白更白、黑更黑，只能靠高阶项在同一方向上叠加贡献。当目标亮度超出该视角下所有基函数能提供的范围时，clamp 就会触发，**该通道梯度被置零**（§5.1）。

于是两个问题是**耦合**的：视角少 → 高阶欠约束 → 亮色靠高阶硬撑 → clamp 频繁触发 → 梯度丢失。这解释了为什么 SH 相关的问题在实践中总是同时出现。

---

## 8. 数值验证

运行 `python3 papers/02-公式推导/verify_sh_color_grad.py`，验证四项：

| 项 | 结论 |
|---|---|
| ① $\partial c/\partial k_l$ | 数值差分 == 解析基函数值 $Y_l(\hat d)$，最大误差 $4.3\times10^{-10}$ |
| ② 单位化雅可比 | 数值 $J\,dv$ == 闭式 $(\mathbf{I}-\hat d\hat d^\top)dv/\lVert v\rVert$ == CUDA `dnormvdv`，三者一致 |
| ③ 径向分量 | $(\mathbf{I}-\hat d\hat d^\top)\hat d = \mathbf{0}$，即沿方向缩放不改变 $\hat d$ |
| ④ §2.3 的 16 个特征 | 均为 $l$ 次**齐次多项式**（$\mathbf{P}(c\hat d)=c^{l}\mathbf{P}(\hat d)$），最大残差 $8.3\times10^{-17}$ |

另外 §2.3 的多项式表已用 sympy 与解析实球谐逐项比对过：**16 项的比值全部是常数**，说明代码基函数与标准球谐只差一个可被系数吸收的标定（奇数阶差符号、第 3 阶差 2 倍），**函数空间完全相同**。

---

## 9. 参考

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023. Sec. 4（SH 表示）、Sec. 7.1（渐进升阶、角向敏感性）
- Ye & Kanazawa, *Mathematical Supplement for the gsplat Library*, arXiv:2312.02121. §3.1 式 (16)（颜色梯度）
- Zhang et al., *Differentiable Point-Based Radiance Fields for Efficient View Synthesis*, 2022（CUDA SH 实现的出处，见 `forward.cu` 注释）
- 代码：`graphdeco-inria/diff-gaussian-rasterization`（`cuda_rasterizer/forward.cu`、`backward.cu`、`auxiliary.h`）、`graphdeco-inria/gaussian-splatting`（`scene/gaussian_model.py`、`train.py`）
