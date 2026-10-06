# 稠密化判据：为什么是位置梯度

> 配套阅读：`gsplat-math-supplement.pdf`（同一目录）。本笔记回答的是那份补充材料**没有涉及**的部分。

## 0. 结论速览

- 补充材料覆盖了前向投影与标准反传（式 1–31），**完全不涉及高斯增删**：`densification / clone / split / prune / view-space / opacity reset` 等词在其中出现 0 次。
- 稠密化的本质是**混合模型的模型选择问题**：高斯个数是离散的模型复杂度，没有梯度可走。
- 3DGS 用**高斯中心的屏幕空间位置梯度** $\partial L/\partial \mu'$ 作为"边际价值"的代理，它恰好是反传的免费副产品（零额外成本）。
- ⚠️ **它不是"对 $(u,v)$ 的导数"**：这里的 $(u,v)$ 是**高斯中心**的投影坐标，不是像素坐标；而且这个梯度是**对覆盖像素求和后的聚合量**，不是某个点上的局部导数。详见 §3.2.1。
- 这个参数化选择也决定了**计算量是 $O(n)$ 而非 $O(W \cdot H \cdot n)$** —— 所以它才能是反传的免费副产品。详见 §2.1。
- 选中心位置梯度而非不透明度梯度，是因为它带方向、对应"中心自由度"、且其模长指示覆盖不足。
- 这个代理有**三个内生缺陷**，对应三个补丁（opacity reset / 尺度阈值分支 / 窗口与剪枝）。缺陷不是工程 bug，可从权重推导直接看出来。

**一句话表述**：判据度量的是"**把这个高斯在屏幕上挪一挪，它覆盖到的那些像素的残差能降多少**"。

---

## 1. 缺口：连续优化管不了离散的"个数"

补充材料描述的训练循环是：

```
渲染 → 算 loss → 反传 → Adam 更新 μ, Σ, c, α
```

这个循环有一件事**永远做不到**：改变高斯的数量。梯度下降只调已有参数的数值，参数张量形状固定。于是：

| 情况 | 纯梯度下降的结果 |
|---|---|
| 某区域完全没被覆盖 | 永远空洞，现有球怎么优化都盖不到 |
| 一批球是噪声/伪影 | 永远占着位置，怎么优化都在那里 |

而初始的 SfM 点云**同时具备这两个毛病**：太稀疏（大量区域没覆盖）、太脏（离群点、错误匹配）。MVS 路线靠稠密重建解决，3DGS 选了另一条路。

所以必须外挂一个机制，回答两个离散问题：**在哪加、加几个；删谁、删几个。**

---

## 2. 本质：模型选择，用"位移的边际价值"当代理

换个视角：3DGS 是**用混合模型拟合场景**，其中

| 对象 | 性质 | 优化方式 |
|---|---|---|
| μ, Σ, c, α | 连续**参数** | 梯度下降（有解析梯度） |
| K = 高斯个数 | 离散**模型复杂度** | ❌ 无梯度可走，属组合优化 |

经典做法（MDL、贝叶斯非参数、分裂-合并 EM）要重新评估整个模型，代价高。3DGS 的做法便宜得多：

> **用"如果把第 k 个原语挪一挪，loss 能降多少"来衡量它的边际价值。**

而这个量**恰好就是 $\partial L/\partial \mu$，反传时已经算出来了**。判据是**零额外成本的副产品** —— 不需要为模型选择多算任何东西。这是整个设计最漂亮的地方。

### 2.1 为什么"零成本"：参数化让计算量是 $O(n)$ 而不是 $O(W \cdot H \cdot n)$

这一点值得单独说清楚，因为**说成"对 $(u,v)$ 的导数"会给出完全错误的计算量直觉**。

内核的组织方式（`backward.cu`）：

```cuda
const uint32_t pix_id = W * pix.y + pix.x;        // 一个线程 = 一个像素
...
for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {   // 遍历该像素前面的高斯
    ...
    atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);   // 当场归约回高斯
    atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
}
```

| 量 | 规模 |
|---|---|
| 梯度累加器 `dL_dmean2D` | **$n \times 2$ 个 float** |
| 稠密化判据读取的 `xyz_gradient_accum` | $n \times 1$ |
| 形状为 $W \times H \times n$ 的中间张量 | **不存在** |

两种表述的差别正在于**自变量是什么**：

| 表述 | 自变量 | 需要求导的量 | 数量级 |
|---|---|---|---|
| "对 $(u,v)$ 的导数" | 每个**像素**的位置 | 每像素每高斯一组 | $O(W \cdot H \cdot n)$ |
| **"高斯中心的屏幕位置梯度"** | 每个**高斯**的投影中心 | 每高斯一组 | $O(n)$ |

关键在于：**所有像素共享同一个自变量 $\mu'_n$**，于是链式法则里的求和 $\sum_i$ 自动完成了归约 —— 像素视角下的"很多个偏导"天然合成一个 2 维向量。

代价并没有消失，而是**转移到了别处**：残差仍需逐 (像素, 高斯) 对地传播，量级 $O(K)$，其中 $K = \sum_n(\text{第 } n \text{ 个高斯覆盖的像素数})$，受 3σ 裁剪与 tile 划分限制 —— **与渲染前向的代价同阶**。但**梯度本身是 $O(n)$**：既不需要物化 per-pixel 的梯度场，也不需要事后聚合。

**这就是"零额外成本"的技术根源** —— 它不是"渲染完再额外算一个指标"，而是反传路径上本来就存在的一个 $n \times 2$ 张量，取个模长就能用。

---

## 3. 判据的完整推导

### 3.1 补充材料给出的部分

需要交叉引用的三处（编号为补充材料原编号）：

- **式 (3)** 投影雅可比 $J \in \mathbb{R}^{2\times3}$（引自 Zwicker et al. 2002）
- **式 (4)** 协方差投影 $\Sigma' = J R_{cw} \Sigma R_{cw}^\top J^\top$
- **式 (13)** 内积形式的链式法则 $\dfrac{\partial f}{\partial x} = \langle \dfrac{\partial f}{\partial X}, \dfrac{\partial X}{\partial x}\rangle$
- **式 (16)** 颜色梯度 $\dfrac{\partial C_i(k)}{\partial c_n(k)} = \alpha_n \cdot T_n$ —— **这就是影响权重的起点**
- **式 (22)** 由 $\partial L/\partial \mu'$ 反传至相机坐标 $t \in \mathbb{R}^4$

补充材料把 $\partial L / \partial \mu'$ **当作已知输入**（原文：*"Given the gradients of L with respect the projected 2D mean μ′..."*），然后继续往 3D 传。它没有回答：**这个作为输入的 $\partial L/\partial \mu'$ 本身是怎么攒出来的、被谁使用。** 那正是稠密化所在的位置。

### 3.2 逐像素如何攒出 $\partial L/\partial \mu'$

对覆盖像素 $i$ 的高斯 $n$：

$$\frac{\partial L}{\partial \mu_n} = \sum_{i \in \text{tiles}} \underbrace{\alpha_n \, T_{n,i} \, G_{n,i} \, \big(\Sigma_{n}^{'-1}(\mu_n - p_i)\big)}_{\text{影响权重 } w_{n,i}} \cdot \frac{\partial L}{\partial C_i}$$

三个因子各有来源：

| 因子 | 含义 | 代码位置 |
|---|---|---|
| $\alpha_n T_{n,i}$ | 该高斯在像素 $i$ 的**可见权重**（前面所有高斯遮挡）| 式 (16) 同源 |
| $G_{n,i}$ | 该像素处的**高斯核值** | `dG_ddelx` |
| $\Sigma'^{-1}(\mu_n - p_i)$ | **随距离增长的位移灵敏度** | `con_o.x/y/z`（conic）|

### 3.2.1 记号警告：这里的 $(u,v)$ 不是像素坐标

**说成"对 $(u,v)$ 的导数"极不准确**，因为 $(u,v)$ 在 3DGS 语境里通常指**像素坐标**，而这里的自变量是**高斯中心在屏幕上的投影位置** $\mu' = (u_n, v_n)$。两者是不同的东西：

| 符号 | 是什么 | 自由度 | 在公式里的角色 |
|---|---|---|---|
| $p_i = (u_i, v_i)$ | 第 $i$ 个**像素**的位置 | 每个像素各是一个 | 求和的**下标**，权重里作为距离基准 |
| $\mu'_n = (u_n, v_n)$ | 第 $n$ 个**高斯中心**的投影 | 每个高斯只有一组 | **自变量**，梯度就是对它求的 |

所以 $\partial L/\partial \mu'_n$ 是**一个高斯的一组 2 维梯度**，不是"很多像素的导数"。写成 $\partial L/\partial(u,v)$ 会让读者以为在逐像素求导。

**更重要的区别：这是一个求和聚合量，不是逐像素导数。**

- 一个像素**一个点**，谈不上导数
- 这个梯度是**该高斯覆盖的所有像素各自贡献一项、累加而成**（代码里的 `atomicAdd` 就是累加）
- 它的**方向**是多像素残差的合力，**大小**是多像素贡献的加权和

**准确的说法**：

> 一个高斯在屏幕平面上移动，对它**覆盖到的所有像素的残差**产生的影响总和 —— 即这些像素的残差，按各自对该高斯位移的敏感度加权后的合力。

所以它不是"某个位置的局部导数"，而是**跨越整个覆盖足迹的聚合量**。三个推论：

1. **覆盖越多像素，求和的项越多** —— 大球的聚合项天然多于小球，这在"大小"这一维上引入了**尺度偏置**，是 §6.3 缺陷的另一个来源（不只是"大球边界像素多"）。
2. **权重在足迹内极不均匀**（峰值在 $1\sigma$、中心为零，见 §6.1）—— 所以它不是"覆盖区域的平均残差"，而是**按 $r\,e^{-r^2/2\sigma^2}$ 加权的边界加权和**。
3. **换一次求和范围（tile 覆盖判断）就换一个数** —— 这也解释了为什么它是启发式代理，而非某个可微目标函数的精确梯度。

反过来，正因为自变量是**每高斯一组**（而不是每像素一组），这个聚合才能在反传中**免费完成**，输出维度只有 $n \times 2$ —— 计算量与存储量的分析见 §2.1。

### 3.3 权重的精确形式（已数值验证）

内核中 `d = p - μ`，于是

$$\frac{\partial G_{n,i}}{\partial \mu_n} = -G_{n,i}\,\Sigma'^{-1}(\mu_n - p_i) = G_{n,i}\,\Sigma'^{-1}(p_i - \mu_n)$$

对应代码（`backward.cu`，像素级反传）：

```cuda
const float dchannel_dcolor = alpha * T;                    // ① 可见权重
atomicAdd(&dL_dcolors[...], dchannel_dcolor * dL_dchannel);

dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;             // ② 残差汇聚到 α
dL_dalpha *= T;
const float dL_dG = con_o.w * dL_dalpha;                    // con_o.w = alpha

const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;      // ③ 高斯核对均值的偏导
const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
```

注意最后的 `atomicAdd`：**同一高斯在不同像素上的贡献累加**，即上面的 $\sum_i$。

### 3.4 反传到 3D：投影雅可比的转置

内核构造一个 $3\times2$ 矩阵把 $\partial L/\partial\mu' \in \mathbb{R}^2$ 映射到 $\partial L/\partial\mu \in \mathbb{R}^3$：

```cuda
float m_w  = 1.0f / (m_hom.w + 0.0000001f);
float mul1 = (proj[0]*m.x + proj[4]*m.y + proj[8]*m.z + proj[12])  * m_w * m_w;
float mul2 = (proj[1]*m.x + proj[5]*m.y + proj[9]*m.z + proj[13])  * m_w * m_w;
dL_dmean.x = (proj[0]*m_w - proj[3]*mul1)*dL_dmean2D[idx].x + (proj[1]*m_w - proj[3]*mul2)*dL_dmean2D[idx].y;
dL_dmean.y = (proj[4]*m_w - proj[7]*mul1)*dL_dmean2D[idx].x + (proj[5]*m_w - proj[7]*mul2)*dL_dmean2D[idx].y;
dL_dmean.z = (proj[8]*m_w - proj[11]*mul1)*dL_dmean2D[idx].x + (proj[9]*m_w - proj[11]*mul2)*dL_dmean2D[idx].y;
```

**这个矩阵就是 $J^\top$**，即补充材料式 (3) 的转置。验证见附录 A。

### 3.5 判据：只用前两维

```python
# scene/gaussian_model.py
self.xyz_gradient_accum[update_filter] += torch.norm(
    viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
self.denom[update_filter] += 1
...
grads = self.xyz_gradient_accum / self.denom          # 取平均
```

四个必须注意的点：

1. **变量名骗人**：`viewspace_point_tensor` 实际是**屏幕像素坐标**，不是 view space。`gaussian_renderer/__init__.py` 的注释写明 *"gradients of the 2D (screen-space) means"*。
2. **只取 `[:, :2]`**：丢掉第 3 维（我们推出来的那个 `dL_dmean.z`）。代码里算了但判据没用。
3. **可见性过滤**：只统计 `visibility_filter` 内的球。
4. **滑动窗口平均**：`densification_postfix` 每 100 次迭代把**所有**球的累加器清零，所以平均的是"最近 100 次迭代"，不是"球诞生至今"。

---

## 4. 为什么选位置梯度，而不是别的

候选信号很多：不透明度梯度、颜色梯度、协方差梯度、局部残差大小……为什么是位置？

**因为位置梯度是唯一同时满足三个条件的信号：**

**(1) 它天然带方向。** $\partial L/\partial\mu'$ 是 2 维向量，不只说"这个球有问题"，还说"往哪挪能变好"。clone/split 都需要方向信息。标量做不到。

**(2) 它与"形状自由度""存在感自由度"正交。** 对原语而言：

| 参数 | 改变的是什么 | 对应问题 |
|---|---|---|
| μ | **在哪里**（位置分配）| 稠密化要解决的 |
| Σ | **有多大**（形状）| split 要解决的 |
| α | **有多少**（贡献强度）| 外观问题 |

稠密化解决的是**分配问题**，所以该听位置那一支。不透明度梯度反映"该更实还是更虚"，跟"这里缺不缺几何"无关 —— 这也是代码把 `dL_dopacity` 与 `dL_dmean2D` 分开处理的原因。

**(3) 它的模长指示"覆盖不足"。** 权重 ∝ $r\,e^{-r^2/2\sigma^2}$，峰值在 $1\sigma$（见 §6.1）。于是：

> 梯度大 ⟺ **这个球的边界上躺着未被满足的误差** ⟺ 它的覆盖范围和真实几何对不上

盖不住（球太小/太少）和盖过头（一个球糊住多处细节）都会让误差堆在边界上。**一个标量同时抓住两种病。**

---

## 5. 判据不是唯一的准入条件

经常被简化成"梯度大就稠密化"，实际是三个正交条件：

```python
if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
```

| 条件 | 值 | 作用 |
|---|---|---|
| 时间窗口 | 500 < iter < 15000，每 100 次 | **硬门槛** |
| 可见性 | `visibility_filter` | 排除当前视角看不到的球 |
| **梯度模长均值** | ≥ 0.0002 | **准入信号**：哪些球被处理 |
| 尺度比较 | > `percent_dense`·extent | **只决定 split 还是 clone** |

```
梯度模长  ──→ 决定「哪些球被处理」
尺度阈值  ──→ 决定「怎么处理」
```

两者正交：**梯度判据在"大小"这一维上是失明的**（见 §6.3），必须外挂尺度判据。

---

## 5.5 插问：颜色为什么可导

这个问题值得单独回答，因为**颜色的可导性来源和位置/协方差完全不同** —— 后者靠手工推导雅可比（式 3、§3.4），前者靠**参数化本身就是初等函数**，压根不需要特殊处理。

### 5.5.1 颜色模型：一个方向的多项式，系数是唯一可学参数

每个高斯存 $3 \times 16 = 48$ 个 SH 系数（4 阶 × RGB 3 通道）。**这是唯一没有激活函数的参数**：

| 参数 | 激活 | 来源 |
|---|---|---|
| 尺度 $s$ | $\exp$ | `scaling_activation = torch.exp` |
| 旋转 $q$ | 单位化 | `rotation_activation = F.normalize` |
| 不透明度 $\alpha$ | $\sigma$（sigmoid）| `opacity_activation = torch.sigmoid` |
| **颜色（SH 系数）** | **无**（恒等）| 存的就是系数本身 |

因为没有激活，颜色可以取任意实数 —— 负值只能靠前向的 clamp 兜住（见 §5.5.2a）。前向求值（`forward.cu` 的 `computeColorFromSH`）：

```cuda
glm::vec3 dir = pos - campos;  dir = dir / glm::length(dir);   // 方向：从相机指向高斯
glm::vec3 result = SH_C0 * sh[0];
if (deg > 0) {
    result = result - SH_C1*y*sh[1] + SH_C1*z*sh[2] - SH_C1*x*sh[3];
    ...
}
result += 0.5f;
clamped[...] = (result < 0);
return glm::max(result, 0.0f);
```

用符号写就是

$$c_n = \mathrm{clamp}_{[0,\infty)}\Big(0.5 + \sum_{l=0}^{L}\sum_{m=-l}^{l} Y_{lm}(\hat{d}_n)\, k_{nlm}\Big),\qquad \hat{d}_n = \frac{\mu_n - o}{\lVert \mu_n - o\rVert}$$

（$k$ 是学习到的 SH 系数，$o$ 是相机位置，$Y_{lm}$ 是实球谐基。）

**可导性来自三个结构性质：**

| 性质 | 后果 |
|---|---|
| $Y_{lm}(\hat d)$ 是 $\hat d$ 分量的**多项式**（最高 3 次）| 对 $\hat d$ 的偏导是初等式，`backward.cu` 里的 `dRGBdx/dy/dz` 就是这些多项式 |
| $c_n$ 对系数 $k_{nlm}$ **线性** | $\partial c_n/\partial k_{nlm} = Y_{lm}(\hat d)$，无需推导 |
| α 混合对颜色**线性**：$C = \sum_n c_n \alpha_n T_n$ | $\partial C/\partial c_n = \alpha_n T_n$，就是补充材料式 (16) |

**结果**：整张图像 $C$ 是 SH 系数的**低次多项式** —— 没有神经网络、没有体渲染积分、没有采样。求导是纯代数操作，精确且廉价。**这是 3DGS 相比 NeRF 在可导性上的核心优势**：NeRF 的 MLP 靠 autograd 反传，3DGS 的颜色是显式解析式。

### 5.5.2 两处不光滑，各有一处分段处理

**(a) clamp 到非负，梯度按 PyTorch ReLU 约定置零。** 前向把决定记录下来：

```cuda
clamped[3*idx + ch] = (result < 0);
```

反传据此切断该通道：

```cuda
// Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0.
glm::vec3 dL_dRGB = dL_dcolor[idx];
dL_dRGB.x *= clamped[3*idx + 0] ? 0 : 1;
```

**为什么必须这么做**：一旦某通道被夹到 0，`∂c/∂k` 在数学上为零（输出不再随系数变化）。若仍把残差回传，优化器会把系数往错误方向推，越推越被夹住 —— 这是饱和死区。置零等于承认"这个方向暂时没有信息"。

**(b) 方向单位化，梯度按投影算子回传。**

$$\frac{\partial \hat d}{\partial d} = \frac{\mathbf{I} - \hat d\hat d^\top}{\lVert d\rVert}$$

单位化的雅可比是**去掉径向分量**的投影（沿方向缩放不改变 $\hat d$），实现为 `auxiliary.h` 的 `dnormvdv`：

```cuda
float sum2 = dot(v,v);
float invsum32 = 1.0f / sqrt(sum2*sum2*sum2);
out.x = ((+sum2 - v.x*v.x)*dv.x - v.y*v.x*dv.y - v.z*v.x*dv.z) * invsum32;
```

### 5.5.3 ⚠️ 一个容易猜错的点：方向的梯度**确实**回传到了位置

我一开始以为方向是"当作常数、梯度截断"—— 因为视觉上它由相机位置定义，像外部输入。**代码明确否定了这个猜测**（`backward.cu` 结尾的注释）：

```cuda
// The view direction is an input to the computation. View direction
// is influenced by the Gaussian's mean, so SHs gradients
// must propagate back into 3D position.
glm::vec3 dL_ddir(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));
float3 dL_dmean = dnormvdv(dir_orig, dL_ddir);
dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
```

所以 3D 位置 μ 的**总**梯度有**两条通路**叠加：

| 通路 | 来源 | 是否进稠密化判据 |
|---|---|---|
| 屏幕位置：$\partial L/\partial\mu' \cdot J^\top$ | 几何（覆盖哪里）| ✅ **是** |
| 视角相关颜色：$\mathrm{dnormvdv}\cdot\partial L/\partial\hat d$ | 外观（从哪个角度看）| ❌ 否（走了另一个聚合器 `dL_dmeans`）|

这印证了 §4 的论点：**稠密化只听"几何分配"那一支**，而"移动改变观察方向"这一支虽然后向传播时存在，但不参与增删决策 —— 否则反射高光会让球不断被克隆。

### 5.5.4 SH 的两类优化问题（论文自己也承认）

**(a) 高阶系数需要足够的角向信息。** 论文 Sec. 7.1：

> SH coefficient optimization is sensitive to the lack of angular information.

高阶 $Y_{lm}$ 的系数只在多个视角下才能被约束；视角少时它们会去拟合噪声。**对策是渐进升阶**（`train.py`）：

```python
# Every 1000 its we increase the levels of SH up to a maximum degree
if iteration % 1000 == 0:
    gaussians.oneupSHdegree()
```

**(b) SH 的取值区间受限，亮色要靠高阶项硬撑。** 0 阶项 $Y_{00} = \tfrac{1}{2}\sqrt{1/\pi} \approx 0.2821$ 是个**固定常数**，而 $c = 0.5 + 0.2821\,k_0 + (\text{高阶})$。要让某通道饱和到 1，需要 $\sum Y_{lm}k_{lm} \ge 0.5$：

| 只用 $k_0$ | 需要的 $k_0$ |
|---|---|
| 输出 0.5（中性灰）| 0 |
| 输出 1.0（饱和）| $\approx 1.77$ |

高阶项把可用区间略微拓宽，但仍然**无法表示任意亮度**。这正是 clamp 频繁触发、进而导致 (a) 里梯度被置零的根源之一 —— 两个问题是耦合的。

### 5.5.5 小结：为什么"颜色可导"这件事不平凡

| 对比 | NeRF | 3DGS |
|---|---|---|
| 颜色函数 | MLP（多层非线性）| SH 多项式（显式解析）|
| 求导方式 | autograd 反传整张网络 | 手工初等偏导（`dRGBdx/dy/dz`）|
| 参数规模 | 网络权重 | 每高斯 48 个系数 |
| 不光滑点 | 采样、位置编码 | clamp + 方向单位化（均有分段处理）|

**一句话**：颜色可导，是因为它被**故意参数化成"方向的多项式 + 线性混合"** —— 把可导性设计进了表示里，而不是靠自动微分去处理一个复杂函数。代价是表达能力受限（§5.5.4b），收益是求导精确、廉价、且梯度有明确物理含义。

---

## 6. 三个内生缺陷（可从权重推导直接看出）

### 6.1 权重峰值在 $1\sigma$，中心贡献为零

沿主轴，$|w| \propto r\,e^{-r^2/2\sigma^2}$：

```
r = 0.00σ : |w| ∝ 0.0000     ← 高斯贡献最大的中心，梯度为零
r = 0.75σ : |w| ∝ 0.5188
r = 1.00σ : |w| ∝ 0.5558     ← 峰值
r = 1.50σ : |w| ∝ 0.4462
r = 2.50σ : |w| ∝ 0.1007     ← 尾部迅速衰减
```

（精确峰值在 $r = \sigma$，相对值 $e^{-1/2} \approx 0.6065$。）

**梯度完全由边缘像素驱动。** 这既赋予了判据发现欠重建的能力，也说明它测的是"**边界是否被满足**"，而不是"这个球整体拟合得好不好"。

### 6.2 对深度完全失明

权重 $w$ 只有 2 个自由度（屏幕平面）。整个反传链从 $\partial L/\partial\mu'$ 到 $\partial L/\partial\mu$ 只经这 2 维传递，**沿视线的深度方向没有对应项**。

不是代码里 `[:, :2]` 才丢的 —— 而是**图像损失本身对深度无感**：球沿射线滑动时 $G$、$T$、$\alpha$ 全不变，权重不变。所以这个代理是**秩亏的**。

后果：球可在射线上任意漂移而无惩罚，即**浮点伪影（floaters）**的数学根源。论文 Sec. 5.2 承认：

> our optimization can get stuck with floaters close to the input cameras

### 6.3 分不清"缺覆盖"与"高对比度"，也分不清大小

- 权重含图像局部结构（$\partial L/\partial C$ 由图像梯度决定）→ 高频纹理、镜面高光同样制造大梯度
- **聚合项数随覆盖像素数增长**（§3.2.1 推论 1）→ 大球天然比小球求和更多项，梯度更大

所以判据**无法区分**"这里真的缺几何"、"这里本来就难拟合"、"这个球该拆开"。后两条各自指向**尺度偏置**：判据对"球的大小"没有归一化，这既让大球更容易被选中，也让"该拆开"这件事无法从梯度本身读出来 —— 只能靠外挂的尺度阈值补。

---

## 7. 补丁与缺陷的对应

| 缺陷 | 补丁 | 代码机制 |
|---|---|---|
| 深度失明（秩亏）| **periodic opacity reset** | 每 3000 次迭代把所有 α 压到 ≤ 0.01（`reset_opacity`），强迫每个球重新证明自己；无支撑的球随后被 α < 0.005 剪掉 |
| 分不清大小 | **尺度阈值分支** | `> percent_dense * scene_extent`（默认 0.01，即场景半径的 1%）决定 split，否则 clone |
| 代理可被噪声刷高 | **窗口 + 可见性 + 屏幕尺寸剪枝** | 只在 500–15000 迭代稠密化；只统计可见球；首次 reset 后剪掉屏幕半径 > 20px 或世界最长轴 > 0.1·extent 的球 |

**两个值得注意的设计信号：**

1. **稠密化在训练过半就关闭**（15000 / 30000）。说明它被当作"打地基"阶段，而非全程机制；后期交给梯度下降精修。
2. **论文结尾自认这块没做完**：
   > we currently do not apply any regularization to our optimization; doing so would help with both the unseen region and popping artifacts

   即：浮点伪影本应由正则项处理，而当前只能用 opacity reset 这种启发式兜住。

---

## 8. 同构：这不是新思想

| 领域 | "哪里分辨率不够就加自由度" |
|---|---|
| 有限元 | **自适应网格细分（AFEM）**：按后验误差估计局部加密单元 |
| 数值 PDE | **自适应网格细化（AMR）**：按解的梯度细化网格 |
| SPH / 粒子法 | 粒子分裂与合并：按局部误差/密度增减粒子 |
| 混合模型 | **分裂-合并 EM**：按分量对似然的贡献增删分量、重整协方差 |
| **3DGS** | **按高斯中心的屏幕位置梯度增删高斯**（聚合量，见 §3.2.1）|

$r\,e^{-r^2/2\sigma^2}$ 的形式与 AFEM 里"误差指示子在单元边界最大"完全同构 —— 都在说"**病在边界上**"。

**一句话本质：**

> 3DGS 的稠密化是一套用于混合模型的自适应自由度分配策略，用反传免费得到的位置梯度作为"边际价值"的代理，求解一个本来没有梯度的离散模型选择问题。

---

## 9. 旁证：近似精度其实不重要

本目录 `does-3dgs-need-accurate-volumetric-rendering.pdf`（Eurographics 2025）的结论是：**把 3DGS 那些粗糙的体渲染近似换成更精确的，质量反而未必更好** —— 因为"高效优化 + 大量高斯"的威力盖过了近似误差。

这支持上述读法：**3DGS 的成功不主要来自渲染方程的精确，而来自把自由度投放到该投放的地方。** 高斯数量和位置分配对了，粗糙的 α 混合也能拟合得好；反之再精确的渲染，自由度放错地方也没用。

---

## 10. 论文与代码的差异

| 项 | 论文 | 官方代码 |
|---|---|---|
| split 的尺度因子 | "divide their scale by a factor of ϕ = 1.6" | `scaling / (0.8 * N)`，N=2 → 1/1.6 ✓ 一致 |
| clone 的位置 | "moving it in the direction of the positional gradient" | `new_xyz = self._xyz[mask]` —— **纯复制，无梯度偏移** ❌ 未兑现 |
| 判据的空间 | "view-space positional gradients" | 屏幕像素空间（变量名 `viewspace_*` 是错误命名）|
| 阈值 | τ_pos = 0.0002 ✓、ϕ = 1.6 ✓、N = 3000 ✓、每 100 迭代 ✓ | 一致 |
| 稠密化起止 | "after optimization warm-up"（未给数）| `densify_from_iter=500`、`densify_until_iter=15000`、总 30000 |
| 尺寸剪枝阈值 | 未给具体数 | 屏幕半径 20px；世界 0.1·extent |
| α 剪枝阈值 ε_α | 未给具体数 | 0.005 |

---

## 附录 A：$J^\top$ 的数值验证

可直接运行：`python3 papers/02-公式推导/verify_backward_jacobian.py`（同目录，附在本笔记旁边）。

复刻 `backward.cu` 的索引方式（`proj` 为 `W @ P` 展平的 16 元素），对屏幕坐标做中心差分：

```python
import numpy as np
rng = np.random.default_rng(1)
P = rng.normal(size=16)
m = rng.normal(size=3)

def proj4(P, m):
    return np.array([P[0]*m[0]+P[4]*m[1]+P[8]*m[2]+P[12],
                     P[1]*m[0]+P[5]*m[1]+P[9]*m[2]+P[13],
                     P[2]*m[0]+P[6]*m[1]+P[10]*m[2]+P[14],
                     P[3]*m[0]+P[7]*m[1]+P[11]*m[2]+P[15]])

def screen(P, m):
    h = proj4(P, m)
    return h[:2] / h[3]

# 数值雅可比 J = ∂(u,v)/∂(x,y,z)
J = np.zeros((2, 3))
for i in range(3):
    e = np.zeros(3); e[i] = 1e-7
    J[:, i] = (screen(P, m+e) - screen(P, m-e)) / 2e-7

# 内核构造的矩阵
m_hom = proj4(P, m)
m_w = 1.0 / (m_hom[3] + 1e-7)
mul1 = (P[0]*m[0] + P[4]*m[1] + P[8]*m[2] + P[12]) * m_w * m_w
mul2 = (P[1]*m[0] + P[5]*m[1] + P[9]*m[2] + P[13]) * m_w * m_w
A = np.array([[P[0]*m_w - P[3]*mul1, P[1]*m_w - P[3]*mul2],
              [P[4]*m_w - P[7]*mul1, P[5]*m_w - P[7]*mul2],
              [P[8]*m_w - P[11]*mul1, P[9]*m_w - P[11]*mul2]])

print(np.allclose(A, J.T, atol=1e-4), np.abs(A - J.T).max())
```

实测输出 `True`，最大误差 `0.0026` —— **内核构造的矩阵确为 $J^\top$**，与补充材料式 (3) 一致。

## 附录 B：参数表（官方 `arguments/__init__.py`）

```python
self.percent_dense            = 0.01      # 大/小球的划分：场景半径的 1%
self.densification_interval   = 100
self.opacity_reset_interval   = 3000
self.densify_from_iter        = 500
self.densify_until_iter       = 15_000
self.densify_grad_threshold   = 0.0002    # τ_pos
self.lambda_dssim             = 0.2
self.iterations               = 30_000
```

剪枝阈值在 `train.py` 中硬编码：不透明度 `0.005`、屏幕半径 `20`、世界尺度 `0.1 * extent`。

---

## 参考

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023. Sec. 5.1 / 5.2 / 7.1，附录 B（Algorithm 1）
- Ye & Kanazawa, *Mathematical Supplement for the gsplat Library*, arXiv:2312.02121. §2.1 式 (3)(4)、§3.1 式 (13)(16)、§3.2 式 (22)
- 代码：`graphdeco-inria/gaussian-splatting`（`train.py`、`scene/gaussian_model.py`、`gaussian_renderer/__init__.py`）、`graphdeco-inria/diff-gaussian-rasterization`（`cuda_rasterizer/backward.cu`）
