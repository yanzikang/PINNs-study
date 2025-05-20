# PINNs学习文档

==初步文档后续需要整理==

![image-20250427164407913](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250427164407913.png)

# 第一周（4.27~5.9）

任务列表

- [x] 1.神经网络框架
- [x] 2.用deepxde构建简单的PDE方程并形成案例
- [ ] PINNs数学原理

# 一、PINNs入门课

## 1.PINNs入门介绍

从整体上体会，PINNs是什么？基本的学习内容是什么？

- [x] 介绍视频一：[DeepOnet介绍_哔哩哔哩_ bilibili](https://www.bilibili.com/video/BV1dwzhYEEdf/?spm_id_from=333.337.search-card.all.click&vd_source=b056df9f055466883abc77833142c71a)

- [x] 介绍视频二：[飞桨+DeepXDE全面支持科学计算多领域方程求解_哔哩哔哩_ bilibili](https://www.bilibili.com/video/BV18e411c7m7/?spm_id_from=333.337.search-card.all.click&vd_source=b056df9f055466883abc77833142c71a)

通过上述两个视频，可以较好的解决PINNs是什么的问题。

总的来说：PINNs是解偏微分方程(PDE)，什么微分方程？含有物理信息的微分方程；用什么方法解？用神经网络的方法求解。

主要的流程如下：

- 问题建模：明确是什么样的物理模型，要考虑到几何形状，模型尺度大小，模型的维度等问题
- 损失设计：一般而言，分为PDE损失，初始损失，边界损失
- 模型设计：一般采用MLP作为模型的网络架构
- 优化设计：由于PINNs的一个难点就是收敛困难，为了保证模型正常的收敛，需要采用一些策略保证模型正常收敛，常见的策略有，自适应采样，自适应权重等。同时也有进行域分解等方法。

## 2.神经网络框架

### 1.基本知识介绍

PINNs通过在损失函数中引入物理方程（如偏微分方程），使神经网络在训练过程中不仅依赖数据，还遵循已知的物理规律。

![image-20250506151412828](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250506151412828.png)

最为常见的神经网络框架为全连接神经网络，起到近似函数的作用。

<img src="https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250506153453601.png" alt="image-20250506153453601" style="zoom:80%;" />

图示为二维瞬态热传导方程，我们只知道其偏微分方程(PDE)，但是不知道$T$具体的解析解，全连接神经网络的作用就是用于拟合这个解析解$T$，由于pytorch等深度学习框架带有自动微分的功能，自动微分由计算图作为保证，这种性质自然的对PDE的计算是有利的，基于此得出一下的结论**（Q&A）**：

- 输入是什么？

  输入是自变量$\mathcal{X}$,可能包含如坐标$x,y,z,$以及时间$t$

- 输出是什么？

  输出为所需要的预测结果$u$

- 训练什么？

  训练网络参数$\theta$

- 靠什么进行监督？ 

  损失函数，即各种具有物理意义的方程(如：边界条件，初始条件，以及残差方程)

- 为什么PINN中要设置边界条件，初始条件，以及残差方程呢？每一个的作用是什么？

  残差方程（物理方程损失）：通过偏微分方程（PDE）的残差项，强制神经网络输出的解满足控制物理过程的数学方程

  边界条件（BCs）：约束解在计算域边界的行为，确保物理场景的合理性

  初始条件（ICs）：定义系统在初始时刻的状态，为瞬态问题提供时间演化的起点。

### 2.pytorch+deepxde框架学习

由于后面的基础部分学习是基于清华大学的**PINNacle**开展的，所以有必要学习deepxde框架，常见的科学机器学习框架还有$JAX$等

对于$JAX$库的学习以及在训练过程中肯能会遇到的问题可以参考这篇文章：[物理信息神经网络训练的专家指南（Part I）| An Expert’s Guide To Training PINNs](https://mp.weixin.qq.com/s/gCPJQGYiWaw2OsXzZ8RTPA)

deepxde参考文档：[DeepXDE — DeepXDE 1.13.3.dev9+g28cb8f0 documentation](https://deepxde.readthedocs.io/en/latest/)

##### 2.1 deepxde基础代码说明(常见代码说明)

deepxde 常见代码：

**2.1.1 几何构造**

- [ ] 基本几何构造

```python
# 导入dde
import deepxde as dde
# 1维区域
dde.geometry.Interval(x_min, x_max)  # 例如定义[-1,1]区间：geom = dde.geometry.Interval(-1, 1)

# 2维区域
#矩形
dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])  # 如矩形域[-1,1]×[-1,1]
# 圆形
dde.geometry.Disk(center, radius)  # 例如中心在原点、半径为1的圆：dde.geometry.Disk([0,0], 1)
# 椭圆、多边形、星形、三角形等(不太常用详情参考deepxde文档)

# 三维区域
# 长方形:xmin 左下角,xmax 右上角
dde.geometry.Cuboid(xmin, xmax) # dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[1, 1, 1])
# 球
dde.geometry.Sphere(center, radius) # dde.geometry.Cuboid(center=[0, 0, 0], radius= 5)

```

- [ ] 几何运算

```python
# 几何运算
# 并集:合并两个几何区域，常用于组合不规则形状：
rect = dde.geometry.Rectangle([0,0], [2,1])
disk = dde.geometry.Disk([2,0.5], 0.5)
geom = dde.geometry.CSGUnion(rect, disk)  # 创建跑道形区域[1,4](@ref)
# 差集：从一个几何体中挖去另一个区域，适用于孔洞结构：
cube = dde.geometry.Cuboid([0,0,0], [1,1,1])
sphere = dde.geometry.Sphere([0.5,0.5,0.5], 0.3)
geom = dde.geometry.CSGDifference(cube, sphere)  # 立方体中挖去球体[5,6](@ref)
# 保留重叠区域，用于构建复杂边界条件
circle1 = dde.geometry.Disk([-1,0], 1.5)
circle2 = dde.geometry.Disk([1,0], 1.5)
lens = dde.geometry.CSGIntersection(circle1, circle2)  # 双凸透镜形状[1](@ref)
```

- [ ] 几何构造进阶(自定义几何)

```python 
# 先读球构造的源码
```

- [ ] 时间域(瞬态方程)以及时空域

```python
# 建立时空域
timedomain = dde.geometry.TimeDomain(t0,t1)
spatial_time_domain = dde.geometry.GeometryXTime(geom,timedomain)
```

2.1.2 边界条件以及初始条件设定

```python
# Dirichlet boundary conditions: y(x) = func(x).
# 参数说明：spatial_time_domain 时空域；lambda x: 100 设定恒定温度为100，x为输入的值x[0]表示x轴
# lambda x, on_boundary: on_boundary and  np.isclose(x[1], H/2) on_boundary是否在边界上，
# np.isclose(x[1], H/2) 是否满足y=H/2 H为自定义矩形的高
dde.icbc.DirichletBC(spatial_time_domain, 
    lambda x: np.float64(100), 
    lambda x, on_boundary: on_boundary and  np.isclose(x[1], H/2) 
    )
# Neumann boundary conditions: dy/dn(x) = func(x).
dde.icbc.NeumannBC(
        spatial_time_domain, # 区域
        lambda x: 0, # func(x)
        lambda x, on_boundary: on_boundary and np.isclose(x[0], -L/2) # 边界区域
# Robin boundary conditions: dy/dn(x) = func(x, y).
dde.icbc.RobinBC(
        spatial_time_domain, 
    	lambda _, T:(30 - T),  # func(x, y).
        lambda x, on_boundary: on_boundary and disk.on_boundary(x[:2])

# 初始条件 Initial conditions: y([x, t0]) = func([x, t0]).
ic = dde.icbc.IC(
    spatial_time_domain,
    lambda x:0,
    lambda _,on_initial:on_initial)
```

2.1.3 构建PDE方程以及数据集data

```python
# 
def pde(x,y):
    dy_t = dde.grad.jacobian(y,x,i=0,j=2)
    dy_xx = dde.grad.hessian(y,x,i=0,j=0)
    dy_yy = dde.grad.hessian(y,x,i=1,j=1)
    return dy_t - alpha*(dy_xx + dy_yy)
# 汇总边界条件
ibcs = [bc_top,bc_bottom,bc_left,bc_right,ic]
# 定义数据
data = dde.data.TimePDE(
    spatial_time_domain,
    pde,
    ibcs,
    num_domain=8000,
    num_boundary=320,
    num_initial=800,
    num_test=8000,
)
```

2.1.4 训练技巧

```python
# 自适应采样
resampler = dde.callbacks.PDEPointResampler(period=1000)  # 每1000步触发一次RAR
model.train(iterations=10000, callbacks=[resampler])
# 优化器组合策略
model.compile("adam", lr=0.001)  
model.train(epochs=5000)
model.compile("L-BFGS")  # 切换优化器
model.train()
# 损失函数动态加权
weights = [1e3, 1e3, 1]  # 分别为PDE、边界、初始条件权重
model.compile(optimizer, loss_weights=weights)
```

2.1.5 设置网络

```python
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")  # 输入维度2，输出维度1
```

由于使用deepxde当遇到比较复杂的问题时，可能需要重写封装代码，故我的建议是用pytorch 自己构建PINNs的代码

### 3.用pytorch+deepxde实现2D热传导问题

案例1：瞬态热传导问题（deepxde）

详情阅读参考代码网页链接，以及github链接中的README文件

参考代码：[PINN求解瞬态热传导问题](http://www.360doc.com/content/23/1220/08/83587350_1108171032.shtml)

Github链接：[2D_Heat](https://github.com/yanzikang/2D_Heat)

案例2：稳态热传导稳态（pytorch）

Github链接：[2D 多域稳态热传导](https://github.com/yanzikang/2D-heat-transfer)

### 3.PINNs数学原理（有待进一步完成）

- PDE理论基础
- 自动微分原理

# 第二周（5.12-5.19）学不完了

- [ ] 神经正切核理论分析以及代码实现
- [ ] 收敛性分析
- [ ] 采样策略优化
- [ ] 自适应权重方法

沿着是什么？为什么？如何实现的路线进行学习。

# 二、PINNs高级理论（==重点==）

### 1.收敛性分析

目标：如果模型不收敛应该从哪些方面进行分析找原因

### 2.采样策略优化                                                                                                                                                                                                                                                                                                                                                                       

目标：采样对模型有什么影响？如何正确的进行采样

### 3.自适应权重方法

作用：动态加权，实际上通过衡量loss的梯度的大小，进行加权，防止某一项下降的过快，或者过慢

原理理解：在训练过程中会出现这样的情况，训练的时候PDE损失下降的非常的快，从比较大的数量级下降到很小的数量级，但是对于我们的边界调价下降的非常的慢，对于这种情况非常容易将我们的解，解到错误的解，或者说是解到次优解上去，这是我们不愿意看到的。所以要对各项损失进行加权，那么如何进行加权？

**我这里有一个建议，损失之间的冲突问题再多任务学习中是一个较为常见的问题，可否从CV的多任务学习中获取相干灵感？**

#### 1.神经切向核理论（NTK）【基于梯度的加权方式】

在论文中提到的**NTK分析（Neural Tangent Kernel Analysis）**是一种基于神经网络训练动力学的理论分析工具，主要用于研究神经网络在参数初始化附近的训练行为，以及不同损失项的优化动态。以下是其核心概念和在论文中的具体应用解释：

------

**1. NTK（神经切向核）的基本概念**

- **定义**：NTK是描述无限宽神经网络在梯度下降训练过程中参数演化规律的数学工具。它通过核函数（类似于支持向量机中的核）的形式，刻画了神经网络输出对输入的敏感性。

- **关键性质：**

​	在神经网络足够宽且使用随机初始化时，NTK在训练初期**保持近似不变**（尽管参数在更新）。

​	NTK的特征值谱（Eigenvalue Spectrum）决定了不同方向的参数更新速度（大特征值对应快速收敛方向，小特征值对	应缓慢收敛方向）。

------

**2. 论文中的NTK分析目的**

论文通过对PINNs（物理信息神经网络）进行NTK分析，主要解决以下问题：

1. **训练动态不平衡**：
   PINNs的损失函数通常包含多个竞争项（如PDE残差、边界条件、初始条件等），这些项的梯度量级差异会导致训练偏向某些项而忽略其他项。NTK分析可**量化各损失项的梯度贡献比例**。
2. **变量缩放的理论支持**：
   论文提出的变量缩放技术通过调整不同物理量的权重，改变了NTK的特征值分布。NTK分析证明了这种调整能：
   - 平衡不同损失项的梯度量级（如防止PDE残差的梯度淹没边界条件）
   - 改善优化过程的整体收敛性（通过调整特征值谱的分布）

------

**3. NTK分析在论文中的具体应用**

**（1）揭示训练瓶颈**

- **现象**：未缩放时，PINNs的训练可能因不同损失项的梯度量级差异而陷入局部最优。

- NTK分析过程：

  - 计算各损失项对应的NTK子矩阵（如PDE残差项的核$K_{PDE}$、边界条件项$K_{BC}$

  - 比较它们的最大特征值$\lambda_{\max}$和特征值分布： 
    $$
    梯度主导项=arg⁡max⁡(λmax⁡(K_{PDE}),λmax⁡(K_{BC})
    $$

  - 发现：未缩放的PINNs中，PDE残差项的NTK特征值显著大于边界条件项，导致训练初期优先优化PDE残差，忽视边界条件。

**（2）验证变量缩放的有效性**

- **变量缩放操作**：对不同的物理变量（如位移$u$、温度$T$）施加缩放因子$\alpha_u, \alpha_T$，使其量级对齐。

- NTK分析结论：

  - 缩放后，各损失项的NTK矩阵被重新加权： 
    $$
    K_{\text{total}} = \alpha_{PDE}^2 K_{PDE} + \alpha_{BC}^2
    $$

  - 调整$\alpha$因子可**平衡各子矩阵的贡献**，使得最大特征值接近，避免单一损失项主导训练。

  - 理论证明：存在一组最优的缩放因子$\{\alpha\}$，使得调整后的NTK特征值谱更均匀，从而提升收敛速度。

------

**4. NTK分析的实际意义**

通过该分析，论文从理论上证明了：

1. **变量缩放必要性**：
   对于具有快速变化解（如高雷诺数流动）的PDE问题，未缩放的PINNs因NTK特征值差异过大会完全失败，而缩放后能恢复有效训练。
2. **超参数设计指导**：
   NTK特征值的计算为缩放因子的选择提供了理论依据（例如，根据特征值比例设置$α∝1$

------

**直观类比理解**

将NTK分析类比为**调整不同乐器的音量平衡**：

- **未缩放时**：某些乐器（如鼓）音量过大，掩盖其他乐器（如小提琴），导致音乐不和谐（训练偏向某些损失项）。
- **缩放后**：通过调节音量旋钮（缩放因子$\alpha$），使所有乐器的音量均衡（梯度量级对齐），最终得到和谐乐曲（优化过程平衡收敛）。

------

**总结**

在论文中，**NTK分析**不仅为变量缩放技术提供了理论保障，还揭示了PINNs训练的深层机制（如梯度竞争问题），并通过调整NTK特征值分布证明了方法的有效性。这一分析框架为改进其他物理驱动机器学习模型提供了重要工具。



代码实现：参考代码

```python

```

#### 2.实验过程中对损失进行加权会出现的问题

看似损失下降了，但是实际上损失并没有真正的下降，而是用将大损失的权重调小了。

### 4.归一化策略（无量纲化策略）

目标：将输入或者输出规范到神经网络”**喜欢**“的范围内

**1.输入**

在做热传导实验的时候发现，对于输入时有讲究的，**不是只要满足物理规律的就可以作为网络的输入**，还需要考虑输入的范围，就像做CV的时候会考虑**归一化**，在做PINNs的时候也要考虑**无量纲化**，但是并不是所有的数据都可以较好的进行无量纲化，所以要考虑输入的范围如何设置在一个合理的值。这是非常值得考虑的一个点。**过小的输入**值会导致最终的预测温度趋于一个相似的值，即神经网络无法将他们分离；**过大的输入**虽然还没有做实验，但是主观的感觉是，他们会有较大的差距。

对于输入的看法已经可以从实验中进行验证了，不合理的归一化操作会使得模型出现大问题。

[如何理解归一化（normalization）? - 知乎](https://zhuanlan.zhihu.com/p/424518359)

**2.输出**

同样对于输出的预测值对实验也是存在影响的，在CV领域中监督学习都是有真值作为指导的，比如说IoU(交并集)，BCE（二元交叉熵损失函数）。有的是分类任务，有的是回归任务。很显然我们的任务是**回归任务**；要对所有网格中的温度值进行预测。在做热传导实验的时候会出现这样的一个现象，对于不同的初始条件，会导**致输出的量级发生变化**，比如对于恒温边界条件，100摄氏度的温度；在最开始的时候假设预测为0摄氏度，那么损失为100的平方，假设初始温度为600摄氏度，预测温度为0摄氏度，那么损失就为600的平方。对于控制方程不含温度定值，损失较小；这种情况下，会导致各个损失之间不平衡，即网络有限关注损失大的损失项目。同样的应该对输出也进行**归一化处理（无量纲化处理**）。

在回归任务中需要进行归一化操作，但与此同时也有部分需要进行注意的点：由于输出是没有min-max的，并且在初始的时候由于存在着**迪利克雷边界条件**，min和max难以用输出值进行界定，所以一般不对输出进行归一化，但是实际中，如果对于不同的情况，量级相差过大：

**需要归一化的情况**：

- **输出范围过大**：若目标变量（如房价、温度）的数值范围差异大（例如从0到1e6），直接使用原始值训练可能导致梯度爆炸或损失函数计算不稳定（如MSE）。
- **激活函数限制**：若模型输出层使用Sigmoid或Tanh等激活函数（限制输出范围在[-1,1]或[0,1]），需将目标变量归一化到对应区间，否则模型无法拟合真实值。
- **多任务回归**：当多个回归任务的输出尺度差异较大时，归一化可平衡各任务的损失贡献。

常用的归一化策略：

min-max归一化：

![image-20250519160858151](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250519160858151.png)

权重归一化：

![image-20250519161052188](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250519161052188.png)

# **三、PINNs前沿发展**

这个先自己调研一下

### 文章1

**Challenges in Training PINNs: A Loss Landscape Perspective**

文章连接：[Challenges in Training PINNs: A Loss Landscape Perspective](https://arxiv.org/pdf/2402.01868)

代码连接：[pratikrathore8/opt_for_pinns](https://github.com/pratikrathore8/opt_for_pinns)

从本文中可以学习到一点，Adam+L-BFGS 在PINNs训练过程中的有效性，可以帮助我们进一步的提高训练的精度！

### 文章2

**Parameterized Physics-informed Neural Networks for Parameterized PDEs**

文章连接：[Parameterized Physics-informed Neural Networks for Parameterized PDEs](https://arxiv.org/pdf/2408.09446)

代码连接：[WooJin-Cho/Parameterized-Physics-informed-Neural-Networks](https://github.com/WooJin-Cho/Parameterized-Physics-informed-Neural-Networks)

后续做多个介质的PINNs任务的时候可以参考，参数化的PINNs这个东西还是比较重要的

# 现阶段学习内容

日期：2025/5/12

想法：不能让知识局限于PINNs这个领域，而是要进行泛化的学习，什么意思？比如对于元学习的策略，不能说在PINNs中没有就不做了，而是要在其他的领域中，比如cv中找到对应的元学习策略，套用在PINNs的方法中。

- [ ] 认识什么是元学习，cv中是如何实现的，注意找时间比较靠前的文章！这种文章实现起来比较简单。是否实现？

日期：2025/5/20

总结归纳一下，现在手头上的热传导任务后续需要进行的尝试及工作应该如何进行

### 热传导尝试工作：

#### 1.优化策略

- [ ] 归一化策略，包括输入和输出
- [ ] Adam+L-BFGS 优化器使用策略
- [ ] 自适应采样策略
- [ ] 动态学习率策略
- [ ] 动态加权策略（神经正切核、config等都属于对于梯度的操作）
- [ ] **硬约束策略**（这一点后续可能比较重要）

#### 2.模型框架策略

- [ ] MLP
- [ ] Transformer
- [ ] KAN
- [ ] LSTM
- [ ] Mamba(这个是基于transformer的设想，这个做出来就是一个创新点)

#### 3.后续尝试的工作

- [ ] 基于各种优化策略的调优，可能需要各种策略之间做组合，以及超参数的调整等操作
- [ ] 考虑不同多层介质的问题，即参数化PDE问题，以及PINNs的泛化能力考虑

依照目前我的理解不同的介质，实际上是分为两类，第一种问题就是不同的域，不同的材料所属的空间位置，第二个问题实际上是参数的问题，依旧是说实际上不同的介质对应的是不同的参数，类比于NS 方程中的雷洛系数不同导致流体的运动状态完全不同。所以这个参数化的PDE问题是比较值得深思的问题

- [ ] 多元边界条件问题

虽然名字叫做多元边界条件问题，实际上就是边界条件复杂问题，参考苏航老师团



# 其他

这个大佬准备些一个库，用于将所有的现有方法进行总结

[thuml/Neural-Solver-Library: A Library for Advanced Neural PDE Solvers.](https://github.com/thuml/Neural-Solver-Library)

### 阅读推荐（一些我读过的感觉比较有用的文章）

PINNs综述：https://mp.weixin.qq.com/s/dpQlQUDAv3VoTjddLWAZLg
