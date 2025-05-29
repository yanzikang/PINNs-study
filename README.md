# PINNs学习文档

==初步文档后续需要整理==

![image-20250427164407913](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250427164407913.png)

# 第一周（4.27~5.9）

任务列表

- [x] 1.神经网络框架
- [x] 2.用deepxde构建简单的PDE方程并形成案例
- [x] PINNs数学原理

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

对于$JAX$库的学习以及在训练过程中肯能会遇到的问题可以参考这篇文章：

[物理信息神经网络训练的专家指南（Part I）| An Expert’s Guide To Training PINNs](https://mp.weixin.qq.com/s/gCPJQGYiWaw2OsXzZ8RTPA)

deepxde参考文档：

[DeepXDE — DeepXDE 1.13.3.dev9+g28cb8f0 documentation](https://deepxde.readthedocs.io/en/latest/)

##### 2.1 deepxde基础代码说明(常见代码说明)

deepxde 常见代码：[yanzikang/deepxde-study: 用于学习deepxde](https://github.com/yanzikang/deepxde-study)

由于使用deepxde当遇到比较复杂的问题时，可能需要重写封装代码，故我的建议是用pytorch 自己构建PINNs的代码，但缺点也非常的明显，许多的代码要自己去构建，同时会一定程度上增加代码的冗余性。

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
- [AI与PDE（一）：PINNs模型的设计理念和我碰到的一些问题 - 知乎](https://zhuanlan.zhihu.com/p/411843646)

### 4.PINNs学习过程中的难点

- 收敛难，由于在训练的过程中可能会有多个损失项，这些损失项之间不平衡，导致损失降不下去

- 收敛到了平凡解上去，举一个简单的例子

  <img src="https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250522132053435.png" alt="image-20250522132053435" style="zoom:80%;" />

对于上述方程来说，$u_{\theta}=0$ 明显是上述偏微分方程的解，但是很明显这是一个平凡解，像上述的情况，会导致训练误差非常的低，验证误差非常的高，上述的情况我们称之为**传播失败（Propagation failures）**

# 第二周（5.12-5.19）学不完了

- [x] 神经正切核理论分析以及代码实现
- [ ] 收敛性分析
- [ ] 采样策略优化
- [ ] 自适应权重方法

沿着是什么？为什么？如何实现的路线进行学习。

# 二、PINNs高级理论（==重点==）

### 1.收敛性分析

目标：如果模型不收敛应该从哪些方面进行分析找原因

**输入**：我目前主要遇到的问题就是输入，输入如果量级相差太大或者相差太小模型会识别不出输入，比如量级较小，那么模型非常有可能将输入都视为同样的一个值。

**输出：**因为常用的损失是L2损失，所以输出的量级也会非常的影响结果，比如说直接设置边界上的条件为600℃，那么最初的损失会直接为600的平方（假设随机初始化后预测的温度为0℃左右）

**损失**：损失下降不平衡也会导致收敛问题，也就是说，在实际过程中会使得PDE损失下降的非常快，而其他损失下降的非常的慢。这可能会导致模型优化到一个错误的解上去。对于这个解到错误解上的问题，我们称其为**传播错误**。首先最简单的解决方案是加权，我们后续学习的一项重点在于**自适应**，也就是将整个网络设置为动态的网络，一些策略的问题让网络自己去学习

### 2.采样策略优化                                                                                                                                                                                                                                                                                                                                                                       

目标：采样对模型有什么影响？如何正确的进行采样

采样分为两点：1） 采样数量 ；2）采样的频率 （各个区域的采样频率）; 

文章连接：

[A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782522006260)

文章带读连接：[PINN论文精读（8）：Adaptive Sampling for PINN - 知乎](https://zhuanlan.zhihu.com/p/671933907)

代码连接：[lu-group/pinn-sampling: Non-adaptive and residual-based adaptive sampling for PINNs](https://github.com/lu-group/pinn-sampling)

从文章中我们应该要知道俩个相关的信息：

1.为什么会采样点的分布会导致出现上述问题

2.作者提出的解决方案，是从什么角度解决的

除了一般的采样算法之外，如随机采样，文中第一个提到的算法是**RAR-G**，这个采样策略是基于贪心算法，其实质是将残差值较大的采样点，保持不变，其他的点重新采样，这也就意味着

### 3.动态权重策略

作用：动态加权，实际上通过衡量loss的梯度的大小，进行加权，防止某一项下降的过快，或者过慢

原理理解：在训练过程中会出现这样的情况，训练的时候PDE损失下降的非常的快，从比较大的数量级下降到很小的数量级，但是对于我们的边界调价下降的非常的慢，对于这种情况非常容易将我们的解，解到错误的解，或者说是解到次优解上去，这是我们不愿意看到的。所以要对各项损失进行加权，那么如何进行加权？

**我这里有一个建议，损失之间的冲突问题再多任务学习中是一个较为常见的问题，可否从CV的多任务学习中获取相干灵感？**

#### 1.神经切向核理论（NTK）【基于梯度的加权方式】

要我总结就一句话：通过调整梯度实现，各个损失之间的平衡

代码实现：参考代码

我没有能力自己实现[数学不太OK]，看到有的文章中使用了此代码这里贴出来文章的连接

参考文章：**PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks**

文章连接：[[2307.11833\] PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks](https://arxiv.org/abs/2307.11833)

代码连接：[AdityaLab/pinnsformer](https://github.com/AdityaLab/pinnsformer)

参考文章：**PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs**

文章连接：[[2306.08827\] PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs](https://arxiv.org/abs/2306.08827)

代码连接：[i207M/PINNacle: Codebase for PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs.](https://github.com/i207M/PINNacle)

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

### 5.硬约束策略

对于一个比较复杂的网络可能存在多个边界条件，如果所有的边界条件都用网络进行训练，会出现冲突的情况，也就是说我们的损失项至少有：

<img src="https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250522102850089.png" alt="image-20250522102850089" style="zoom:80%;" />

其中$m_j$为各个损失边界损失项。N为其他的损失如PDE损失等。对于Neumann 边界条件和Robin边界条件，比较不好处理，如果想处理可以参考下面的文章，我的建议是对Dirichlet 边界条件进行硬约束：

![image-20250522103239909](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250522103239909.png)

![image-20250522103305973](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250522103305973.png)

![image-20250522103316221](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250522103316221.png)

![image-20250522103324968](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250522103324968.png)

硬约束的作用以及进一步的学习可以参考这篇文章[2210.03526](https://arxiv.org/pdf/2210.03526)

### 6.区域分解策略

最为经典的文章为FBPINNs

文章连接：[Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations | Advances in Computational Mathematics](https://link.springer.com/article/10.1007/s10444-023-10065-9)

代码连接：[benmoseley/FBPINNs: Solve forward and inverse problems related to partial differential equations using finite basis physics-informed neural networks (FBPINNs)](https://github.com/benmoseley/FBPINNs)

### 7.错误传播问题

最近看了许多有关于**错误传播**的问题

对于错误传播的问题的解决方案是出奇的一致，利用有限元的方法，用网格进行采样

### **8.因果关系**

有些文章讲了一个所谓的因果关系，最简单的因果关系，就是时序上的因果关系，也就是说拿前一个时刻的输出结果作为下一时刻的初始条件

# **第三周（5.24-6.8）**

- [ ] 多任务学习框架
- [ ] PINNs逆问题求解示例
- [ ] 将第二周没有搞完的东西搞完
- [ ] 可能实现的创新点举例
  - 自适应
  - 元学习
  - 多任务

- [ ] 神经算子概述

# 三、PINNs实际应用

1.多任务学习框架调研

2.PINNs求解逆问题示例

# 四、初探神经算子

神经算子概述

# **PINNs前沿发展**

这个先自己调研一下

### 文章1

**Challenges in Training PINNs: A Loss Landscape Perspective**

文章连接：[Challenges in Training PINNs: A Loss Landscape Perspective](https://arxiv.org/pdf/2402.01868)

代码连接：[pratikrathore8/opt_for_pinns](https://github.com/pratikrathore8/opt_for_pinns)

从本文中可以学习到一点，Adam+L-BFGS 在PINNs训练过程中的有效性，可以帮助我们进一步的**提高训练的精度**！

### 文章2

**Parameterized Physics-informed Neural Networks for Parameterized PDEs**

文章连接：[Parameterized Physics-informed Neural Networks for Parameterized PDEs](https://arxiv.org/pdf/2408.09446)

代码连接：[WooJin-Cho/Parameterized-Physics-informed-Neural-Networks](https://github.com/WooJin-Cho/Parameterized-Physics-informed-Neural-Networks)

后续做多个介质的PINNs任务的时候可以参考，**参数化PDE**这个东西还是比较重要的

### 文章3

**ProPINN: Demystifying Propagation Failures in Physics-Informed Neural Networks**

文章连接：[[2502.00803\] ProPINN: Demystifying Propagation Failures in Physics-Informed Neural Networks](https://arxiv.org/abs/2502.00803)

这篇文章在分析如何使用参考有限元的方法，解决**传播失败问题**

# 现阶段学习内容

日期：2025/5/12

想法：不能让知识局限于PINNs这个领域，而是要进行泛化的学习，什么意思？比如对于元学习的策略，不能说在PINNs中没有就不做了，而是要在其他的领域中，比如cv中找到对应的元学习策略，套用在PINNs的方法中。

- [ ] 认识什么是元学习，cv中是如何实现的，注意找时间比较靠前的文章！这种文章实现起来比较简单。是否实现？

日期：2025/5/20

总结归纳一下，现在手头上的热传导任务后续需要进行的尝试及工作应该如何进行

### 热传导尝试工作：

#### 1.优化策略

- [x] 归一化策略，包括输入和输出

  结论：输入归一化非常必要，用了上面的类似于min-max的归一化策略将输入值规范在了[-1,1]之间，通过这样的操作可以有如下的优势：

  - 训练过程中损失下降的更加的平滑
  - 训练过程中最后的损失更加的小

- [x] Adam+L-BFGS 优化器使用策略

  结论：非常有效的一种优化器使用方法，可以更有效的帮助收敛

  - Adam优化器是一阶优化器，可以帮助我们找到一个局部最优解的大致范围
  - L-BFGS 是二阶优化器，可以帮助我们找到更精确的解
  - [[2402.01868\] Challenges in Training PINNs: A Loss Landscape Perspective](https://arxiv.org/abs/2402.01868)

- [ ] 自适应采样策略

  先理解为什么要进行自适应的采样，在人为的设置采样部分，最后使用自适应采样

- [ ] 动态学习率策略

- [ ] 动态加权策略（神经正切核、config等都属于对于梯度的操作）

- [ ] **硬约束策略**（这一点后续可能比较重要）

#### 2.模型框架策略

- [x] MLP

  目前是将MLP作为我们的baseline去使用的，也是作为工程项目中最主要的网路框架

- [ ] Transformer

- [ ] KAN

- [ ] LSTM

- [ ] Mamba (这个是基于transformer的设想，这个做出来就是一个创新点) ==这个一定要做出来==

- [ ] 自适应的激活函数

相关框架可以有一个新的参考文章：

文章连接：[2502.00803](https://arxiv.org/pdf/2502.00803)

#### 3.后续尝试的工作

- [ ] 基于各种优化策略的调优，可能需要各种策略之间做组合，以及超参数的调整等操作
- [ ] 考虑不同多层介质的问题，即参数化PDE问题，以及PINNs的泛化能力考虑

依照目前我的理解不同的介质，实际上是分为两类，第一种问题就是不同的域，不同的材料所属的空间位置，第二个问题实际上是参数的问题，依旧是说实际上不同的介质对应的是不同的参数，类比于NS 方程中的雷洛系数不同导致流体的运动状态完全不同。所以这个参数化的PDE问题是比较值得深思的问题

- [ ] 多元边界条件问题

虽然名字叫做多元边界条件问题，实际上就是边界条件复杂问题，参考清华苏航老师团队的文章

#### 4.其他问题

- [ ] 加入初始条件应该是非常有必要的，但是在实践的时候却发现了一个比较大的问题，**加入初始条件没有什么效果**，按道理说我把初始条件加入到模型中去的话，应该在1s的时候损失是比较低的，这是一定的，但是实际上并不是在加入init_loss之后并没有将损失降低非常低，他依旧是最高的，进行了消融实验，**将初始条件作为消融的项目**，发现，初始条件几乎没有发挥作用，这是为什么？
- [ ] 现在的问题是我要设置一个共享的层，之前的想法都是每个层训练自己的，各个层之间的底层逻辑是相同的，现在我要将三个层之间加一个共享的层，同时保留各个层自己的特色，应该如何实现？
- [ ] 基于图结构的思想，我一直搞不明白一个问题，就是说网格化和非网格化的优势和劣势，也就是说如果有规则的网格我就可以用卷积等现有的视觉上面的方法进行处理，但是非网格化的数据就没有这个优势了，但是网格化的时候有两个问题，一就是网格区域的规则问题，规则网格体现不出来尺度信息，比如湍流在涡旋出是高信息量的，其他区域是低信息量的，那么对于这个问题，在有限元方法中采用的是划分不同粒度的网格，但是这会导致无法进行有效的卷积。二传统的PINNs算法会有没有**位置信息**的问题，或者说压根就不关注位置信息，因为是随机采样点，由于是随机采样，就是让网络去拟合函数，那么随机性就比较大，**有没有什么基于特征的信息，具有一些指导价值，这是值得考虑的点。**

# 其他

这个大佬准备些一个库，用于将所有的现有方法进行总结

[thuml/Neural-Solver-Library: A Library for Advanced Neural PDE Solvers.](https://github.com/thuml/Neural-Solver-Library)

推荐清华大学在PINNs方面工作的郝博士

郝博士主页：[Zhongkai Hao (郝中楷) - Homepage](https://haozhongkai.github.io/)

### 阅读推荐（一些我读过的感觉比较有用的文章）

PINNs综述：https://mp.weixin.qq.com/s/dpQlQUDAv3VoTjddLWAZLg

### 还没看但是准备看的文章

1.[Respecting causality for training physics-informed neural networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782524000690)

这篇文章说的是与时间相关的因果PINNs，应该是考虑了不同时间步之间的关系吧.

### 如何寻找相关论文文献

我其实不是非常建议从顶会顶刊上直接找文章，虽然说文章的质量高，但是许多都是从数学分析的角度去工作的，阅读难度大不说，而且对我的实际作用也非常的小，我觉得找一写基础的问题，一些能直接套用的问题，可能对于解决工程上的问题更有帮助。

我觉得从**公众号**，或者**期刊**，**会议**上找文章是可以的。但是，实际上最好的方法我觉得是**通过文章去搜索文章**，即找该文章所**引用的文章**。特别是大佬写的文章，一般应用的都是高质量的文章。

### PINNs自我baseline构建

构建一个属于自己的，自己完全清楚的baseline
