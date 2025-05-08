# PINNs学习文档

==初步文档后续需要整理==

![image-20250427164407913](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250427164407913.png)

# 第一周（4.27~5.9）

任务列表

- [ ] 1.神经网络框架
- [ ] 2.用deepxde构建简单的PDE方程并形成案例
- [ ] PINNs数学原理

# 一、PINNs入门课

## 1.PINNs入门介绍

从整体上体会，PINNs是什么？基本的学习内容是什么？

- [ ] 介绍视频一：[DeepOnet介绍_哔哩哔哩_ bilibili](https://www.bilibili.com/video/BV1dwzhYEEdf/?spm_id_from=333.337.search-card.all.click&vd_source=b056df9f055466883abc77833142c71a)

- [ ] 介绍视频二：[飞桨+DeepXDE全面支持科学计算多领域方程求解_哔哩哔哩_ bilibili](https://www.bilibili.com/video/BV18e411c7m7/?spm_id_from=333.337.search-card.all.click&vd_source=b056df9f055466883abc77833142c71a)

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

图示为二维瞬态热传导方程，我们只知道其偏微分方程(PDE)，但是不知道$T$具体的解析解，全连接神经网络的作用就是去拟合这个解析解$T$，由于pytorch等深度学习框架带有自动微分的功能，自动微分由计算图作为保证，这种性质自然的对PDE的计算是有利的，基于此得出一下的结论**（Q&A）**：

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

### 3.用pytorch+deepxde实现2D热传导问题

案例1：瞬态热传导问题

详情阅读参考代码网页链接，以及github链接中的README文件

参考代码：[PINN求解瞬态热传导问题](http://www.360doc.com/content/23/1220/08/83587350_1108171032.shtml)

Github链接：[2D_Heat](https://github.com/yanzikang/2D_Heat)

案例2：稳态热传导稳态

Github链接：[2D 多域稳态热传导](https://github.com/yanzikang/2D-heat-transfer)

### 3.PINNs数学原理

# 其他

这个大佬准备些一个库，用于将所有的现有方法进行总结

[thuml/Neural-Solver-Library: A Library for Advanced Neural PDE Solvers.](https://github.com/thuml/Neural-Solver-Library)

