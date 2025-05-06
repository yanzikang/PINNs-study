# PINNs学习文档

==初步文档后续需要整理==

![image-20250427164407913](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250427164407913.png)

## 第一周（4.27~5.9）

任务列表

- [ ] 1.神经网络框架
- [ ] 2.用deepxde构建简单的PDE方程并形成案例
- [ ] PINNs数学原理

## 一、PINNs入门课

### 1.PINNs入门介绍

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

### 2.神经网络框架

#### 1.基本知识介绍

PINNs通过在损失函数中引入物理方程（如偏微分方程），使神经网络在训练过程中不仅依赖数据，还遵循已知的物理规律。

![image-20250506151412828](https://yzk-ahu-1330633968.cos.ap-guangzhou.myqcloud.com/imgs/image-20250506151412828.png)



#### 2.pytorch+deepxde框架学习

#### 3.用pytorch+deepxde实现2D热传导问题

详情阅读参考代码网页链接，以及github链接中的README文件

参考代码：[PINN求解瞬态热传导问题](http://www.360doc.com/content/23/1220/08/83587350_1108171032.shtml)

Github链接：[2D_Heat](https://github.com/yanzikang/2D_Heat)

### 3.PINNs数学原理

