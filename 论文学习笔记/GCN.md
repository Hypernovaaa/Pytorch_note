# 0.摘要
- 网络架构的一个趋势是用小的卷积核堆叠的方式来替换大的卷积核，因为在计算复杂度相同的情况下，小的卷积核更有效率。
> One of recent trends [31, 32, 14] in network architecture design is stacking small filters (e.g., 1x1 or 3x3) in the entire network because the stacked small filters is more efficient than a large kernel, given the same computational complexity.

- 但是语义分割是一个需要进行像素级预测的任务，大的卷积核在同时进行分类和预测任务时扮演了非常重要的角色。
> However, in the field of semantic segmentation, where we need to perform dense per-pixel prediction,we find that the large kernel (and effective receptive field) plays an important role when we have to perform the classification and localization tasks simultaneously.

- 同时提出了基于残差模块来改善物体边界的分割
> Following our design principle, we propose a Global Convolutional Network to address both the classification and localization issues for the semantic segmentation. We also suggest a residual-based boundary refinement to further refine the object boundaries.

# 1.引言
- 语义分割面临的两个挑战：
  - 分类：目标像素类别标签准确
  - 定位：被正确分类的像素坐标与gt保持一致

- 这两个任务具有天然的矛盾性：
  - 分类任务要求对于图片的变换或者旋转保持不变（不敏感）
  - 定位任务则要求对图像的变换敏感
  - 传统像素预测可能会降低网络性能

- GCN是能够同时解决两个问题的模型，GCN的两个设计原则：
  - 从定位的角度出发：网络是全连接的，并且舍弃全连接和 [全局池化](https://blog.csdn.net/jningwei/article/details/80064451)
  - 从分类的角度出发：应该采用大的卷积核，来使特征图和像素获取更密集的连接，加强了网络处理不同的图像变化的能力。
  - 为了减少参数采用了非对称卷积

- 文章贡献
# 2. 相关工作
- 回顾了fcn模型，并且本文基于fcn-based模型提出了三点改进

- 上下文信息的提取：
  - Zoom-out传统的手工构建层次化的上下文特征
  - Parse-net采用全局池化
  - Dilated-net全部采用空洞卷积
  - DeeplabV2采用ASPP模块整合上下文信息

- 扩大分辨率：
  - fcn使用反卷积
  - Deconvnet和SegNet采用反池化操作
  - LRR认为上采样特征图优于上采样分数图
  - Deeplab和Dilated-Net没有可学习的上采样过程

- 边界优化：
  - crf
  - deeplab使用denseCRF作为后处理
  - crf和rnn
  - **将crf和cnn相结合**

# 3.算法架构
## 3.1GCN网络
- 重述了语义分割中的两个任务：分类、定位。以及其本身的矛盾

- 由于侧重点的不同，分为了两种网络：
  - 分类网络：一般为锥形结构，特征图由相对较小的隐藏层提取，并且空间维度上比较粗糙，特征图和分类器为稠密的连接。
  - 定位网络：需要更大的特征图来编码位置信息，一般为桶型结构，并且采用反卷积、反池化、空洞卷积来生成高分辨率的特征图，随后分类器与特征图进行 **局部**的连接。

- 当前的state-of-the-art结果都是基于定位问题的，这对于像素的分类有可能是一种负优化。分类器与特征图进行的是局部连接而不是全局连接所以难以处理输入的各种形变。

- 基于以上的讨论，提出本文的架构：
  - 从定位的角度出发：网络应该是全卷积的并且没有全连接或者全局池化，这些操作会丢失位置信息。
  - 从分类的角度出发：为了生成更加稠密的连接，卷积核应该尽可能的大。最极端的情况，卷积核与输入大小相同，这就是全局卷积(global-convolution)
  - 没有直接用大的卷积核，而是用非对称卷积作为替代
  - 卷积层之后没有非线性激活层
  - 计算开销比直接用大的卷积核要小，为 O(2/k)
