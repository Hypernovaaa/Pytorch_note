# 1.摘要
## 1.1SegNet摘要
- 网络结构为编码器，对应的解码器，像素级的分类层（指softmax），网络的编码器的拓扑结构和VGG16是相同的。
>  This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network

- 解码器的作用是将编码器输出的低分辨率的特征图映射为输入特征图分辨率，以进行像素分类。
> The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification.

- SegNet的创新性在于其解码器对低分辨率输入特征图的上采样的方法。
  - 用了最大值池化索引来进行一个非线性的上采样。消除了上采样中的学习过程。
  - 生成的特征图是稀疏的，随后通过卷积来生成稠密的特征图。
> The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s).Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps.

- SegNet主要用于场景理解任务，所以网络无论是在内存开销还是推理时间都很有效率。相比于对比中的其他模型，此模型结构的可训练参数非常少，同时可以用sgd进行端到端的训练
> SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures and can be trained end-to-end using stochastic gradient descent.

## 1.2DeconvNet摘要
- DeconvNet是在vgg-16模型上做的改进
> We learn the network on top of the convolutional layers adopted from VGG
16-layer net.

- DeconvNet由反卷积和反池化组成，用来预测像素类别和分割掩码。最后生成的特征图是每个区域训练后的简单结合。
> The deconvolution network is composed of deconvolution and unpooling layers, which identify pixel-wise class labels and predict segmentation masks. We apply the trained network to each proposal in an input image, and construct the final semantic segmentation map by combining the results from all proposals in a simple manner.

- DeconvNet通过融合深度反卷积网络和块预测，减少了基于fcn的算法的限制。对于多尺度的目标都有很好的作用。
> The proposed algorithm mitigates the limitations of the existing methods based on fully convolutional networks by integrating deep deconvolution network and proposal-wise prediction; our segmentation method typically identifies detailed structures and handles objects in multiple scales naturally.
