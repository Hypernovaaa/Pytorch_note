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

# 2.引言
## 2.1SegNet引言
- cnn的发展和应用
- 最近的一些方法是直接吧用于类别预测的深层架构用于图像分割。虽然取得了不错的结果但是结果图还是有些粗糙的，主要是因为采用了最大值池化和子采样操作降低了特征图的分辨率。
>  However, some of these recent approaches have tried to directly adopt deep archi-tectures designed for category prediction to pixel-wise labelling[7]. The results, although very encouraging, appear coarse [3].This is primarily because max pooling and sub-sampling reduce feature map resolution.

- SegNet模型用来解决精准的边界定位问题
> Our motivation to design SegNet arises from this need to map low resolution features to input resolution for pixel-wise classification. This mapping must produce features which are useful for accurate boundary localization.

- 道路场景的语义分割要求模型具有外观建模，理解形状，理解不同类别间空间关系的能力。
>  It is primarily motivated by road scene understanding applications which require the ability to model appearance (road, building), shape (cars,pedestrians) and understand the spatial-relationship (context) between different classes such as road and side-walk.

- 在道路场景中，模型应具有1.生成平滑边界的能力 2.具有勾画小尺寸物体的能力。所以在特征图中保留边界信息非常重要。
>  In typical road scenes, the majority of the pixels belong to large classes such as road, building and hence the network must produce smooth segmentations. The engine must also have the ability to delineate objects based on their shape despite their small size. Hence it is important to retain boundary information in the extracted image representation.

- 模型如果可以端到端的进行训练，就可以很方便的使用优化器（算是一个优势）
- SegNet的关键组成部分是与每一个编码器对应的解码器部分。采用了最大池化索引来进行非线性的上采样。  
>  The key component of SegNet is the decoder network which consists of a hierarchy of decoders one corresponding to each encoder. Of these, the appropriate decoders use the max-pooling indices received from the corresponding encoder to perform non-linear upsampling of their input feature maps.

- 反池化的优点：
  - 更好的勾画了边界
  - 减少了端到端训练的参数
  - 这种上采样形式可以被集成到任何encoder-decoder架构的网络中  
>  (i) it improves boundary delineation , (ii) it reduces the number of parameters enabling end-to-end training, and (iii) this form of upsampling can be incorporated into any encoder-decoder architecture such as [2], [10] with only a little modification.

- 最近的用于分割的深层架构一般具有相同的编码结构，例如vgg。在解码网络、训练、推理中有所不同。另一个相同的特征是他们具有数以亿计的可训练参数，这使其难以进行端到端的训练。对此问题有以下解决方法。
  - 多段训练
  - 增加预训练
  - 增加辅助编码（区域）帮助推理
  - 数据增强进行预训练，或者是全训练
  - 后处理技巧例如crf
>  Most recent deep architectures for segmentation have identical encoder networks, i.e VGG16, but differ in the form of the decoder network, training and inference. Another common feature is they have trainable parameters in the order of hundreds of millions and thus encounter difficulties in performing end-to-end training [4]. The difficulty of training these networks has led to multi-stage training [2], appending networks to a pre-trained architecture such as FCN [10], use of supporting aids such as region proposals for inference [4], disjoint training of classification and segmentation networks [18] and use of additional training data for pre-training [11] [20] or for full training [10]. In addition,performance boosting post-processing techniques [3] have also been popular. Although all these factors improve performance on challenging benchmarks [21], it is unfortunately difficult from their quantitative results to disentangle the key design factors necessary to achieve good performance

- Pascal VOC数据集中有少数foreground与background有明显区分，这让一些投机者可以使用类似于边缘检测来刷分数。因此本文使用了Camvid，Sun RGBD这两个数据集，而不是用PascalVOC数据集
>  the majority of this task has one or two foreground classes surrounded by a highly varied background. This implicitly favours techniques used for detection as shown by the recent work on a decoupled classification-segmentation network [18] where the classification network can be trained with a large set of weakly labelled data and the independent segmentation network performance is improved

## 2.2DeconvNet引言
- cnn的发展和应用
- 由cnn引出fcn的相关介绍
- 基于fcn的算法有一些限制；
  - 网络具有固定的感受野，导致过大的物体会被分割，小的物体会被忽略分为背景。
  - 由于输入到解码器中的特征图过于粗糙，上采样形式简单粗暴，造成了物体边界细节丢失
>  First, the network has a pre- defined fixed-size receptive field. Therefore, the object that is substantially larger or smaller than the receptive field may be fragmented or mislabeled. In other words, label prediction is done with only local information for large objects and the pixels that belong to the same object may have inconsistent labels as shown in Figure 1(a). Also, small objects are often ignored and classified as background, which is illustrated in Figure1(b).  

>  Second, the detailed structures of an object are often lost or smoothed because the label map, input to the deconvolutional layer, is too coarse anddeconvolution procedure is overly simple.

本文的主要创新点：
- 学习了一个深层的反卷积网络
>  We learn a deep deconvolution network, which is composed of deconvolution, unpooling, and rectified linear unit (ReLU) layers. Learning deep deconvolution networks for semantic segmentation is meaningful but no one has attempted to do it yet to our knowledge.

- 将训练的网络用于独立的目标区域来获取实例分割，最后在进行组合
> The trained network is applied to individual objec tproposals to obtain instance-wise segmentations, which are combined for the final semantic segmentation; it is free from scale issues found in the original FCN-based methods and identifies finer details of an object.

- 将本文网络和fcn进行融合取得了非常好的效果。
> We achieve outstanding performance using the deconvolution network trained on PASCAL VOC 2012 augmented dataset, and obtain the best accuracy through the ensemble with [19] by exploiting the heterogeneous(异质性) and complementary characteristics of our algorithm with respect to FCN-based methods.
