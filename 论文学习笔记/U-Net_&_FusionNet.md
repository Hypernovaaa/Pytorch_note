# 1.两篇论文的基本信息  
U-Net和FusionNet都是医学分割的模型
![](assets/U-Net_&_FusionNet-8832641b.png)  
![](assets/U-Net_&_FusionNet-02312e4b.png)  
# 2.论文的创新点以及成就
![](assets/U-Net_&_FusionNet-f4be7d8c.png)
# 3.摘要部分
## 3.1 U-Net摘要
- 提出了一种可以通过数据增强更高效的利用可用样本的网络和训练策略。
>  we present a network and training strategy that relies on the strong
use of data augmentation to use the available annotated samples more
efficiently  

 - 这种网络架构由一个用来捕获上下文信息的收缩路径，和一个用来精确定位的对称的扩张路径组成。(提出了编码器和解码器的概念)
 >  The architecture consists of a contracting path to capture
context and a symmetric expanding path that enables precise localiza-
tion   

- 此模型可以通过非常少的样本来做端到端的训练，同时得到了2015年非常好的效果，ISBI挑战赛。  
> We show that such a network can be trained end-to-end from very
few images and outperforms the prior best method (a sliding-window
convolutional network) on the ISBI challenge for segmentation of neu-
ronal structures in electron microscopic stacks  

- 除此之外，模型比较快，512x512的图片在2015年的cpu上分割时间少于一秒钟。
>  Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU.
