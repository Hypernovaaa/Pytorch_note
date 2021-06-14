# 1.数据输入
- [导入其他模块或者包报错的问题](https://blog.csdn.net/sinat_32336967/article/details/105058577)：将当前文件夹标记为Sources Root,其作用是将文件目录加入到sys.path中（参考python [文件导入的路径查找顺序](https://www.cnblogs.com/tulintao/p/11196893.html)）。PyCharm打开的当前文件夹不用标记，默认自动加入到sys.path中。
## 1.1 dataset
自己实现的dataset类都继承于`torch.utils.data.Dataset`并且要复写其中的`__getitem__`方法，实现接收一个样本返回一个索引的功能。  
训练集dataset的实例化：
`Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
`参数1：路径列表，2：裁剪尺寸。参数如下：
```python
TRAIN_ROOT = './CamVid/train'  #数据路径
TRAIN_LABEL = './CamVid/train_labels'  #标签路径
crop_size = (352, 480)  #图片的裁剪尺寸
```
### 1.1.1 __init__函数
```python
self.imgs = self.read_file(self.img_path)
self.labels = self.read_file(self.label_path)
```
