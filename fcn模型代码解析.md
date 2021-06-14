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
### 1.1.1__init__函数
初始化dataset类中的参数。读取数据`self.imgs` `self.labels`，同时设定裁剪尺寸`self.crop_size`。
```python
def __init__(self, file_path=[], crop_size=None):
  self.img_path = file_path[0]
  self.label_path = file_path[1]
  self.imgs = self.read_file(self.img_path)#调用self.read_file
  self.labels = self.read_file(self.label_path)
  self.crop_size = crop_size
```  
`self.labels和self.imgs`分别为所有的标签和训练数据的路径的列表。例如`'./CamVid/train\\0001TP_006690.png'`
### 1.1.2__getitem__函数
功能：输入训练图片和标签的索引，经过中心裁剪、img_transform处理，将训练图片和标签打包为字典返回。
```python
def __getitem__(self, index):
    img = self.imgs[index]
    label = self.labels[index]
    # 从文件名中读取数据（图片和标签都是png格式的图像数据）
    img = Image.open(img)
    label = Image.open(label).convert('RGB')#图片打开为PIL格式

    img, label = self.center_crop(img, label, self.crop_size)
    img, label = self.img_transform(img, label)

    sample = {'img': img, 'label': label}
    return sample
```
`PIL.Image.open`PIL是python中处理图片的库
### 1.1.3__len__(self)
### 1.1.4 center_crop 中心裁剪函数
功能：裁剪输入PIL图片或者Tensor的大小,返回值也是PIL或者是Tensor格式
```python
def center_crop(self, data, label, crop_size):
    data = ff.center_crop(data, crop_size)
    label = ff.center_crop(label, crop_size)
    return data, label
```
[center_crop](https://pytorch.org/vision/stable/transforms.html?highlight=center_crop#torchvision.transforms.functional.center_crop)


### 1.1.5 read_file(self, path)
功能：输入一个路径，返回其路径下所有文件及文件夹的路径的列表
```python
def read_file(self, path):
  files_list = os.listdir(path)#files_list为输入路径下的文件或者文件夹名字的列表
  file_path_list = [os.path.join(path, img) for img in files_list]
  file_path_list.sort()
  return file_path_list
```
[listdir](https://docs.python.org/3/library/os.html?highlight=os%20listdir#os.listdir)
### 1.1.6 img_transform函数
功能：输入输出都是PIL格式的图片和标签，对图片和标签做些数值处理
```python
def img_transform(self, img, label):
    label = np.array(label)  # 以免不是np格式的数据
    label = Image.fromarray(label.astype('uint8'))
    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    img = transform_img(img)
    label = label_processor.encode_label_img(label)
    label = t.from_numpy(label)

    return img, label
```
