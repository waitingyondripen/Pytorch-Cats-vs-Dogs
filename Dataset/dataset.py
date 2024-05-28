'''
@author wyr
@date 2024/05/24
@content 定义Dataset数据集,划分训练集、验证集、测试集
'''
import os 
import numpy as np
from PIL import Image #用于读取图像
from torchvision import transforms as T  #用于对数据进行预处理和数据增强
from torch.utils import data  # 包含 Dataset 和 Dataloader

# Dataset类是一个抽象类，用于表示数据集。
# 如果要加载自定义的数据，需要继承该类，并实现 __len__ 和 __getitem__ 方法，分别用于返回数据集的长度和获取单个样本（样本数据和样本标签）。

class DogCat(data.Dataset):
    def __init__(self, root, transforms = None, mode = None):
        '''
        root        表示图像所在的路径;
        transforms  表示数据处理的变换器;
        mode        表示当前数据所属的类型(训练集、验证集、测试集);
        调用该函数可以实现训练集、验证集、测试集的处理和生成
        '''
        assert mode in ['train', 'val', 'test'] #如果mode不是此表中的元素，则报错
        self.mode = mode
        #对于每一个在root路径下的图像，将路径与图像文件名结合在一起作为图像的表示
        imgs = [os.path.join(root, img) for img in os.listdir(root)] 

        #将数据按照名称中的序号位进行排序
        #其中的lambda表示Python中的一种匿名函数，它允许在需要函数对象的任何地方使用简短的函数定义
        #当 sorted() 函数调用 lambda 函数时，它会将列表中的每个元素作为参数传递给 lambda 函数（即x表示imgs中的每个路径字符串）
        if mode == 'test':
            imgs = sorted(imgs, key= lambda x : int(x.split('.')[-2].split('/')[-1]))  #测试集的名称示例：/Images/test/1.jpg
        else:
            imgs = sorted(imgs, key= lambda x : int(x.split('.')[-2])) #训练集和验证集的名称示例：/Images/train/cat.0.jpg

        #获取imgs的长度，用于分割训练集和验证集
        imgs_num = len(imgs)

        #划分训练集、验证集、测试集，其中训练集和验证集的比例是3：7
        if self.mode == 'test': self.imgs = imgs
        if self.mode == 'train': self.imgs = imgs[:int(imgs_num * 0.7)] #从0取到int(imgs_num * 0.7)作为训练集数据
        if self.mode == 'val':self.imgs = imgs[int(imgs_num * 0.7):] #从int(imgs_num * 0.7)取到结尾作为验证集数据

        #划分好数据之后，需要对数据进行transforms处理
        if transforms is None:
            #数据转换操作
            normalization = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #标准化操作，用在之后的transform模块中
            #验证集和测试集不需要进行数据加强
            if self.mode == 'test' or self.mode == 'val':
                self.tranforms = T.Compose([
                    T.Resize(224),  #将图像大小调整为224*224
                    T.CenterCrop(224), #以图像中心作为裁减中心向外裁减224大小
                    T.ToTensor(), #将图像转化为张量
                    normalization #调用标准化操作
                ])
            else:
                #训练集需要进行数据加强
                self.tranforms = T.Compose([
                    T.Resize(256),  #将图像大小调整为256*256
                    #该操作通常用于数据增强，以增加模型对图像尺度和位置变化的鲁棒性
                    T.RandomResizedCrop(224),  #它会在图像中随机选择一个区域并裁减出224的大小
                    #用于随机水平翻转图像
                    T.RandomHorizontalFlip(), #这个操作可以增加数据的多样性，可以有效地增加数据量。它会以一定的概率（默认为0.5）对图像进行水平翻转。
                    T.ToTensor(), #将图像转化为张量
                    normalization #进行标准化操作
                ])
        #至此,数据集划分及处理任务完成
    
    #继承Dataset类，必须要重写一个__getitem__函数
    def __getitem__(self, index):
        '''
        index   表示待返回图像的索引；
        该函数返回一张图像的数据(包括图像数据data和标签label);
        对于训练集和验证集来说,若为狗则标签为1,若为猫则标签为0;
        测试集不需要标签,则只需要返回图像ID即可
        '''

        #获取标签
        if self.mode == 'test':
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in self.imgs[index].split('/')[-1] else 0

        #获取图像数据
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.tranforms(data) #将图像数据进行变换操作

        #返回图像数据和标签信息
        return data, label
    
    #继承Dataset类，必须要重写一个__len__函数
    def __len__(self):
        '''
        该函数返回数据集中所有图像的数量
        '''
        return len(self.imgs)
        

