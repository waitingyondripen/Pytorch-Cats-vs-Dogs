import torch
import time

class BasicModel(torch.nn.Module):
    '''
    封装了nn.Module,主要提供load和save两种方法
    '''
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self)) #默认的模型名字

    def load(self, path):
        '''
        path表示模型的路径位置;
        可加载指定路径下的模型
        '''
        self.load(path)

    def save(self, name = None):
        '''
        name表示想要保存的模型的名称;
        保存模型到指定路径
        '''
        if name is None:
            prefix = 'Checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self, name)

        return name
    
    def get_optimizer(self, lr, weight_decay):
        '''
        lr表示Adam的学习率;
        weight_decay表示权重衰减参数,权重衰减是一种正则化形式，惩罚模型中的大权重，有助于防止过拟合
        '''
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

class Flat(torch.nn.Module):
    '''
    将输入reshape成(batch_size, dim_length)的形式
    '''
    def __init__(self) :
        super(Flat, self).__init__()

    def forward(self, x):
        return x.reshape(x.size()[0], -1)