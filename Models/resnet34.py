import torchvision
from .BasicModel import BasicModel
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet34

class ResNet34(BasicModel):

    def __init__(self, output_nums=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'
        self.model = resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        for params in self.model.parameters():
            params.requires_grad_(False)    #将所有层的梯度计算设置为False，此时所有参数都不会被反向传播更新
        self.model.fc = nn.Linear(in_features=512, out_features=output_nums, bias=True) #修改最后的输出层，此时该层的equires_grad自动变为true

    def forward(self, x):
        return self.model(x)
    
    def get_optimizer(self, lr, weight_decay):
        #只更新最后一层的参数即可
        #前面的特征提取部分可以保持不变
        return Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)
