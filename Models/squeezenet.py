import torchvision
from torchvision.models import squeezenet1_1
from .BasicModel import BasicModel
from torch import nn
from torch.optim import Adam

class SqueezeNet(BasicModel):
    def __init__(self, output_nums = 2):
        '''
        output_num表示输出的数据个数;
        初始化导入SqueezeNet网络并修改最终的网络层输出信息
        '''
        super(SqueezeNet, self).__init__()
        self.model_name = 'squeezenet'      #定义模型名称 
        self.model = squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)       #加载模型网络结构和参数（预训练后）
        #Attention : 也可以按照如下的形式进行定义，直接修改最终输出结果有2个
        #但是此时整个模型不会加载相应的参数，因此在此不采用这种方式
        #self.model = squeezenet1_1(pretrained = False, num_classes = 2)
        self.model.num_classes = output_nums  #定义模型的输出类别
        #整个网络结构可以通过print model查看
        #令起一个test.py,在其中调用model = squeezenet1_1(pretrained = True),然后更改最后一层即可
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, output_nums, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU(inplace=True), #这里特别注意nn.ReLU和nn.functional.relu的区别
            #nn.ReLU是一个层，它继承自torch.nn.Module
            #nn.functional.relu是一个函数，需要向其传递张量
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        # self.model = squeezenet1_1(weights=None, num_classes=output_nums)


    def forward(self, x):
        '''
        x 表示传递到模型的参数
        最终返回经过神经网络之后得到的输出
        '''
        return self.model(x)
    
    def get_optimizer(self, lr, weight_decay):
        '''
        重写BasicModel的get_optimizer函数
        因为是预加载模型，因此只需要训练最后一层的相关参数即可
        '''
        return Adam(self.model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        # return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
