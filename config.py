'''
@author wyr
@date 2024/05/25
@content 配置文件，所有可配置的变量都集中于此，并提供默认值
'''
import warnings
import torch

class DefaultConfig(object):
    env = 'default'         # visdom环境
    vis_port = 8097         # visdom端口
    model = 'ResNet34'    #使用的模型，名字必须与名字必须与models/__init__.py中的名字一致

    train_data_root = '/data/jyc/cats_vs_dogs/train/'     #训练集存放路径
    test_data_root = '/data/jyc/cats_vs_dogs/test/'       #测试集存放路径
    load_model_path = None                  #加载与训练模型的路径，为None表示不加载

    batch_size = 128         #batch size
    use_gpu = True          #是否使用GPU
    num_workers = 8         #加载数据时使用多少个工作单位
    print_freq = 4         #每N个batch打印一次信息

    debug_file = 'tmp/debug'        ## if os.path.exists(debug_file): enter ipdb
    result_file = 'result_ResNet34.csv'      #存放最终结果的文件

    max_epoch = 20          #训练轮数
    lr = 0.01                #初始化学习率
    lr_decay = 0.95         #学习率衰减，lr = lr * lr_decay，随着训练的进行逐渐减小学习率的大小，使得模型在训练后期更容易收敛到全局最优解而不是在最优解附近振荡
    weight_decay = 1e-4     #权重衰减，一种正则化技术，通过向损失函数添加一个惩罚项来减小模型的权重值，以防止过拟合，提高模型的泛化能力

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        @使用示例
        @opt = DefaultConfig()
        @new_config = {'lr':0.1, 'use_gpu':False}
        @opt.parse(new_config)
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device = torch.device('cuda')  if opt.use_gpu else torch.device('cpu')

        # 打印配置信息	
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
        
        

#测试代码    
opt = DefaultConfig()
# new_config = {'lr':0.1, 'use_gpu':False}
# opt.parse(new_config)
# print(opt.lr)