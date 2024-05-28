'''
@author wyr
@date 2024/05/25
@content  训练函数、验证函数、测试函数、帮助函数
'''

import os
import torch
import torch.utils
import torch.utils.data
import Models
import fire
import matplotlib.pyplot as plt
from config import opt #获取默认参数
from Dataset import DogCat
from Utils import Visualize
from torchnet import meter
from tqdm import tqdm



def train(**kwargs):

    #根据命令行更新参数
    opt.parse(kwargs)
    
    #创建可视化窗口
    #vis = Visualize(opt.env)
    
    #------------------第一步：加载模型------------------#
    model = None
    #如果路径存在，就加载之前训练出的模型
    if opt.load_model_path:
        model = torch.load(opt.load_model_path)
    #否则加载预训练模型
    else:
        model = getattr(Models, opt.model)()
    #将模型赋给运算设备
    model.to(opt.device)
        
    #------------------第二步：完成数据加载------------------#
    #加载训练数据集
    train_data = DogCat(opt.train_data_root, mode='train')
    #加载验证数据集
    val_data = DogCat(opt.train_data_root, mode='val')
    #加载train_dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, 
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers)
    #加载验证数据集
    val_dataloader = torch.utils.data.DataLoader(dataset=val_data,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 num_workers=opt.num_workers)
    
    #------------------第三步：设定损失函数和优化器------------------#
    criterion = torch.nn.CrossEntropyLoss() #使用交叉熵损失函数
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay) #获取模型设定的优化器（此处用的是Adam）

    #------------------第四步：统计指标------------------#
    #一个用于计算平均损失值的容器。它可以用于存储每个批次的损失值，并计算它们的平均值
    #这在训练过程中通常用于监控损失值的变化趋势，并评估模型的性能
    loss_meter = meter.AverageValueMeter()      

    #一个混淆矩阵容器，用于评估分类模型的性能
    #混淆矩阵用于统计模型在测试数据上的分类结果，以便评估模型在不同类别上的准确率、召回率等性能指标
    confusion_matrix = meter.ConfusionMeter(2)

    #用于存储上一个时刻的损失值
    previous_loss = 1e100

    #------------------第五步：训练------------------#
    for epoch in range(opt.max_epoch):
        #重置平均损失计算容器
        loss_meter.reset()
        #重置混淆矩阵容器
        confusion_matrix.reset()
        #将神经网络模型置于训练状态
        model.train()
        history = {'train_loss':[], 'train_accuracy':[]}
        processbar = tqdm(train_dataloader, unit='step')
        for step, (data, label) in enumerate(processbar):
            #将data和label赋给运算设备device
            data = data.to(opt.device)
            label = label.to(opt.device)
            #将梯度清零
            model.zero_grad()
            #将数据输入网络
            score = model(data)
            # #求解损失
            loss = criterion(score, label)
            #反向传播
            loss.backward()
            #梯度优化
            optimizer.step()

            #更新平均损失计算容器
            loss_meter.add(loss.item())
            #更新混淆矩阵容器
            #记得detach一下,更保险一点
            #detach()方法用于从计算图中分离张量,由于新张量不再与计算图相关联，因此任何基于新张量的操作都不会在计算图中记录梯度信息。
            confusion_matrix.add(score.detach(), label.detach())
            
            #在vis中绘制图像
            if (step+1)%opt.print_freq == 0:
                #loss_meter.value()的返回值为(mean, std)
                #vis.plot('loss', loss_meter.value()[0])
                history['train_loss'].append(loss.item())
                #获取混淆矩阵的值
                cm_value = confusion_matrix.value()
                #计算准确率
                accuracy = 100. * ((cm_value[0][0] + cm_value[1][1]) / cm_value.sum())     
                history['train_accuracy'].append(accuracy)

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()
            #进度条输出
            processbar.set_description("[ TEST PART ][ %d / %d ], Loss:%.4f"%(epoch+1, opt.max_epoch, loss.item()))
        
        processbar.close()


        model.save()
        
        # validate and visualize
        val_cm,val_accuracy = val(model,val_dataloader)
        print("当前轮次下验证集的正确率为%.4f"%(val_accuracy))
        #绘制准确率曲线
        #vis.plot('val_accuracy',val_accuracy)
        #记录准确率日志
        #vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #            epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=opt.lr))
        
        # 更新学习率
        #如果当前的loss平均值大于前一次的loss平均值,则降低学习率
        if loss_meter.value()[0] > previous_loss:          
            opt.lr = opt.lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
        
        previous_loss = loss_meter.value()[0]
    
    #draw(history['train_loss'], history['train_accuracy'])

@torch.no_grad()
def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    #将模型切换到验证状态
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    processbar = tqdm(dataloader, unit='step')
    for step, (data, label) in enumerate(processbar):
        #将数据转换到计算设备device上
        val_input = data.to(opt.device)
        #获取网络输出
        score = model(val_input)
        #更新混淆矩阵
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))
        #进度条设置
        processbar.set_description("[ VALIDATION PART ]")



    #完成验证后,将网络切换回训练模式
    model.train()
    #获取混淆矩阵的值
    cm_value = confusion_matrix.value()
    #计算准确率
    accuracy = 100. * ((cm_value[0][0] + cm_value[1][1]) / cm_value.sum())
    #返回混淆矩阵和准确率
    return confusion_matrix, accuracy


@torch.no_grad()
def test(**kwargs):
    # 更新配置文件
    opt.parse(kwargs)
    # 加载模型并输入到计算设备上
    model = None
    if opt.load_model_path:
        model = torch.load(opt.load_model_path)
    else:
        model = getattr(Models, opt.model)().eval()
    model.to(opt.device)
    #加载数据
    test_data = DogCat(opt.test_data_root, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    results = []
    for ii,(data, label) in enumerate(tqdm(test_dataloader)):
        data = data.to(opt.device)
        score = model(data)
        #计算每个样本属于狗的概率
        probabilities = torch.nn.functional.softmax(score, dim=1)[:,1].detach().tolist()
        #保存批次结果
        batch_results = [(label_.item(), probabilities_) for label_, probabilities_ in zip(label, probabilities)]
    
        #每个batch结束之后都需要保存一下结果
        results += batch_results

    #将预测概率写到目标输出文件中
    write_csv(results, opt.result_file)

    return results



def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
    
def help():
    '''
    打印帮助的信息： python file.py help
    '''
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

def draw(lista, listb):
    #绘制训练损失变化曲线
    plt.plot(lista, label = 'Train_Loss')
    plt.legend('best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./Train_Loss.png', dpi = 300)
    plt.close()

    #绘制训练准确率变化曲线
    plt.plot(listb, label = 'Train_Accuracy')
    plt.legend('best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('./Train_Accuracy.png', dpi = 300)
    plt.close()


if __name__ == '__main__':
    fire.Fire()