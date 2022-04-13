import pandas as pd
import os
import torch
from torch.nn import Module
import random
import numpy as np
import matplotlib.pyplot as plt
import xlwt


def data_deal():
    os.chdir('G:\爬虫练习\深度学习\动手学深度学习')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_df.drop('Id',axis=1,inplace=True)  # (1460, 80)
    test_df.drop('Id',axis=1,inplace=True)   # (1459, 79)
    data_df = pd.concat([train_df,test_df],axis=0,join='inner')
    data_df.iloc[:,5] = data_df.iloc[:,5].astype(str)
    data_df.iloc[:,56] = data_df.iloc[:,56].astype(str)
    data_df.iloc[:,71] = data_df.iloc[:,71].astype(str)
    data_df.iloc[:,72] = data_df.iloc[:,72].astype(str)
    data_df.iloc[:,73] = data_df.iloc[:,73].astype(str)
    for i in range(data_df.shape[1]):
        if type(data_df.iloc[0,i]) is str:
            continue
        else:
            data_df.iloc[:,i] = data_df.iloc[:,i].fillna(0.0)
            data_df.iloc[:,i] = (data_df.iloc[:,i] - data_df.iloc[:,i].mean()) / data_df.iloc[:,i].std()
    data_df = pd.get_dummies(data_df,dummy_na=True)
    # 保存为csv
    data_df.to_csv('data.csv')

# 生成bath_size大小的数据
def bath_data(bath_size,feature,label):
    num_exp = len(label)
    index = list(range(num_exp))
    random.shuffle(index)
    for i in range(0,num_exp,bath_size):
        bath_index = torch.tensor(index[i:min(i + bath_size,num_exp)])
        yield feature[bath_index,:],label[bath_index]

def get_data():
    os.chdir('G:\爬虫练习\深度学习\动手学深度学习')
    train_df = pd.read_csv('train.csv')
    data_df = pd.read_csv('data.csv')
    all_x = data_df.iloc[0:1460,1:].values
    all_y = train_df.iloc[:,80].values
    train_x = torch.tensor(all_x[0:1022,:]).to(torch.float32)
    train_y = torch.tensor(all_y[0:1022]).to(torch.float32)
    val_x= torch.tensor(all_x[1022:,:]).to(torch.float32)
    val_y = torch.tensor(all_y[1022:]).to(torch.float32)
    return train_x,train_y.reshape([-1,1]),val_x,val_y.reshape([-1,1])

class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.dense1 = torch.nn.Linear(336,56)
        self.dense2 = torch.nn.Linear(56,28)
        self.dense3 = torch.nn.Linear(28,1)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

class process_data():
    def __init__(self):
        self.loss = []
        self.step = 0
    def add(self,x):
        self.loss.append(x)
        self.step = self.step + 1

def train_epoch(net,loss,optimizer,train_x,train_y,bath_size,process):
    for x,y in bath_data(bath_size,train_x,train_y):
        if isinstance(net,torch.nn.Module):
            net.train()
        l = loss(net(x),y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        with torch.no_grad():
            process.add(log_mse(net(x),y))

def init_params(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

# 定义评测指标
def log_mse(y_pre,y_true):
    loss = torch.nn.MSELoss()
    clipped_preds = torch.clamp(y_pre, 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(y_true)))
    return rmse.item()

def train():
    epochs = 50
    train_x,train_y,val_x,val_y = get_data()   # train_x (1022, 336)  train_y (1022,)
    net = MyModule()
    net.apply(init_params)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01,weight_decay=0)
    bath_size = 64
    process = process_data()
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print('开始第{}轮训练...'.format(epoch + 1))
        train_epoch(net,loss,optimizer,train_x,train_y,bath_size,process)
        if isinstance(net,torch.nn.Module):
            net.eval()
        val_l = log_mse(net(val_x),val_y)
        train_l = log_mse(net(train_x),train_y)
        val_loss.append(val_l)
        train_loss.append(train_l)
        print('第{}轮训练损失为：{}'.format(epoch + 1,train_l))
        print('第{}轮测试损失为：{}'.format(epoch + 1,val_l))
    # 保存训练参数
    torch.save(net.state_dict(),'mlp.params')
    # 显示训练时每个step的损失变化
    process_show(process)
    # 显示每轮训练和测试误差的变化
    x_line = np.arange(epochs)
    fig,ax = plt.subplots()
    ax.plot(x_line,train_loss,'b',label='train_loss')
    ax.plot(x_line,val_loss,'r',label='val_loss')
    ax.legend()
    plt.show()

def process_show(process):
    print(process.step)
    x_line = np.arange(process.step)
    loss = process.loss
    fig,ax = plt.subplots()
    ax.plot(x_line,loss)
    plt.show()

def test():
    net = MyModule()
    os.chdir('G:\爬虫练习\深度学习\动手学深度学习')
    net.load_state_dict(torch.load('mlp.params'))
    net.eval()
    data_df = pd.read_csv('data.csv')
    test_x = torch.tensor(data_df.iloc[1460:,1:].values).to(torch.float32)
    y_pre = net(test_x).detach().numpy()
    test_data = pd.read_csv('test.csv')
    test_data.loc[:,'price'] = y_pre
    test_data.to_csv('test_pre.csv')



test()









