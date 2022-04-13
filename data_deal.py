import torch
import numpy as np
import os
import pandas as pd

def write_data():
    os.chdir('G:\爬虫练习\深度学习\动手学深度学习')
    with open('house_tiny.csv','w') as f:
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
        f.close()

def load_data():
    data = pd.read_csv('house_tiny.csv')
    print(data)
    input,output = data.iloc[:,0:2],data.iloc[:,2]
    input = input.fillna(input.mean())
    input = pd.get_dummies(input,dummy_na=True)
    print(input)

def grad():
    x = torch.arange(4.0,requires_grad=True)
    y = 2 * torch.dot(x,x)
    u = y.detach()
    z = u + x
    z.sum().backward()
    print(x.grad)
    x.grad.zero_()
    y.backward()
    print(x.grad)

grad()