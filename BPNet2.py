# BP ，更新阈值和权重，回归预测问题最后一层不带激活函数
# coding: UTF-8

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tkinter import _flatten

# 读取数据预处理
def load_data_wrapper(file_name):
    
    line_data = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line_data.append(line.strip().split(','))

    return np.array(line_data, dtype='float32')

# 提出特征和标签，特征做输入，标签为输出
def split_data(dataset):
    
    character = dataset[:, 0].T
    label = dataset[:, 2].T
    
    return character, label


# 初始化参数
# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
def parameter_initialization(x, y, z):
    # 隐层阈值从（-5,5）之间的随机数
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float32)

    # 输出层阈值
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float32)

    # 输入层与隐层的连接权重
    weight1 = np.random.randint(-5, 5, (1, y)).astype(np.float32)

    # 隐层与输出层的连接权重
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float32)

    return weight1, weight2, value1, value2

#定义激活函数
def sigmoid(inx):
    if inx.any()>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))
def relu(z):
    return np.where(z < 0, 0, z)

#训练过程
def train_process(dataset, labelset, weight1, weight2, value1, value2):
    # 步长
    x = 0.05
    for i in range(len(dataset)):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.double)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.double)
        # 隐层输入
        input1 = np.dot(inputset, weight1).astype(np.double)
        # 隐层输出
        output2 = sigmoid(input1 - value1).astype(np.double)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.double)
        # 输出层输出
        output3 = input2 - value2

        # 更新公式由矩阵运算表示 用的是平方误差
        g = outputset - output3 #最后一层直接求导 ，为输出层阈值求导
        b = np.dot(g, np.transpose(weight2))
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)  # 隐藏层之间阈值

        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        # 更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change

    return weight1, weight2, value1, value2

def test_process(dataset, weight1, weight2, value1, value2):
    pre_data = []
    for i in range(len(dataset)):
        # 计算通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.double)
        # outputset = np.mat(labelset[i]).astype(np.double)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = np.dot(output2, weight2) - value2
        output3 = output3.tolist()
        pre_data.append(output3)
        pre_data = list(_flatten(pre_data))
        # 返回预测值
    return pre_data

if __name__ == '__main__':
    #要打开的文件名
    iris_file = 'D:\\Users\\LanluZ\\Desktop\\BP\\Dataset\\T-D.txt'
    #预测模型（使用训练结果）
    x_test=np.array([14.5,17])
    #初始化权重
    weight1 = np.array([[1.32554030418396, -0.29923900961875916, -15.884475708007812, -2.2334399223327637]])
    weight2 = np.array([[3.1228485107421875], [-2.5544259548187256], [-5.974942207336426], [-0.8869158625602722]])
    value1 = np.array([[6.539976596832275, 2.7331912517547607, -26.606788635253906, 3.120114803314209]])
    value2 = np.array([[-16.19719886779785]])
    #预测
    pre = test_process(x_test,weight1,weight2,value1,value2)
    print(pre)


    

    

'''
    while(1):
        #数据预处理
        Data = load_data_wrapper(iris_file)
        #分离特征标签值，x为数据集的feature数据，y为label.
        x, y = split_data(Data)
        #分训练和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        #初始化权重
        weight1, weight2, value1, value2 = parameter_initialization(len(x_train), 4, 1)
        #训练
        for i in range(1000):
            weight1, weight2, value1, value2 = train_process(x_train, y_train, weight1, weight2, value1, value2)
        #预测
        pre = test_process(x_test, weight1, weight2, value1, value2)

        errors_std = np.std(np.array(pre) - np.array(y_test))
        
        print("errors_std:\n", errors_std)
        print(weight1.tolist())
        print(weight2.tolist())
        print(value1.tolist())
        print(value2.tolist())
        print()
'''


