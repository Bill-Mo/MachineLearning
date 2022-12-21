import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as Fun
from sklearn.metrics import confusion_matrix
#  set hyper-parameters
lr = 0.02  # learning rate
epochs = 300  # training epochs

n_hidden = 20  # hidden layers
n_output = 9  #

# 1.prepare data

Date_clear = pd.read_csv('data_true.csv')

# n_feature = len(list(Date_clear.columns)) - 1  # features
n_feature = 6

print(n_feature)


x, y = Date_clear.iloc[:, 1:].values, Date_clear.iloc[:, 0].values

# divide the data into 70% and 30%

x_train0, x_test0, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


from sklearn.decomposition import PCA
estimator = PCA(n_components=n_feature)   # 初始化，23维压缩至20维
# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征
x_train0 = estimator.fit_transform(x_train0)
print(x_train0.shape)
   # 维度从23变为20
# 测试特征也按照上述的20个正交维度方向进行转化（transform）
x_test0 = estimator.transform(x_test0)

# 归一化(也就是所说的min-max标准化)通过调用sklearn库的标准化函数
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train0)
x_test = min_max_scaler.fit_transform(x_test0)




# transform data type into Tensor
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)


# 2.define BPNN
class BPNetModel(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPNetModel, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden(x))  # 隐层激活函数采用relu()函数
        out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        return out


# 3.define optimizer and loss function
net = BPNetModel(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)  # 调用网络
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用Adam优化器，并设置学习率
loss_fun = torch.nn.CrossEntropyLoss()  # 对于多分类一般使用交叉熵损失函数

# 4.training model
loss_steps = np.zeros(epochs)  # 构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
accuracy_steps = np.zeros(epochs)

for epoch in range(epochs):
    y_pred = net(x_train)  # 前向传播
    loss = loss_fun(y_pred, y_train)  # 预测值和真实值对比
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新梯度
    loss_steps[epoch] = loss.item()  # 保存loss
    running_loss = loss.item()
    acc = (torch.argmax(y_pred, dim=1) == y_train).type(torch.FloatTensor).mean()
    print(f"The {epoch} training epochs, loss={running_loss}, training accuracy={acc}".format(epoch, running_loss, acc))
    with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        y_pred = net(x_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_steps[epoch] = correct.mean()
        print("accuracy on test set", accuracy_steps[epoch])
        print('\n')




# print("测试鸢尾花的预测准确率",accuracy_steps[-1])

# 5.draw loss function and loss
fig_name = "dataset_classify_Net"
fontsize = 15
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 12), sharex=True)
ax1.plot(accuracy_steps)
ax1.set_ylabel("test accuracy", fontsize=fontsize)
ax1.set_title(fig_name, fontsize="xx-large")
ax2.plot(loss_steps)
ax2.set_ylabel("train loss", fontsize=fontsize)
ax2.set_xlabel("epochs", fontsize=fontsize)
plt.tight_layout()
plt.savefig(fig_name + '.png')
plt.show()

