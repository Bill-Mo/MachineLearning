import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from preprocessing import preprocessing
#  set hyper-parameters
lr = 0.001  # learning rate
episodes= 300  # training episodes
epoch = 50

n_feature = 40
n_hidden = 30  # hidden layers
n_output = 6  #

x_train, x_test, y_train, y_test = preprocessing(n_feature)


# transform data type into Tensor
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

# define BPNN
class BPNetModel(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPNetModel, self).__init__()
        self.hid1 = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hid1(x))
        out = F.softmax(self.out(x), dim=1) 
        return out


# define optimizer and loss function
net = BPNetModel(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output) 
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fun = torch.nn.CrossEntropyLoss()

# training model
loss_steps = np.zeros(episodes) 
accuracy_steps = np.zeros(episodes)

for i in range(episodes):
    y_pred = net(x_train)
    loss = loss_fun(y_pred, y_train)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    loss_steps[i] = loss.item()
    running_loss = loss.item()
    acc = (torch.argmax(y_pred, dim=1) == y_train).type(torch.FloatTensor).mean()
    if i % epoch == 0: 
        print(f"The {i} training episodes, loss={running_loss}, training accuracy={acc}".format(i, running_loss, acc))
    with torch.no_grad():
        y_pred = net(x_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_steps[i] = correct.mean()
    if i % epoch == 0:
        print("accuracy on test set", accuracy_steps[i])

# draw loss function and loss
fig_name = "dataset_classify_Net"
fontsize = 10
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 5), sharex=True)
ax1.plot(accuracy_steps)
ax1.set_ylabel("test accuracy", fontsize=fontsize)
ax1.set_title(fig_name, fontsize="xx-large")
ax2.plot(loss_steps)
ax2.set_ylabel("train loss", fontsize=fontsize)
ax2.set_xlabel("episodes", fontsize=fontsize)
plt.tight_layout()
plt.savefig(fig_name + '.png')
plt.show()
