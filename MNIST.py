import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#数据集下载
from modelscope.msdatasets import MsDataset
from datasets import disable_caching

# 1. 禁用缓存
disable_caching()

try:
    # 2. 使用正确数据集名称
    ds = MsDataset.load('mnist', split='train', force_download=True)
    print("数据集加载成功！样本数:", len(ds))
    
except Exception as e:
    print(f"加载失败: {e}")
    
    # 3. 备用方案：使用 torchvision
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([transforms.ToTensor()])
    ds = MNIST(root='./data', train=True, download=True, transform=transform)
    print(f"备用方案成功！样本数: {len(ds)}")
#您可按需配置 subset_name、split，参照“快速使用”示例代码
n_epochs = 3 #设置训练的轮数为3
batch_size_train = 64 #设置训练时每个批次的样本数量为64。
batch_size_test = 1000 #设置测试时每个批次的样本数量为1000。
learning_rate = 0.01 #设置学习率为0.01
momentum = 0.5 #设置动量为0.5，用于优化算法  
log_interval = 10 #设置日志记录间隔为10，即每处理10个批次记录一次日志
random_seed = 1 #设置随机种子为1，以确保实验的可重复性。
torch.manual_seed(random_seed) #设置PyTorch的随机种子。
 
train_loader = torch.utils.data.DataLoader( #创建一个DataLoader对象，用于批量加载训练数据
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)) #加载MNIST训练数据集
                               ])),
    batch_size=batch_size_train, shuffle=True) #设置每个批次的样本数量为64，shuffle=True表示在每个epoch开始时打乱数据顺序。
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True) #类似于train_loader，但加载的是MNIST测试数据集，并且每个批次的样本数量为1000。
examples = enumerate(test_loader) #将test_loader转换为一个枚举对象，可以逐个获取批次数据
batch_idx, (example_data, example_targets) = next(examples)
#获取第一个批次的数据，包括数据索引batch_idx、图像数据example_data和标签example_targets
# print(example_targets)
# print(example_data.shape)
 
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout() #创建一个图形对象fig。使用循环绘制6张测试图像，每张图像显示在2x3的子图中。
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')#显示图像，灰度模式，不进行插值。
    plt.title("Ground Truth: {}".format(example_targets[i]))#设置子图标题为真实标签。
    plt.xticks([])
    plt.yticks([])#移除坐标轴刻度。
plt.show()#显示图形
 
 
class Net(nn.Module):#定义一个名为Net的神经网络类，继承自nn.Module
    def __init__(self):
        super(Net, self).__init__()#用“__init__”方法初始化网络层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)#第二个卷积层，输入通道数为10，输出通道数为20，卷积核大小为5x5。上一个类似
        self.conv2_drop = nn.Dropout2d()#第二个卷积层后的dropout层，用于防止过拟合。
        self.fc1 = nn.Linear(320, 50)#第一个全连接层，输入特征数为320，输出特征数为50。
        self.fc2 = nn.Linear(50, 10)#如上类似
 
    def forward(self, x):#forward方法定义了前向传播过程
        x = F.relu(F.max_pool2d(self.conv1(x), 2))#对第一个卷积层的输出进行ReLU激活和最大池化操作。
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))#对第二个卷积层的输出进行dropout、ReLU激活和最大池化操作。
        x = x.view(-1, 320)#将特征图展平为一维向量。
        x = F.relu(self.fc1(x))#对第一个全连接层的输出进行ReLU激活。
        x = F.dropout(x, training=self.training)#对全连接层的输出进行dropout操作。
        x = self.fc2(x)#第二个全连接层的输出
        return F.log_softmax(x, dim=1)#对输出进行log_softmax操作，返回每个类别的对数概率。
 
network = Net()#创建一个Net类的实例，即神经网络模型。
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)#创建一个优化器对象，使用随机梯度下降（SGD）算法优化网络参数。lr设置学习率，momentum设置动量。
 
train_losses = []#存储训练过程中的损失值。
train_counter = []#存储训练过程中的样本计数。
test_losses = []#存储测试过程中的损失值
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
 
def train(epoch):  # 定义一个训练函数train，参数为当前epoch数
    network.train()  # 将网络设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练数据加载器中的每个批次
        optimizer.zero_grad()  # 清零优化器的梯度，避免梯度累积
        output = network(data)  # 前向传播，计算网络输出
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新网络参数
        if batch_idx % log_interval == 0:  # 每log_interval个批次打印一次训练信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),   
                                                                           len(train_loader.dataset),     
                                                                           100. * batch_idx / len(train_loader),    
                                                                           loss.item()))     
            train_losses.append(loss.item())  # 将当前批次的损失值添加到训练损失列表中
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))  
            torch.save(network.state_dict(), './model.pth')  # 保存网络参数
            torch.save(optimizer.state_dict(), './optimizer.pth')  # 保存优化器状态

def test():  # 定义测试函数
    network.eval()  # 将网络设置为评估模式
    test_loss = 0  # 初始化测试损失为0
    correct = 0  # 初始化正确预测的数量为0
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for data, target in test_loader: 
            output = network(data)  
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.data.max(1, keepdim=True)[1]  
            correct += pred.eq(target.data.view_as(pred)).sum() 
    test_loss /= len(test_loader.dataset) 
    test_losses.append(test_loss)  
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),  
        100. * correct / len(test_loader.dataset)))
train(1)
test()  # 调用测试函数，进行一次测试

for epoch in range(1, n_epochs + 1):  
    train(epoch)  
    test() 
fig = plt.figure() 
plt.plot(train_counter, train_losses, color='blue') 
plt.scatter(test_counter, test_losses, color='red') 
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  
plt.xlabel('number of training examples seen') 
plt.ylabel('negative log likelihood loss')  

examples = enumerate(test_loader) 
batch_idx, (example_data, example_targets) = next(examples) 
with torch.no_grad():  
    output = network(example_data)  
fig = plt.figure()  
for i in range(6):  # 绘制前6个样本的预测结果
    plt.subplot(2, 3, i + 1)  # 创建子图
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')  # 显示图像
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))  # 设置标题为预测类别
    plt.xticks([])  # 移除x轴刻度
    plt.yticks([])  # 移除y轴刻度
plt.show()  # 显示图形

# ----------------------------------------------------------- #

continued_network = Net()  # 初始化一个新的网络实例
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)  # 初始化优化器

network_state_dict = torch.load('model.pth')  # 加载保存的网络参数
continued_network.load_state_dict(network_state_dict)  # 将加载的参数加载到新的网络实例中
optimizer_state_dict = torch.load('optimizer.pth')  # 加载保存的优化器状态
continued_optimizer.load_state_dict(optimizer_state_dict)  # 将加载的状态加载到新的优化器中

#   注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，不然报错：
#  x and y must be the same size
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)

for i in range(4, 9):  # 循环从4到8（不包括9）
    test_counter.append(i*len(train_loader.dataset))  # 将当前epoch数乘以训练集大小，添加到测试计数器中
    train(i)  # 调用train函数进行训练，传入当前epoch数
    test()  # 调用test函数进行测试

fig = plt.figure()  # 创建一个新的图形对象
plt.plot(train_counter, train_losses, color='blue')  # 绘制训练损失曲线，颜色为蓝色
plt.scatter(test_counter, test_losses, color='red')  # 绘制测试损失散点图，颜色为红色
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # 添加图例，显示训练损失和测试损失
plt.xlabel('number of training examples seen')  # 设置x轴标签为“看到的训练样本数量”
plt.ylabel('negative log likelihood loss')  # 设置y轴标签为“负对数似然损失”
plt.show()  # 显示图形
