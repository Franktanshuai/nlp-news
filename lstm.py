import torch
import torch.nn as nn
import pickle

# 读取数据
with open('dataset\\training_set.pkl', 'rb') as f1:
    train_set = pickle.load(f1)
    train_x = torch.from_numpy(train_set['X'])
    train_y = torch.from_numpy(train_set['y'])
    train_y = train_y + 1
with open('dataset\\test_set.pkl', 'rb') as f2:
    test_set = pickle.load(f2)
    test_x = torch.from_numpy(test_set['X'])
    test_y = torch.from_numpy(test_set['y'])
    test_y = test_y + 1

# 定义模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # 前向传播LSTM
        x = x.to(torch.float32)
        www = x.unsqueeze(1)
        out, _ = self.lstm(www, (h0, c0))

        # 解码最后一个时间步长的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型并定义超参数
input_size = train_x.shape[1]
hidden_size = 128
num_layers = 2
num_classes = 3
model = LSTM(input_size, hidden_size, num_layers, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_x)
    loss = criterion(outputs, train_y.to(torch.long))

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(test_x)
    _, predicted = torch.max(outputs.data, 1)
    total += test_y.size(0)
    correct += (predicted == test_y).sum().item()

print('Test Accuracy: {} %'.format(100 * correct / total))
