import pickle

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		accounts_dict = pickle.load(file)
	return accounts_dict

# 定义一个简单的多层感知器（MLP）模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)          # 隐藏层
        self.fc3 = nn.Linear(64, 1)            # 隐藏层到输出层


    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 输出层使用Sigmoid激活函数
        return x

# 定义模型训练函数
def train_model(train_data):
    # 检测是否有可用的GPU，如果有，则使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    features_tensor, labels_tensor = train_data
    # 将数据移动到设备上
    features_tensor = features_tensor.to(device)
    labels_tensor = labels_tensor.to(device)

    input_size = features_tensor.shape[1]

    # 设置一些超参数
    epochs = 300  # 训练轮数
    learning_rate = 0.001  # 学习率
    batch_size = 128  # 批处理大小

    print('Data size:', features_tensor.size())

    # 创建数据集
    dataset = TensorDataset(features_tensor, labels_tensor)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 创建模型、损失函数和优化器
    model = SimpleMLP(input_size).to(device)  # 将模型移动到设备上
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和测试模型
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备上

            optimizer.zero_grad()  # 清除之前的梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

        if (epoch + 1) % 10 == 0:
            # 测试模型
            model.eval()
            all_labels = []
            all_outputs = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备上
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float()
                    all_labels.extend(labels.cpu().view(-1).tolist())  # 将标签移回CPU
                    all_outputs.extend(predicted.cpu().view(-1).tolist())  # 将预测结果移回CPU

            # 计算评估指标
            precision = precision_score(all_labels, all_outputs, zero_division=0)
            recall = recall_score(all_labels, all_outputs)
            f1 = f1_score(all_labels, all_outputs)
            accuracy = accuracy_score(all_labels, all_outputs)
            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')


if __name__ == '__main__':
    # 训练模型
    train_data = read_pkl('mlp_data_all_feature.pkl')
    train_model(train_data)

    train_data = read_pkl('mlp_data_embed.pkl')
    train_model(train_data)

    train_data = read_pkl('mlp_data_static.pkl')
    train_model(train_data)


