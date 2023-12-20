import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 创建一个示例图
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]], dtype=torch.long)
x = torch.randn(4, 16)  # 4个节点，每个节点的特征维度为16

data = Data(x=x, edge_index=edge_index)

# 定义Graph Autoencoders模型
class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(in_channels, hidden_channels)
        self.decoder = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = F.relu(x)
        x = self.decoder(x, edge_index)
        return x

# 初始化模型
model = GraphAutoencoder(in_channels=16, hidden_channels=8)

# 定义训练函数
def train(data, model, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.mse_loss(output, data.x)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 定义优化器和训练参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

# 开始训练
train(data, model, optimizer, num_epochs)

# 获取节点表示
model.eval()
with torch.no_grad():
    node_representations = model.encoder(data.x, data.edge_index)
