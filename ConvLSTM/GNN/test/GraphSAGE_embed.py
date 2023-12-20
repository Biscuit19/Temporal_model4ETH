import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph

# 创建一个图（示例使用Cora数据集，你可以使用自己的图数据）
data = citation_graph.load_cora()
graph = DGLGraph(data.graph)
features = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)

# 定义GraphSage模型
class GraphSage(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSage, self).__init__()
        self.sageconv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.sageconv2 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, features):
        x = F.relu(self.sageconv1(g, features))
        x = self.sageconv2(g, x)
        return x

# 定义SAGEConv层
class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, aggr):
        super(SAGEConv, self).__init__()
        self.aggr = aggr
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, features):
        g = g.local_var()
        g.ndata['h'] = features
        g.update_all(fn.copy_src(src='h', out='m'), fn.mean(msg='m', out='h'))
        h = g.ndata['h']
        return self.linear(h)

# 创建模型和优化器
model = GraphSage(in_feats=features.shape[1], h_feats=16, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    logits = model(graph, features)
    loss = F.cross_entropy(logits[data.train_mask], labels[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 获取节点表示
with torch.no_grad():
    model.eval()
    node_representations = model(graph, features)
    print("Node Representations:", node_representations)

# 在测试集上进行推断
model.eval()
logits = model(graph, features)
predicted_labels = logits.argmax(1)
accuracy = (predicted_labels[data.test_mask] == labels[data.test_mask]).float().mean()
print(f'Test Accuracy: {accuracy.item()}')
