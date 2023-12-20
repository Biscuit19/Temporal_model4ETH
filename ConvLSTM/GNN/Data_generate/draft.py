import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

# 假设有一个简单的图数据
# 节点特征矩阵
x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
# 边索引
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2],
                           [1, 2, 3, 4, 0, 3, 4, 0]], dtype=torch.long)
edge_attr = torch.tensor([1, 2, 3, 4, 4, 4, 4, 4], dtype=torch.float)
# 目标/标签（如果有的话）
y = torch.tensor([0, 1], dtype=torch.float)

# 创建一个图数据对象
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 定义一个子集来选择节点
filter_nodes = torch.tensor([0, 2, 4], dtype=torch.long)

# 创建子图
subgraph_edge_index, _, edge_mask = subgraph(subset=filter_nodes, edge_index=data.edge_index,relabel_nodes=True, return_edge_mask=True)

# 子图的节点特征
sub_x = x[filter_nodes]

# 创建子图数据对象
sub_graph = Data(x=sub_x, edge_index=subgraph_edge_index, edge_attr=data.edge_attr[edge_mask])

# 打印子图信息
print("Subgraph Node Features:\n", sub_graph.x)
print("Subgraph Edge Index:\n", sub_graph.edge_index)

print("Subgraph Edge num:\n", sub_graph.num_edges)

print("Original Node Indices in Subgraph:\n", filter_nodes)

