import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


node_features=[1,2,3,4,5]
torch1=torch.tensor(node_features, dtype=torch.float32)

print(torch1.size())