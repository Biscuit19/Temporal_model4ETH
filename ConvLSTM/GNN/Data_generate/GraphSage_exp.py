import pickle

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		data = pickle.load(file)
	return data


def dump_pkl(pkl_file, data):
	# 从pkl文件加载数据
	print(f'Dumping {pkl_file}...')
	with open(pkl_file, 'wb') as file:
		pickle.dump(data, file)
	return

def graph_view(graph):
	num_nodes = graph.num_nodes
	print("Number of nodes:", num_nodes)

	num_edges = graph.num_edges
	print("Number of edges:", num_edges)

	# 图的入度
	degree=num_edges / (num_nodes * 2)
	print("Degree of the graph: {:.6f}".format(degree))

	# 获取 y=1 和 y=0 的节点数量
	y_zero = (graph.y == 0).sum().item()
	y_one = (graph.y == 1).sum().item()

	print("Number of nodes where y=0:", y_zero)
	print("Number of nodes where y=1:", y_one)

	# y=0 和 y=1 的节点比例
	ratio = y_zero / y_one if y_one != 0 else float('inf')
	print("Ratio of y=0 to y=1 nodes: {:.2f}".format(ratio))


class GraphSAGE(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels):
		super(GraphSAGE, self).__init__()
		self.conv1 = SAGEConv(in_channels, hidden_channels)
		self.conv2 = SAGEConv(hidden_channels, 1)  # 输出层为1，用于二分类

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		return torch.sigmoid(x).view(-1)  # 应用sigmoid激活函数并调整输出形状

# class GraphSAGE(torch.nn.Module):
# 	def __init__(self, in_channels, hidden_channels):
# 		super(GraphSAGE, self).__init__()
# 		self.conv1 = SAGEConv(in_channels, hidden_channels)
# 		self.conv2 = SAGEConv(hidden_channels, hidden_channels)
# 		self.fc = torch.nn.Linear(hidden_channels, 1)  # 将高维特征映射为1维
#
# 	def forward(self, data):
# 		x, edge_index = data.x, data.edge_index
# 		x = self.conv1(x, edge_index)
# 		x = F.relu(x)
# 		x = F.dropout(x, training=self.training)
# 		x = self.conv2(x, edge_index)
# 		x = F.relu(x)
# 		x = self.fc(x)  # 应用全连接层
# 		return torch.sigmoid(x).view(-1)


def train_graphsage_model(train_data, test_data, save_model=False):
	num_epochs = 5000
	lr = 0.1
	hidden_dim = 8

	print('--------------------Train Dataset-------------------------')
	graph_view(train_data)
	print('--------------------Test Dataset-------------------------')
	graph_view(test_data)

	model = GraphSAGE(in_channels=train_data.num_node_features, hidden_channels=hidden_dim)
	print(f'node_features size: {train_data.num_node_features}')
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	train_data = train_data.to(device)
	test_data = test_data.to(device)

	print(f'Start Training...')
	for epoch in range(num_epochs):
		# 训练逻辑
		model.train()
		optimizer.zero_grad()
		output = model(train_data)  # 直接得到概率
		loss = criterion(output, train_data.y.float())
		loss.backward()
		optimizer.step()

		# 每10个epoch进行一次评估
		if (epoch + 1) % 50 == 0:
			model.eval()
			with torch.no_grad():
				output = model(test_data)
				predictions = (output > 0.5).float()  # 二分类阈值
				true_labels = test_data.y

				# 计算评估指标
				precision = precision_score(true_labels.cpu(), predictions.cpu(), zero_division=0)
				recall = recall_score(true_labels.cpu(), predictions.cpu())
				f1 = f1_score(true_labels.cpu(), predictions.cpu())

				# 计算FPR
				tn, fp, fn, tp = confusion_matrix(true_labels.cpu(), predictions.cpu()).ravel()
				fpr = fp / (fp + tn)

				accuracy = accuracy_score(true_labels.cpu(), predictions.cpu())
				print(
					f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}')

	if save_model:
		model_path = './GraphSAGE_model.pth'
		torch.save(model.state_dict(), model_path)

	return model

if __name__ == "__main__":
	train_data, test_data = read_pkl('train+test_data_embed_0.pkl')
	train_graphsage_model(train_data, test_data, save_model=True)

	train_data, test_data = read_pkl('train+test_data_no_embed_0.pkl')
	train_graphsage_model(train_data, test_data, save_model=False)


