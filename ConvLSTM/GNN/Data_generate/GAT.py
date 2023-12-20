import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import subgraph

from torch.optim import Adam
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

print(torch.__version__)
# 定义张量输出结果
torch.set_printoptions(precision=4, sci_mode=False)


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


# 定义 GAT 模型
class GATWithClassifier(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, num_heads):
		super(GATWithClassifier, self).__init__()
		self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
		self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
		self.classifier = nn.Sequential(
			# 全连接层，将输入的维度从hidden_dim * num_heads 转换为1，这是二分类的输出
			nn.Linear(hidden_dim * num_heads, 1),
			# Sigmoid激活函数，它将输出压缩到范围 [0, 1]
			nn.Sigmoid()
		)

	def forward(self, data):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		x = self.conv1(x, edge_index, edge_attr)
		x = self.conv2(x, edge_index, edge_attr)
		x = self.classifier(x)
		return x

	def get_node_embedding(self, data, node_idx):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		x = self.conv1(x, edge_index, edge_attr)
		x = self.conv2(x, edge_index, edge_attr)

		# 提取特定节点的嵌入表示
		node_embedding = x[node_idx]

		return node_embedding


def graph_view(graph):
	num_nodes = graph.num_nodes
	print("Number of nodes:", num_nodes)

	num_edges = graph.num_edges
	print("Number of edges:", num_edges)

	# 图的密度
	density = num_edges / (num_nodes * (num_nodes - 1))
	print("Density of the graph: {:.6f}".format(density))

	# 获取 y=1 和 y=0 的节点数量
	y_zero = (graph.y == 0).sum().item()
	y_one = (graph.y == 1).sum().item()

	print("Number of normal:", y_zero)
	print("Number of phisher", y_one)


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def train_gat_model(train_data, test_data,save_model=False):
	num_epochs = 150
	lr = 0.01

	print('--------------------Train Dataset-------------------------')
	graph_view(train_data)
	print('--------------------Test Dataset-------------------------')
	graph_view(test_data)

	# Initialization
	model = GATWithClassifier(in_dim=train_data.num_node_features, hidden_dim=4, num_heads=2)
	print(f'node_features size: {train_data.num_node_features}')
	criterion = nn.BCELoss()
	optimizer = Adam(model.parameters(), lr=lr)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	train_data = train_data.to(device)
	test_data = test_data.to(device)

	print(f'Start Training...')
	for epoch in range(num_epochs):
		model.train()
		optimizer.zero_grad()
		output = model(train_data).view(-1)
		loss = criterion(output, train_data.y.float())
		loss.backward()
		optimizer.step()

		if (epoch + 1) % 1 == 0:
			model.eval()
			with torch.no_grad():
				output = model(test_data).view(-1)
				predictions = (output > 0.5).float()
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
		model_path = './GAT_model.pth'  # 指定保存的文件路径
		torch.save(model.state_dict(), model_path)
	return model


def test_gat_model(test_data, model_path):
	print(f'Testing Model')
	graph_view(test_data)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 创建模型实例并加载权重
	model = GATWithClassifier(in_dim=test_data.num_node_features, hidden_dim=4, num_heads=2)
	print(f'node_features size: {test_data.num_node_features}')
	model.load_state_dict(torch.load(model_path, map_location=device))
	model = model.to(device)

	# 确保模型处于评估模式
	model.eval()

	# 将测试数据移至相同的设备
	test_data = test_data.to(device)

	# 不需要计算梯度
	with torch.no_grad():
		# 计算模型在测试数据上的输出
		output = model(test_data).view(-1)
		predictions = (output > 0.5).float()  # 将输出转换为二进制预测
		true_labels = test_data.y

		# 计算性能指标
		precision = precision_score(true_labels.cpu(), predictions.cpu(), zero_division=0)
		recall = recall_score(true_labels.cpu(), predictions.cpu())
		f1 = f1_score(true_labels.cpu(), predictions.cpu())

		# 计算FPR
		tn, fp, fn, tp = confusion_matrix(true_labels.cpu(), predictions.cpu()).ravel()
		fpr = fp / (fp + tn)

		accuracy = accuracy_score(true_labels.cpu(), predictions.cpu())

		# 打印性能指标
		print(
			f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}')


def validate_edge_index(edge_index):
	if edge_index.dim() != 2 or edge_index.size(0) != 2:
		raise ValueError("edge_index must be a 2xN tensor where N is the number of edges.")
	if edge_index.is_sparse:
		raise ValueError("edge_index should not be a sparse tensor.")
	print("edge_index is valid.")


def validate_edge_attr(edge_index, edge_attr):
	if edge_attr is None:
		print("No edge_attr provided. Skipping validation for edge_attr.")
		return
	if edge_attr.dim() != 2:
		raise ValueError("edge_attr must be a 2D tensor.")
	if edge_attr.size(0) != edge_index.size(1):
		raise ValueError("The length of edge_attr does not match the number of edges in edge_index.")
	print("edge_attr is valid.")


if __name__ == "__main__":
	# # 训练模型
	# train_data, test_data = read_pkl('train+test_data_embed_0.pkl')
	# train_gat_model(train_data, test_data,save_model=True)
	# # 不带嵌入：
	# train_data, test_data = read_pkl('train+test_data_no_embed_0.pkl')
	# train_gat_model(train_data, test_data,save_model=False)

	# 测试模型
	test_data = read_pkl('test_data_embed_0.pkl')
	# train_data, test_data = read_pkl('train+test_data_embed_0.pkl')
	gat_model_path = './GAT_model.pth'
	test_gat_model(test_data, gat_model_path)


# validate_edge_index(train_data.edge_index)
# validate_edge_attr(train_data.edge_index, train_data.edge_attr)
# validate_edge_index(valid_data.edge_index)
# validate_edge_attr(valid_data.edge_index, valid_data.edge_attr)
