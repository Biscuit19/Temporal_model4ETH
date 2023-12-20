import os
import pickle
import random

import torch
from tqdm import tqdm  # 导入tqdm
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		accounts_dict = pickle.load(file)
	return accounts_dict


def dump_pkl(pkl_file, data):
	# 从pkl文件加载数据
	print(f'Dumping {pkl_file}...')
	with open(pkl_file, 'wb') as file:
		pickle.dump(data, file)
	return


def test_graph_generate():
	n = 1
	for i in range(n):
		data = read_pkl(f'part_graph_data_{i}.pkl')
		# 获取所有节点的索引
		all_nodes_indices = list(range(data.num_nodes))
		# 转换为张量
		all_nodes_tensor = torch.tensor(all_nodes_indices)
		# 为全部节点创建子图
		all_subgraph_edge_index, _, all_edge_mask = subgraph(subset=all_nodes_tensor, edge_index=data.edge_index,
															 relabel_nodes=True, return_edge_mask=True)
		print(data.x.size())

		# 创建包含所有节点的子图
		all_sub_graph = Data(x=data.x[all_nodes_tensor], edge_index=all_subgraph_edge_index,
							 edge_attr=data.edge_attr[all_edge_mask], y=data.y[all_nodes_tensor])

		dump_pkl(f'test_data_embed_{i}.pkl', all_sub_graph)
	return


def graph_split(test=False):
	if test:
		test_graph_generate()
		return

	split_ratio = 0.8
	n = 1
	for i in range(n):
		data = read_pkl(f'part_graph_data_{i}.pkl')
		print(f'graph nodes:{data.num_nodes}')

		# address_to_index = read_pkl('address_to_index.pkl')
		# reverse_dict = {v: k for k, v in address_to_index.items()}
		# address_list=[]
		# # 节点索引转地址
		# for node_idx in data.edge_index:
		# 	print(node_idx)
		# 	address = reverse_dict.get(node_idx)
		# 	address_list.append(address)
		# dump_pkl('address_list.pkl',address_list)

		# 预先计算每个节点的邻居
		neighbor_dict = {}
		for start, end in zip(*data.edge_index.tolist()):
			if start not in neighbor_dict:
				neighbor_dict[start] = set()
			neighbor_dict[start].add(end)

		# 分别获取 y=0 和 y=1 的节点索引
		y_zero_indices = [i for i, y in enumerate(data.y) if y == 0]
		y_one_indices = [i for i, y in enumerate(data.y) if y == 1]

		# 为了保证大多数节点有邻居，随机选择一个节点，然后添加其邻居到相应集合
		def select_connected_nodes(node_indices, num_required):
			selected = set()
			while len(selected) < num_required:
				node = random.choice(node_indices)
				selected.add(node)
				if node in neighbor_dict:
					for nei_node in neighbor_dict[node]:
						selected.add(nei_node)
						if len(selected) == num_required:
							return list(selected)

			return list(selected)[:num_required]  # 限制数量

		# 简化后的函数，仅随机选择指定数量的节点
		def select_random_nodes(node_indices, num_required):
			return random.sample(node_indices, num_required)

		# 计算每个类别应有的节点数
		num_train_zero = int(len(y_zero_indices) * split_ratio)
		num_train_one = int(len(y_one_indices) * split_ratio)

		print(f'num_train_zero {num_train_zero}')
		print(f'num_train_one {num_train_one}')

		# 分别选择两个类别的节点
		train_zero_indices = select_random_nodes(y_zero_indices, num_train_zero)
		train_one_indices = select_random_nodes(y_one_indices, num_train_one)

		# 训练集和测试集的节点索引
		train_indices = train_zero_indices + train_one_indices

		test_indices = list(
			(set(y_zero_indices) | set(y_one_indices)) - (set(train_zero_indices) | set(train_one_indices)))

		# 转换为张量
		train_nodes_tensor = torch.tensor(train_indices)
		test_nodes_tensor = torch.tensor(test_indices)

		# 为训练集和测试集创建子图
		train_subgraph_edge_index, _, train_edge_mask = subgraph(subset=train_nodes_tensor, edge_index=data.edge_index,
																 relabel_nodes=True, return_edge_mask=True)
		test_subgraph_edge_index, _, test_edge_mask = subgraph(subset=test_nodes_tensor, edge_index=data.edge_index,
															   relabel_nodes=True, return_edge_mask=True)
		print(data.x.size())

		train_sub_graph = Data(x=data.x[train_nodes_tensor], edge_index=train_subgraph_edge_index,
							   edge_attr=data.edge_attr[train_edge_mask], y=data.y[train_nodes_tensor])

		test_sub_graph = Data(x=data.x[test_nodes_tensor], edge_index=test_subgraph_edge_index,
							  edge_attr=data.edge_attr[test_edge_mask], y=data.y[test_nodes_tensor])
		print(f'Embeding train Size : {train_sub_graph.x.size()}')
		print(f'Embeding test Size : {test_sub_graph.x.size()}')

		dump_pkl(f'train+test_data_embed_{i}.pkl', (train_sub_graph, test_sub_graph))

		static_feature_num = 17
		data.x = data.x[:, :static_feature_num]
		# data.x = torch.cat((data.x, torch.zeros(data.x.shape[0],1)), dim=1)

		train_sub_graph = Data(x=data.x[train_nodes_tensor], edge_index=train_subgraph_edge_index,
							   edge_attr=data.edge_attr[train_edge_mask], y=data.y[train_nodes_tensor])

		test_sub_graph = Data(x=data.x[test_nodes_tensor], edge_index=test_subgraph_edge_index,
							  edge_attr=data.edge_attr[test_edge_mask], y=data.y[test_nodes_tensor])
		print(f'No_embeding Node Size : {train_sub_graph.x.size()}')
		print(f'No_embeding train Size : {train_sub_graph.x.size()}')
		print(f'No_embeding test Size : {test_sub_graph.x.size()}')

		dump_pkl(f'train+test_data_no_embed_{i}.pkl', (train_sub_graph, test_sub_graph))

		# 计算并输出 y=1 和 y=0 的节点数量及其比例
		def calculate_and_print_stats(sub_graph, name):
			y_zero_count = (sub_graph.y == 0).sum().item()
			y_one_count = (sub_graph.y == 1).sum().item()
			ratio = y_zero_count / y_one_count if y_zero_count > 0 else float('inf')
			print(f'{name} - y=0 nodes: {y_zero_count}, y=1 nodes: {y_one_count}, Ratio (y=1/y=0): 1:{ratio:.2f}')

		calculate_and_print_stats(train_sub_graph, "Training graph")
		calculate_and_print_stats(test_sub_graph, "Testing graph")

	# 返回子图
	return


import torch_geometric.utils as pyg_utils


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

	print("Number of nodes where y=0:", y_zero)
	print("Number of nodes where y=1:", y_one)

	# y=0 和 y=1 的节点比例
	ratio = y_zero / y_one if y_one != 0 else float('inf')
	print("Ratio of y=0 to y=1 nodes: {:.2f}".format(ratio))

	# 转换为 NetworkX 图以检查连通性
	nx_graph = pyg_utils.to_networkx(graph, to_undirected=True)
	num_connected_components = nx.number_connected_components(nx_graph)
	print("Number of connected components in the graph:", num_connected_components)


if __name__ == '__main__':
	# graph_split(False)
	graph_split(test=True)

# train_sub_graph, test_sub_graph = read_pkl('train+test_dataC.pkl')
# graph = read_pkl('part_graph_data.pkl')
#
# graph_view(graph)
#
# graph_view(train_sub_graph)
# graph_view(test_sub_graph)
