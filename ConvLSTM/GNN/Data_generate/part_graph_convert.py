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

from train_test_split import graph_split


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


def normalize_tensor(tensor):
	mean = tensor.mean(dim=0)  # 计算均值
	std = tensor.std(dim=0)  # 计算标准差
	normalized_tensor = (tensor - mean) / (std + 1e-8)  # 归一化，并避免除以零
	return normalized_tensor


from AutoEncoder_ConvLSTM import AutoEncoder_ConvLSTM


def load_model():
	AECL_model = AutoEncoder_ConvLSTM(input_channels=4, hidden_channels=64, kernel_size=(301, 1), num_layers=1,
									  batch_first=True,
									  bias=True, return_all_layers=False)
	AECL_model.load_state_dict(torch.load('AECL_model.pth'))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	AECL_model.to(device)
	print(f'Load AECL_model.pth')
	return AECL_model


from Feature_for_node import feature_generate


def create_part_graph(user_dict, address_to_index, data):
	# 1.筛选中心节点
	print('[Step 1 Filter core nodes].....')
	print(f'original_dict numbers: {len(user_dict.keys())}')
	user_category_1 = []  # 存储"category" = 1的用户
	user_category_0 = []  # 存储"category" = 0的用户
	for user, details in user_dict.items():
		if 1000 >= details['all_cnt'] >= 3:
			if details['category'] == 1:
				user_category_1.append(user)
			elif details['category'] == 0:
				user_category_0.append(user)

	print(f'Total phisher numbers: {len(user_category_1)}')  # 3227
	print(f'Total normal numbers: {len(user_category_0)}')  # 593878

	# 恶意用户的数量
	x = 3000
	# 进行随机抽样
	user_category_1_sampled = random.sample(user_category_1, min(x, len(user_category_1)))
	user_category_0_sampled = random.sample(user_category_0, min(20 * len(user_category_1_sampled), len(user_category_0)))

	print(f'sampled phisher numbers: {len(user_category_1_sampled)}')
	print(f'sampled normal numbers: {len(user_category_0_sampled)}')

	# 筛选得到的地址列表
	filtered_list = user_category_0_sampled + user_category_1_sampled

	# 将选择的用户转换为节点索引
	selected_nodes_indices = [address_to_index[user] for user in filtered_list]

	# 反向字典 无敌
	reverse_dict = {v: k for k, v in address_to_index.items()}

	# # 2.搜索邻居节点
	# # 预计算邻居节点字典
	# print('[Step 2 Find neighbour nodes].....')
	# neighbor_dict = {}
	# for start, end in zip(*data.edge_index.tolist()):
	# 	if start not in neighbor_dict:
	# 		neighbor_dict[start] = set()
	# 	neighbor_dict[start].add(end)
	#
	# # 找到他们的邻居节点索引
	# neighbour_nodes_indices = set()
	# for node_idx in tqdm(selected_nodes_indices, desc='Neighbour finding'):
	# 	neighbours = neighbor_dict.get(node_idx, set())
	# 	neighbour_nodes_indices.update(neighbours)
	#
	# print(f'neighbours num {len(neighbour_nodes_indices)}')
	#
	#
	# # 邻居节点索引转地址
	# for node_idx in neighbour_nodes_indices:
	# 	address = reverse_dict.get(node_idx)
	# 	filtered_list.append(address)

	# 3.合并,得到子图节点索引
	print('[Step 3 Merge nodes].....')
	filter_nodes = [address_to_index[user] for user in filtered_list]
	dump_pkl('filtered_list.pkl',filtered_list)
	dump_pkl('filter_nodes_indices.pkl',filter_nodes)


	print(f'nodes num {len(filter_nodes)}')

	# 4.更新节点的特征向量
	print('[Step 4 Update node Features].....')
	# 更新图的x特征形状
	# 假设 data.x 的原始维度是 [x, m]
	size_1, size_2 = data.x.shape
	# 设定目标维度 num
	convlstm_hidden_num=64
	print(f'hidden_temporal feature size: {convlstm_hidden_num}')
	feature_num = 17+convlstm_hidden_num

	# 根据条件改变维度
	data.x = data.x[:, :feature_num] if feature_num <= size_2 else torch.cat(
		(data.x, torch.zeros(size_1, feature_num - size_2)), dim=1)

	# 更新节点特征

	def update_node_features():
		AECL_model = load_model()
		for node_index in tqdm(filter_nodes, desc='Updating Features'):
			if node_index < data.x.shape[0]:
				add = reverse_dict[node_index]
				feature = user_dict[add]
				node_features = feature_generate(feature, AECL_model)
				data.x[node_index] = node_features
			else:
				print(f"节点索引 {node_index} 超出范围，无法更新。")

	update_node_features()

	# 5.创建子图
	print('[Step 5 Create subgraph].....')
	# 使用这些节点索引创建子图，并重新标记节点
	subgraph_edge_index, _, edge_mask = subgraph(subset=filter_nodes, edge_index=data.edge_index,
												return_edge_mask=True)

	# 新的这些并没有标准化，需要标准化一下
	sub_x = normalize_tensor(data.x[filter_nodes])

	# 创建新的图数据对象
	sub_graph = Data(x=sub_x, edge_index=subgraph_edge_index, edge_attr=data.edge_attr[edge_mask],
					 y=data.y[filter_nodes])

	print(f'create node: {(sub_graph.num_nodes)}; create edge {(sub_graph.num_edges)}')

	return sub_graph

import zipfile
def graph_convert(test=False):
	user_dict = read_pkl('all_account_data_zsu.pkl')
	# user_dict = read_pkl('all_account_data.pkl')
	address_to_index = read_pkl('address_to_index.pkl')
	data = read_pkl('whole_graph_data.pkl')
	n = 1
	for i in range(n):
		graph_data = create_part_graph(user_dict, address_to_index, data)
		# 存储为Pickle文件
		dump_pkl(f'part_graph_data_{i}.pkl', graph_data)
		with zipfile.ZipFile(f'part_graph_data_{i}.zip', 'w') as zipf:
			zipf.write(f'part_graph_data_{i}.pkl')
		graph = read_pkl(f'part_graph_data_{i}.pkl')
		graph_view(graph)

		graph_split(test)

		try:
			upload_command = f"oss cp part_graph_data_{i}.zip oss://train/"
			os.system(upload_command)
		except:
			continue


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
	graph_convert(test=True)
	# graph_convert(test=False)



# train_sub_graph, test_sub_graph = read_pkl('train+test_dataC.pkl')

#
# graph_view(train_sub_graph)
# graph_view(test_sub_graph)
