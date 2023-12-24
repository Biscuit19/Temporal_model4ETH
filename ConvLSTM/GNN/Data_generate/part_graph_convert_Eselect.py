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
from Feature_for_node import feature_generate


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

# 带标签过滤的随机游走
def extract_subgraph_0(graph, num_nodes):
	print(f'random walk to {num_nodes}')

	# Ensure the number of nodes requested does not exceed the total number of nodes in the graph
	num_nodes = min(num_nodes, graph.num_nodes)

	# Find all nodes with y label equal to 0
	zero_label_nodes = [i for i, label in enumerate(graph.y) if label == 0]

	visited_nodes = set()

	with tqdm(total=num_nodes, desc="Selecting nodes") as pbar:
		while len(visited_nodes) < num_nodes:
			# Choose a starting node from the remaining zero_label_nodes
			potential_start_nodes = list(set(zero_label_nodes) - visited_nodes)
			if not potential_start_nodes:
				break  # No more zero_label_nodes to choose from

			current_node = random.choice(potential_start_nodes)
			visited_nodes.add(current_node)
			pbar.update(1)

			# Perform the random walk
			while len(visited_nodes) < num_nodes:
				neighbors = [n for n in graph.edge_index[1][graph.edge_index[0] == current_node].tolist() if
							 graph.y[n] == 0]
				neighbors = list(set(neighbors) - visited_nodes)

				if not neighbors:
					break  # No more neighbors to explore from the current node

				current_node = random.choice(neighbors)
				visited_nodes.add(current_node)
				pbar.update(1)

	subgraph_nodes = list(visited_nodes)

	sub_edge_index, sub_edge_attr = subgraph(subgraph_nodes, graph.edge_index, edge_attr=graph.edge_attr,
											 relabel_nodes=True)
	subgraph_data = Data(x=graph.x[subgraph_nodes], edge_index=sub_edge_index, edge_attr=sub_edge_attr,
						 y=graph.y[subgraph_nodes])
	graph_view(subgraph_data)


	return subgraph_nodes

# 带特征过滤的随机游走
def extract_subgraph_filtered(graph, num_nodes, filter_nodes):
    print(f'Random walk to {num_nodes}')

    # Ensure the number of nodes requested does not exceed the total number of nodes in the graph
    num_nodes = min(num_nodes, len(filter_nodes))

    visited_nodes = set()

    with tqdm(total=num_nodes, desc="Selecting nodes") as pbar:
        while len(visited_nodes) < num_nodes:
            # Choose a starting node from the remaining nodes in filter_nodes
            potential_start_nodes = list(set(filter_nodes) - visited_nodes)
            if not potential_start_nodes:
                break  # No more nodes to choose from within filter_nodes

            current_node = random.choice(potential_start_nodes)
            visited_nodes.add(current_node)
            pbar.update(1)

            # Perform the random walk
            while len(visited_nodes) < num_nodes:
                # Choose neighbors from filter_nodes
                neighbors = [n for n in graph.edge_index[1][graph.edge_index[0] == current_node].tolist() if
                             n in filter_nodes and n not in visited_nodes]

                if not neighbors:
                    break  # No more neighbors to explore from the current node within filter_nodes
                current_node = random.choice(neighbors)
                visited_nodes.add(current_node)
                pbar.update(1)
    # Extract the subgraph
    subgraph_nodes = list(visited_nodes)
    sub_edge_index, sub_edge_attr = subgraph(subgraph_nodes, graph.edge_index, edge_attr=graph.edge_attr, relabel_nodes=True)
    # Create a new Data object for the subgraph
    subgraph_data = Data(x=graph.x[subgraph_nodes], edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=graph.y[subgraph_nodes])
    # Assuming graph_view is a function to visualize the graph
    graph_view(subgraph_data)

    return subgraph_nodes

def extract_subgraph(graph, num_nodes):
	print('[Random walk for subgraph...]')

	# Ensure the number of nodes requested does not exceed the total number of nodes in the graph
	num_nodes = min(num_nodes, graph.num_nodes)

	visited_nodes = set()

	with tqdm(total=num_nodes, desc="Selecting nodes") as pbar:
		while len(visited_nodes) < num_nodes:
			# Choose a starting node from the remaining nodes, avoiding already visited nodes
			potential_start_nodes = list(set(range(graph.num_nodes)) - visited_nodes)
			if not potential_start_nodes:
				break  # No more nodes to choose from

			current_node = random.choice(potential_start_nodes)
			visited_nodes.add(current_node)
			pbar.update(1)

			# Perform the random walk
			while len(visited_nodes) < num_nodes:
				neighbors = [n for n in graph.edge_index[1][graph.edge_index[0] == current_node].tolist()]
				neighbors = list(set(neighbors) - visited_nodes)

				if not neighbors:
					break

				current_node = random.choice(neighbors)
				visited_nodes.add(current_node)
				pbar.update(1)

	# Extract the subgraph
	subgraph_nodes = list(visited_nodes)
	sub_edge_index, sub_edge_attr = subgraph(subgraph_nodes, graph.edge_index, edge_attr=graph.edge_attr,
											 relabel_nodes=True)

	# Create a new Data object for the subgraph
	subgraph_data = Data(x=graph.x[subgraph_nodes], edge_index=sub_edge_index, edge_attr=sub_edge_attr,
						 y=graph.y[subgraph_nodes])

	graph_view(subgraph_data)

	return subgraph_nodes


def create_part_graph(user_dict, address_to_index, data):

	# 反向字典 无敌
	reverse_dict = {v: k for k, v in address_to_index.items()}

	# 1.筛选中心恶意节点
	print('[Step 1 Filter phisher nodes].....')
	print(f'original_dict numbers: {len(user_dict.keys())}')
	user_category_1 = []  # 存储"category" = 1的用户
	user_category_0 = []  # 存储"category" = 0的用户
	for user, details in user_dict.items():
		if 1000 >= details['all_cnt'] >= 3:
			if details['category'] == 1:
				user_category_1.append(user)
			elif details['category'] == 0:
				user_category_0.append(user)
	user_filterd = user_category_1 + user_category_0
	print(f'Total phisher numbers: {len(user_category_1)}')  # 996
	print(f'Total normal numbers: {len(user_category_0)}')  # 282536
	# 过滤后的正常节点
	rw_node_list=[]
	for user in user_category_0:
		node = address_to_index.get(user)
		rw_node_list.append(node)

	# 恶意用户的数量
	user_1_num = 10
	ratio = 20
	user_0_num = user_1_num * ratio

	user_num = user_0_num + user_1_num
	print(f'Expect {user_num} nodes')

	# 进行随机抽样
	user_category_1_sampled = random.sample(user_category_1, min(user_1_num, len(user_category_1)))

	print(f'sampled phisher numbers: {len(user_category_1_sampled)}')

	# 筛选得到的地址列表
	filtered_list = set(user_category_1_sampled)

	# 寻找邻居用的
	selected_nodes_indices = [address_to_index[user] for user in filtered_list]

	# 2.搜索恶意节点的邻居
	# 预计算邻居节点字典
	print('[Step 2 Find phisher neighbour nodes].....')
	neighbor_dict = {}
	for start, end in zip(*data.edge_index.tolist()):
		if start not in neighbor_dict:
			neighbor_dict[start] = set()
		# if end in user_filterd:
		neighbor_dict[start].add(end)
	dump_pkl('neighbor_dict.pkl', neighbor_dict)

	# 找到他们的邻居节点索引
	neighbour_nodes_indices = set()
	for node_idx in tqdm(selected_nodes_indices, desc='Neighbour finding'):
		neighbours = neighbor_dict.get(node_idx, set())
		neighbour_nodes_indices.update(neighbours)
	print(f'neighbours num {len(neighbour_nodes_indices)}')

	# 节点索引转地址
	for node_idx in neighbour_nodes_indices:
		address = reverse_dict.get(node_idx)
		filtered_list.add(address)

	# 3.随机游走选择剩余节点
	print('[Step 3 Random walk find normal user].....')
	lake_num=user_num-len(filtered_list)
	# 只选择3-1000的正常节点，因为特征很重要，不能随便选点
	user0_nodes_indices=extract_subgraph_filtered(data,lake_num,rw_node_list)
	# 节点索引转地址
	for node_idx in user0_nodes_indices:
		address = reverse_dict.get(node_idx)
		filtered_list.add(address)

	print(f'unfiltered nodes num {len(filtered_list)}')

	# 合并,得到子图节点索引
	print('[Merge nodes].....')
	filter_nodes=[]
	up1000=[]
	below3=[]
	for user in filtered_list:
		# if 1000 < user_dict[user]['all_cnt']:
		# 	up1000.append(address_to_index[user])
		# 	continue
		# if 3 > user_dict[user]['all_cnt']:
		# 	below3.append(address_to_index[user])

		filter_nodes.append(address_to_index[user])

	# print(f'up1000 {len(up1000)}')
	# print(f'below3 {len(below3)}')

	dump_pkl('filtered_list.pkl', filtered_list)
	dump_pkl('filter_nodes_indices.pkl', filter_nodes)

	print(f'filtered nodes num {len(filter_nodes)}')


	# 更新节点的特征向量
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
	graph_view(sub_graph)
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


def graph_view(graph):
	num_nodes = graph.num_nodes
	print("Number of nodes:", num_nodes)

	num_edges = graph.num_edges
	print("Number of edges:", num_edges)

	# 图的入度
	degree = num_edges / (num_nodes * 2)
	print("Degree of the graph: {:.6f}".format(degree))

	# 获取 y=1 和 y=0 的节点数量
	y_zero = (graph.y == 0).sum().item()
	y_one = (graph.y == 1).sum().item()

	print("Number of nodes where y=0:", y_zero)
	print("Number of nodes where y=1:", y_one)

	# y=0 和 y=1 的节点比例
	ratio = y_zero / y_one if y_one != 0 else float('inf')
	print("Ratio of y=0 to y=1 nodes: {:.2f}".format(ratio))


if __name__ == '__main__':
	graph_convert(test=False)


# train_sub_graph, test_sub_graph = read_pkl('train+test_dataC.pkl')

#
# graph_view(train_sub_graph)
# graph_view(test_sub_graph)
