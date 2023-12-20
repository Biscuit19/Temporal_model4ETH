import os
import pickle

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


def normalize_tensor(tensor):
	mean = tensor.mean(dim=0)  # 计算均值
	std = tensor.std(dim=0)  # 计算标准差
	normalized_tensor = (tensor - mean) / (std + 1e-8)  # 归一化，并避免除以零
	return normalized_tensor


def data_normalize():
	graph = read_pkl('whole_graph_data.pkl')
	graph.x = normalize_tensor(graph.x)
	graph.edge_attr = normalize_tensor(graph.edge_attr)
	data = Data(x=graph.x, edge_attr=graph.edge_attr, edge_index=graph.edge_index, y=graph.y)
	with open('whole_graph_data.pkl', 'wb') as file:
		pickle.dump(data, file)


def create_node_features(user_info):
	node_features = [
		user_info["Time_Diff_Between_first_and_last(Mins)"],
		user_info["Min_Val_Received"],
		user_info["Min_Val_Sent"],
		user_info["Total_Ether_Balance"],
		user_info["Avg_Val_Received"],
		user_info["Avg_Val_Sent"],
		user_info["Avg_time_Between_received_tnx"],
		user_info["Avg_time_Between_sent_tnx"],
		user_info["Max_Val_Received"],
		user_info["Unique_Received_From_Addresses"],
		user_info["Unique_Sent_To_Addresses"],
		user_info["Total_Ether_Sent"]
	]

	for key in user_info:
		if key.startswith("hidden_"):
			node_features.append(user_info[key])

	return node_features


def create_graph(user_dict, tnx_list):
	x = []  # 存储节点特征向量
	y = []  # 存储节点标签
	edge_index_start = []  # 存储边的起始节点
	edge_index_end = []  # 存储边的结束节点
	edge_attr = []  # 存储边的特征

	tmp_feature=[]
	print(f'tmp_feature:{tmp_feature}')

	# 创建地址到索引的映射
	address_to_index = {}
	for addr in tqdm(user_dict.keys(), desc='Establishing Node index'):
		# 第x个地址对应的索引为x，这里的addr是str类型
		address_to_index[addr] = len(x)
		x.append(tmp_feature)
		y.append(user_dict[addr]["category"])

	dump_pkl('address_to_index.pkl', address_to_index)



	print(f'node numbers: {len(x)}')

	# tnx的结构并没有改变
	# tran[0]~tran[5]:
	# amount, block_timestamp, timewindow, from_address, to_address,tnx_hash
	# 使用tqdm包装循环以添加进度条
	for trans in tqdm(tnx_list, desc="Establishing edges"):
		try:
			# 找到对应索引，这里的tran4 5都是str类型
			from_node = address_to_index[trans[3]]
			to_node = address_to_index[trans[4]]
			# 建立从from-to node的边
			edge_index_start.append(from_node)
			edge_index_end.append(to_node)

			# 添加边特征（取交易的前2个值amout time作为特征）
			edge_attr.append(trans[:2])
		except KeyError as e:
			print(e)

	# 创建图数据对象，边索引与节点标签不需要归一化
	edge_index = torch.tensor([edge_index_start, edge_index_end], dtype=torch.long)
	y = torch.tensor(y, dtype=torch.long)
	# 边特征需要归一化
	edge_attr = normalize_tensor(torch.tensor(edge_attr, dtype=torch.float))
	# 节点特征还没加，先不需要归一化
	x = (torch.tensor(x, dtype=torch.float))

	print(f'create node: {len(x)}; create edge {len(edge_attr)}')

	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

	return data


def graph_convert():

	accounts_dict = read_pkl('./all_account_data_zsu.pkl')

	# accounts_dict = {
	# 	'a': {'all_trans': [[0.05, 1611664539, -1, 93, '', '', ''], [0.05, 1611664539, -1, 93, '', '', '']],
	# 		  'category':0},
	# 	'b': {'all_trans': [[0.05, 1611664539, -1, 93, '', '', ''], [0.05, 1611664539, -1, 93, '', '', '']],
	# 		  'category': 1}
	# }

	tnx_list = read_pkl('./all_tnx_data_zsu.pkl')
	graph_data = create_graph(accounts_dict, tnx_list)
	# 存储为Pickle文件
	with open('whole_graph_data.pkl', 'wb') as file:
		pickle.dump(graph_data, file)


if __name__ == '__main__':
	graph_convert()

