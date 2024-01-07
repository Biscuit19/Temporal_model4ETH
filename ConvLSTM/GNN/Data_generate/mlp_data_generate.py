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


def create_mlp_data(user_dict):
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
	x = 900
	# 进行随机抽样
	user_category_1_sampled = random.sample(user_category_1, min(x, len(user_category_1)))
	user_category_0_sampled = random.sample(user_category_0,
											min(20 * len(user_category_1_sampled), len(user_category_0)))

	print(f'sampled phisher numbers: {len(user_category_1_sampled)}')
	print(f'sampled normal numbers: {len(user_category_0_sampled)}')

	# 筛选得到的地址列表
	filtered_list = user_category_0_sampled + user_category_1_sampled

	# 4.更新节点的特征向量
	print('[Step 2 Update node Features].....')

	# 设定目标维度 num
	convlstm_hidden_num = 64
	print(f'hidden_temporal feature size: {convlstm_hidden_num}')
	feature_num = 17 + convlstm_hidden_num

	AECL_model = load_model()
	features = []  # 用于存储所有用户的特征
	labels = []  # 用于存储所有用户的标签

	for address in tqdm(filtered_list, desc='Updating Features'):
		feature = user_dict[address]
		# 生成特征
		address_features = feature_generate(feature, AECL_model)  # 假设这返回一个input_size维的张量
		features.append(address_features)

		# 获取标签
		address_tag = user_dict[address]['category']  # 假设这是0或1
		labels.append(address_tag)

	# 将特征列表和标签列表转换为张量
	features_tensor = normalize_tensor(torch.stack(features))

	only_embed_features_tensor = features_tensor[:, 17:]

	only_static_features_tensor= features_tensor[:, :17]

	labels_tensor = torch.tensor(labels).unsqueeze(1).float()  # 转换为二维张量并确保标签是浮点类型

	train_data = (features_tensor, labels_tensor)

	only_embed_train_data = (only_embed_features_tensor, labels_tensor)

	only_static_train_data = (only_static_features_tensor, labels_tensor)

	dump_pkl('mlp_data_all_feature.pkl',train_data)
	dump_pkl('mlp_data_embed.pkl',only_embed_train_data)
	dump_pkl('mlp_data_static.pkl',only_static_train_data)

	return


def mlp_data():
	user_dict = read_pkl('all_account_data_zsu.pkl')
	create_mlp_data(user_dict)


if __name__ == '__main__':
	mlp_data()
