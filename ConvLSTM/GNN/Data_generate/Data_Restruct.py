# -*- codeing = utf-8 -*-
# 1.填充序列长度 2.数据字典转换成数据张量 3.张量标准化
import os
import pickle
import pprint
import random

import numpy as np
import torch
from torch import tensor, float32
from tqdm import tqdm

torch.set_printoptions(precision=4, sci_mode=False, threshold=10000000)


def time_window_generate(pkl_file):
	with open(pkl_file, 'rb') as f:
		acct_dict = pickle.load(f)

	acct_list = list(acct_dict.keys())

	print(f'{len(acct_list)} accounts')

	# 寻找时间跨度
	# min_time = []
	# max_time = []
	# for acct, attributes in acct_dict.items():
	# 	# [0.05, 1588198188, 'in', 0],
	# 	all_trans = attributes['all_trans']
	# 	# 序列第一个交易是时间最小的
	# 	min_timestamp = all_trans[0][1]
	# 	min_time.append(min_timestamp)
	# 	# 最后一个交易是时间最大的
	# 	max_timestamp = all_trans[-1][1]
	# 	max_time.append(max_timestamp)
	# min_timestamp = min(min_time)
	# max_timestamp = max(max_time)
	# print(f'all_trans:{min_timestamp} - {max_timestamp}')

	# 589270 normal accounts
	# all_trans:1498251906 - 1646092669 / 2017-06-24 05:05:06 - 2022-03-01 07:57:49

	# 3835 phishing accounts
	# all_trans:1498261945 - 1646090962 / 2017-06-24 07:52:25 - 2022-03-01 07:29:22

	# second per unit
	min_timestamp = 1498251906
	max_timestamp = 1646092669

	day_span = 60 * 60 * 24
	day_size = 7
	week_size = 8
	# 时间窗口跨度
	time_window_span = day_span * day_size * week_size
	# 时间窗口数量
	time_window_num = int((max_timestamp - min_timestamp) / time_window_span)
	print(f'0 - {time_window_num} time-windows totally')

	# 计算时间窗口值
	for attributes in acct_dict.values():
		# [0.05, 1611664539, -1, 93],
		all_trans = attributes['all_trans']
		for tran in all_trans:
			# 为什么all_trans值改变了,out_trans和in_trans也会改变，这是因为它们都指向同一个列表。
			# 时间窗口从0开始
			tran[3] = int((tran[1] - min_timestamp) / time_window_span)
	# print(tran[3])

	# 将交易序列按时间窗口值重新划分到不同的子序列中
	for acct, attributes in acct_dict.items():
		# [0.05, 1611664539, -1, 93],
		all_trans = attributes['all_trans']
		# create window-num empty sub lists
		windows_trans = [[] for _ in range(time_window_num + 1)]
		for tran in all_trans:
			windows_trans[tran[3]].append(tran)
		attributes['all_trans'] = windows_trans

	dump_file = ('time_windowed_account_dict.pkl')

	print(f'dumping in {dump_file}...')
	with open(dump_file, 'wb') as f:
		pickle.dump(acct_dict, f)

	return acct_dict


def data_padding(all_trans_list):
	# print('[+]Padding data...')
	trans_size = 300
	padding_tran = [0, 0, 0, 0]
	# print('padding tran:', padding_tran)

	# 处理整个数据集
	# for all_windows in tqdm(all_trans_list, desc='Processing Data', unit='data'):
	for all_windows in (all_trans_list):

		for window in all_windows:
			# 计算要填充几次
			window_len = len(window)
			trans_lack = trans_size - window_len

			if trans_lack > 0:
				# 这里使用引用，减少占用空间
				window.extend([padding_tran] * trans_lack)
			elif trans_lack < 0:
				# 随机删去超过交易长度的元素，以满足交易长度
				del_indices = random.sample(range(window_len), -trans_lack)
				del_indices.sort(reverse=True)
				for i in del_indices:
					del window[i]

	return all_trans_list


# 将账户数据字典变为交易序列的列表，将列表转为张量
def list2tensor(trans_list):
	# print('[+]Generating Tensor...')
	# 筛选最长长度为100的序列作为张量
	# acct_trans = [attribute['all_trans'] for address, attribute in acct_dict.items()
	# 			  if attribute['max_trans_len'] == 100]

	batchs = tensor(trans_list, dtype=float32)

	# 把第3 4维度交换，即 trans_size*feature_size-> feature_size* trans_size
	batchs = torch.transpose(batchs, 2, 3)

	# unsqueeze(-1)将最后一个维度变成2维 trans_size-> trans_size*1
	batchs = batchs.unsqueeze(-1)

	return batchs


def data_normalize(batchs_tensor):
	# print('[+]Normalizing...')

	stand_dim = 3

	mean = batchs_tensor.mean(dim=stand_dim).unsqueeze(stand_dim)
	epsilon = 1e-8
	std = batchs_tensor.std(dim=stand_dim).unsqueeze(stand_dim)

	# 标准化张量
	normalized_batch = (batchs_tensor - mean) / (std + epsilon)
	# normalized_batch=torch.nn.functional.normalize(batchs_tensor, dim=2)

	return normalized_batch


def conv_train_data_generate(pkl_file):
	with open(pkl_file, 'rb') as f:
		acct_dict = pickle.load(f)

	def filter_account():
		filtered_dict = {}
		count_range = (1, 1000)
		ratio = 2

		# 筛选category=1的账户
		for account, attributes in acct_dict.items():
			if attributes['category'] == 1:
				filtered_dict[account] = attributes

		phish_num = len(list(filtered_dict.keys()))
		normal_num = phish_num * ratio
		print(f'phisher num {phish_num} , ratio 1:{ratio}')


		# 筛选category=0且交易数在1-1000之间的账户
		category_0_accounts = [account for account, attributes in acct_dict.items() if
							   attributes['category'] == 0 and count_range[0] <= attributes['all_cnt'] < count_range[1]]

		# 随机抽取normal_num个账户
		sampled_category_0_accounts = random.sample(category_0_accounts, normal_num)

		# 将随机抽取的账户加入filtered_dict
		for account in sampled_category_0_accounts:
			filtered_dict[account] = acct_dict[account]

		return filtered_dict

	sample_dict = filter_account()

	sampled_trans = [attribute['all_trans'] for address, attribute in sample_dict.items()]

	padding_trans = data_padding(sampled_trans)

	batch = list2tensor(padding_trans)

	batch = data_normalize(batch)

	# Save the processed batch as a new pkl file
	batch_dump_file = 'conv_train_data.pkl'
	with open(batch_dump_file, 'wb') as f:
		pickle.dump(batch, f)


if __name__ == '__main__':
	acc_pkl_file= 'conv_account_dict.pkl'
	time_window_generate(acc_pkl_file)

	time_pkl_file = 'time_windowed_account_dict.pkl'
	conv_train_data_generate(time_pkl_file)
