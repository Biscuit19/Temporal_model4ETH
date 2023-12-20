import pprint

import pandas as pd
import functools
import pickle
from tqdm import tqdm
import networkx as nx


def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		data = pickle.load(file)
	return data

# 按时间从小到大排序列表
def cmp_time_ascending(tran1, tran2):
	# 比较两个交易的时间
	time1 = int(tran1[1])
	time2 = int(tran2[1])
	# 按照从小到大的时间戳排序
	if time1 > time2:
		# 1 表示将第一个元素(tran1)放在第二个元素(tran2)之后。
		# 如果time1比time2大，那么tran1会被放在后面
		return 1
	elif time1 < time2:
		return -1
	else:
		return 0


def accounts_data_generate(G):
	# 账户大字典
	accounts_dict = {}
	trans_hash=0
	for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True),desc=f'accounts_data_generate'):
		amount = tnx_info['amount']
		block_timestamp = int(tnx_info['timestamp'])
		# 时间窗口
		timewindow = 0
		# 滤除交易额为0的交易
		if amount == 0:
			continue
		trans_hash+=1

		# 先输入from_address
		try:
			# 对于转出地址而言，这条交易的方向是-1
			out_tnx = [amount, block_timestamp, -1, timewindow, from_address, to_address,trans_hash]
			accounts_dict[from_address]['out_trans'].append(out_tnx)
			accounts_dict[from_address]['all_trans'].append(out_tnx)
		# 如果地址不存在于字典中，则初始化该地址
		except:
			tag = G.nodes[from_address]['isp']
			out_tnx = [amount, block_timestamp, -1, timewindow, from_address, to_address,trans_hash]
			accounts_dict[from_address] = {
				'account_address': from_address,
				'out_trans': [out_tnx],
				'in_trans': [],
				'all_trans': [out_tnx],
				# 初始化时，地址按照数据集标签给
				'category': tag
			}

		# 输入to_address
		try:
			# 对于转入地址而言，这条交易的方向是1
			in_tnx = [amount, block_timestamp, 1, timewindow, from_address, to_address,trans_hash]
			accounts_dict[to_address]['in_trans'].append(in_tnx)
			accounts_dict[to_address]['all_trans'].append(in_tnx)

		except:
			in_tnx = [amount, block_timestamp, 1, timewindow, from_address, to_address,trans_hash]
			tag = G.nodes[to_address]['isp']
			accounts_dict[to_address] = {
				'account_address': to_address,
				'out_trans': [],
				'in_trans': [in_tnx],
				'all_trans': [in_tnx],
				'category': tag
			}

	for account, info in tqdm(accounts_dict.items(), desc="Processing accounts tnx"):
		info['out_trans'] = sorted(info['out_trans'], key=functools.cmp_to_key(cmp_time_ascending))
		info['in_trans'] = sorted(info['in_trans'], key=functools.cmp_to_key(cmp_time_ascending))
		info['all_trans'] = sorted(info['all_trans'], key=functools.cmp_to_key(cmp_time_ascending))

		info['out_cnt'] = len(info['out_trans'])
		info['in_cnt'] = len(info['in_trans'])
		info['all_cnt'] = len(info['all_trans'])

	with open('./all_account_data_zsu.pkl', 'wb') as f:
		pickle.dump(accounts_dict, f)

	conv_addcount_dict = {}
	for address, user_data in tqdm(accounts_dict.items(), desc="Generating conv_account_dict"):
		new_user_data = user_data.copy()  # 复制用户数据
		new_user_data['out_trans'] = [trans[:-3] for trans in user_data['out_trans']]
		new_user_data['in_trans'] = [trans[:-3] for trans in user_data['in_trans']]
		new_user_data['all_trans'] = [trans[:-3] for trans in user_data['all_trans']]
		conv_addcount_dict[address] = new_user_data
	with open('./conv_account_dict.pkl', 'wb') as f:
		pickle.dump(conv_addcount_dict, f)

	return accounts_dict


def tnx_data_generate(G):
	tnx_data = []
	amount_zero = 0

	for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True),desc=f'tnx_data_generate'):
		amount=tnx_info['amount']
		block_timestamp=int(tnx_info['timestamp'])
		# 时间窗口
		timewindow = 0

		# 滤除交易额为0的交易
		if amount == 0:
			amount_zero += 1
			continue
		# 滤除错误交易数据
		if isinstance(from_address, float) or isinstance(to_address, float):
			continue

		tnx = [amount, block_timestamp, timewindow, from_address, to_address]
		tnx_data.append(tnx)

	print(f'out_amount_zero:{amount_zero}')

	# 按时间排序
	tnx_data = sorted(tnx_data, key=functools.cmp_to_key(cmp_time_ascending))

	with open('./all_tnx_data_zsu.pkl', 'wb') as f:
		pickle.dump(tnx_data, f)

	return tnx_data


def data_generate():
	graph_file='./MulDiGraph.pkl'
	graph=read_pkl(graph_file)
	tnx_data_generate(graph)
	accounts_data_generate(graph)



def account_pkl_read(pklfile):
	print('Reading pkl file...')
	with open(pklfile, 'rb') as f:
		accounts_dict = pickle.load(f)

	# 排序一下，这样每次打印出来的内容是一样的，方便看
	def account_analyze():
		# 列表推导式筛选 category 等于 1 的账户
		phish_accounts = [account for account in accounts_dict.values() if account.get('category') == 1]
		normal_accounts = [account for account in accounts_dict.values() if account.get('category') == 0]
		# 输出筛选结果
		print(f"account amount: {len(accounts_dict.keys())}")
		print(f"phish amount: {len(phish_accounts)}")
		print(f"normal amount: {len(normal_accounts)}")

	account_analyze()

	account_list = sorted(list(accounts_dict.keys()))
	account_num = len(account_list)
	print(f'{pklfile} have {account_num} accounts')

	print_dict = {key: accounts_dict[key] for key in account_list[:2]}
	print(print_dict)

def tnx_pkl_read(pklfile):
	print('Reading pkl file...')
	with open(pklfile, 'rb') as f:
		tnx_data = pickle.load(f)

	print(f'{pklfile} have {len(tnx_data)} tnx')

	print('最前',tnx_data[0])
	print('最后',tnx_data[-1])

	print_dict = tnx_data[0:2]
	# 优雅输出
	pprint.pprint(print_dict, indent=4, depth=2)


if __name__ == '__main__':
	# data_generate()

	pkl_file='all_account_data_zsu.pkl'
	account_pkl_read(pkl_file)

	tnx_pkl_read('all_tnx_data_zsu.pkl')

