import json
import pprint

import numpy as np
import pandas as pd
import functools
import os
import pickle
import beeprint

# 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'

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


def load_data_for_GATDataset(file_in, file_out):
	# 账户所有进出交易字典
	accounts_all_trans = {}
	error_trans = []

	# 账户的出交易字典
	accounts_out_trans = {}

	# hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type
	print('Reading out csv')
	df = pd.read_csv(file_out)
	print(f'out: {len(df)}')

	amount_zero = 0

	for index, row in df.iterrows():
		# 获取每一列的值
		trans_hash = (row['hash'])
		# 交易地址
		from_address = row['from_address']
		to_address = row['to_address']

		# 这里原始的交易额度单位是Wei，1eth=10^18 Wei
		amount = int(row['value']) / (10 ** 18)

		# 块创建时间戳，实际上，某个交易并不具有确切的时间戳，仅用块的时间戳代表
		block_timestamp = int(row['block_timestamp'])

		# 交易方向 -1表示out
		direction = -1
		# direction = 'out'
		# 时间窗口
		timewindow = 0

		# 其他属性
		# 交易块
		# block_number = int(row['block_number'])
		# nonce = row['nonce']
		# block_hash = row['block_hash']
		# transaction_index = int(row['transaction_index'])
		# max_fee_per_gas = row['max_fee_per_gas']
		# max_priority_fee_per_gas = row['max_priority_fee_per_gas']
		# transaction_type = row['transaction_type']
		# gas = row['gas']
		# gas_price = row['gas_price']
		# input_data = row['input']

		# 滤除失败交易
		if from_address == "" or to_address == "":
			error_trans.append(trans_hash)
			continue

		# 滤除交易额为0的交易
		if amount == 0:
			amount_zero += 1
			continue

		try:
			accounts_out_trans[from_address].append(
				[amount, block_timestamp, direction, timewindow, from_address, to_address])
		except:
			# 地址未存在于字典中
			accounts_out_trans[from_address] = [
				[amount, block_timestamp, direction, timewindow, from_address, to_address]]
	print(f'out_amount_zero:{amount_zero}')

	amount_zero = 0
	accounts_in_trans = {}
	df = pd.read_csv(file_in)
	print(f'in: {len(df)}')


	for index, row in df.iterrows():
		# 获取每一列的值
		trans_hash = (row['hash'])
		# 交易地址
		from_address = row['from_address']
		to_address = row['to_address']
		# 这里原始的交易额度单位是Wei，1eth=10^18 Wei
		amount = int(row['value']) / (10 ** 18)

		# 块创建时间戳，实际上，某个交易并不具有确切的时间戳，仅用块的时间戳代表
		block_timestamp = int(row['block_timestamp'])

		# 交易方向 1表示in
		direction = 1
		# 时间窗口
		timewindow = 0

		if from_address == "" or to_address == "":
			error_trans.append(trans_hash)
			continue

		# 滤除交易额为0的交易
		if amount == 0:
			amount_zero += 1
			continue

		try:
			accounts_in_trans[to_address].append(
				[amount, block_timestamp, direction, timewindow, from_address, to_address])
		except:
			accounts_in_trans[to_address] = [[amount, block_timestamp, direction, timewindow, from_address, to_address]]

	print(f'in_amount_zero:{amount_zero}')



	return accounts_in_trans, accounts_out_trans


# 为第一步卷积lstm训练加载数据集
def load_data_for_convlstmDataset(file_in, file_out):
	# 账户所有进出交易字典
	accounts_all_trans = {}
	error_trans = []

	# 账户的出交易字典
	accounts_out_trans = {}

	# hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type
	print('Reading out csv')
	df = pd.read_csv(file_out)

	for index, row in df.iterrows():
		# 获取每一列的值
		trans_hash = (row['hash'])
		# 交易地址
		from_address = row['from_address']
		to_address = row['to_address']

		# 这里原始的交易额度单位是Wei，1eth=10^18 Wei
		amount = int(row['value']) / (10 ** 18)

		# 块创建时间戳，实际上，某个交易并不具有确切的时间戳，仅用块的时间戳代表
		block_timestamp = int(row['block_timestamp'])

		# 交易方向 -1表示out
		direction = -1
		# direction = 'out'
		# 时间窗口
		timewindow = 0

		# 其他属性
		# 交易块
		# block_number = int(row['block_number'])
		# nonce = row['nonce']
		# block_hash = row['block_hash']
		# transaction_index = int(row['transaction_index'])
		# max_fee_per_gas = row['max_fee_per_gas']
		# max_priority_fee_per_gas = row['max_priority_fee_per_gas']
		# transaction_type = row['transaction_type']
		# gas = row['gas']
		# gas_price = row['gas_price']
		# input_data = row['input']

		# 失败交易
		if from_address == "" or to_address == "":
			error_trans.append(trans_hash)
			continue

		try:
			accounts_out_trans[from_address].append([amount, block_timestamp, direction, timewindow])
		except:
			# 地址未存在于字典中
			accounts_out_trans[from_address] = [[amount, block_timestamp, direction, timewindow]]

	accounts_in_trans = {}
	df = pd.read_csv(file_in)

	for index, row in df.iterrows():
		# 获取每一列的值
		trans_hash = (row['hash'])
		# 交易地址
		from_address = row['from_address']
		to_address = row['to_address']
		# 这里原始的交易额度单位是Wei，1eth=10^18 Wei
		amount = int(row['value']) / (10 ** 18)

		# 块创建时间戳，实际上，某个交易并不具有确切的时间戳，仅用块的时间戳代表
		block_timestamp = int(row['block_timestamp'])

		# 交易方向 1表示in
		direction = 1
		# 时间窗口
		timewindow = 0

		if from_address == "" or to_address == "":
			error_trans.append(trans_hash)
			continue
		try:
			accounts_in_trans[to_address].append([amount, block_timestamp, direction, timewindow])
		except:
			accounts_in_trans[to_address] = [[amount, block_timestamp, direction, timewindow]]

	return accounts_in_trans, accounts_out_trans


def accounts_generate(accounts_in_trans, accounts_out_trans, category):
	# 转出字典里，from的地址全都是phisher
	out_list = list(accounts_out_trans.keys())
	# 转入字典里,in的地址全是phisher
	in_list = list(accounts_in_trans.keys())
	# eoa_list must include eoa account only (i.e., have out transaction at least)
	accounts_list = list(set(out_list).union(set(in_list)))
	accounts_dict = {}

	for account in accounts_list:
		# 这里是对的,即这个账户字典里的标签都是正确的,没有重复
		accounts_dict[account] = {
			'account_address':account,
			'out_trans': [],
			'in_trans': [],
			'all_trans': [],
			'out_cnt': 0,
			'in_cnt': 0,
			'all_cnt': 0,
			'category': category
		}

		try:
			out_trans = accounts_out_trans[account]
			out_trans = sorted(out_trans, key=functools.cmp_to_key(cmp_time_ascending))
		except:
			out_trans = []
		try:
			in_trans = accounts_in_trans[account]
			in_trans = sorted(in_trans, key=functools.cmp_to_key(cmp_time_ascending))
		except:
			in_trans = []
		all_trans = sorted(out_trans + in_trans, key=functools.cmp_to_key(cmp_time_ascending))

		accounts_dict[account]['out_trans'] = out_trans
		accounts_dict[account]['in_trans'] = in_trans
		accounts_dict[account]['all_trans'] = all_trans
		accounts_dict[account]['out_cnt'] = len(out_trans)
		accounts_dict[account]['in_cnt'] = len(in_trans)
		accounts_dict[account]['all_cnt'] = len(all_trans)

	# 筛选字典
	remove_account = []
	for account, attributes in accounts_dict.items():
		# 筛选交易数为5-500的地址
		if attributes['all_cnt'] < 5 or attributes['all_cnt'] > 1000:
			remove_account.append(account)

	for rm_account in remove_account:
		del accounts_dict[rm_account]

	return accounts_dict


def phish_account_load(dataset=1):
	print("phish_account_load...")
	dataset_dir = './phish_trans/'
	in_trans_file = open((dataset_dir + "phisher_transaction_in.csv"), "r")
	out_trans_file = open((dataset_dir + "phisher_transaction_out.csv"), "r")

	if dataset == 1:
		# load_data_for_convlstmDataset
		accounts_in_trans, accounts_out_trans = load_data_for_convlstmDataset(in_trans_file, out_trans_file)
		phisher_dict = accounts_generate(accounts_in_trans, accounts_out_trans, 1)

		print(phisher_dict['0x634cdd7ebeccccf517cd8f4eca959474bc1b58cc'])

		with open('./phish_account_data.pkl', 'wb') as f:
			pickle.dump(phisher_dict, f)

	if dataset == 2:
		# load_data_for_GATDataset
		accounts_in_trans, accounts_out_trans = load_data_for_GATDataset(in_trans_file, out_trans_file)
		phisher_dict = accounts_generate(accounts_in_trans, accounts_out_trans, 1)

		print(phisher_dict['0x634cdd7ebeccccf517cd8f4eca959474bc1b58cc'])

		with open('./phish_account_data_GAT.pkl', 'wb') as f:
			pickle.dump(phisher_dict, f)
	return


def normal_account_load(dataset=1):
	print("normal_account_load...")
	dataset_dir = './normal_trans/'
	print("Reading_normal_eoa_transaction_in_slice_1000K...")
	in_trans_file = open((dataset_dir + "normal_eoa_transaction_in_slice_1000K.csv"), "r")
	print("Reading_normal_eoa_transaction_out_slice_1000K...")

	out_trans_file = open((dataset_dir + "normal_eoa_transaction_out_slice_1000K.csv"), "r")

	if dataset == 1:
		print("normal_accounts_generate...")
		accounts_in_trans, accounts_out_trans = load_data_for_convlstmDataset(in_trans_file, out_trans_file)
		normal_dict = accounts_generate(accounts_in_trans, accounts_out_trans, 0)
		with open('./normal_account_data.pkl', 'wb') as f:
			pickle.dump(normal_dict, f)

	if dataset == 2:
		print("normal_accounts_generate...")
		accounts_in_trans, accounts_out_trans = load_data_for_GATDataset(in_trans_file, out_trans_file)
		normal_dict = accounts_generate(accounts_in_trans, accounts_out_trans, 0)
		with open('./normal_account_data_GAT.pkl', 'wb') as f:
			pickle.dump(normal_dict, f)

	return


def pkl_read_test(pklfile):
	print('Reading pkl file...')
	with open(pklfile, 'rb') as f:
		dict = pickle.load(f)
	# 排序一下，这样每次打印出来的内容是一样的，方便看
	account_list = sorted(list(dict.keys()))
	account_num = len(account_list)
	print(f'{pklfile} have {account_num} accounts')

	print_dict = {key: dict[key] for key in account_list[0:2]}

	# 优雅输出
	# pprint.pprint(dict,indent=4,depth=2)
	beeprint.pp(print_dict, sort_keys=False)


# beeprint.pp(dict['0xed1eab18667ad15c161a227c462f5196bf071580'],sort_keys=False)


if __name__ == '__main__':
	# 参数1代表生成第一步的DataSet，2代表生成第二步的dataset
	phish_account_load(1)

	normal_account_load(1)

	# pkl_file = './phish_account_data.pkl'
	#
	# pkl_read_test(pkl_file)
