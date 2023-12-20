import pprint

import pandas as pd
import functools
import pickle
from tqdm import tqdm


def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		data = pickle.load(file)
	return data


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


def accounts_data_generate(phish_in, phish_out, normal_in, normal_out):
	# 账户大字典
	accounts_dict = {}

	# phish_in
	def load_data(file, in_out, phisher_normal):
		# hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type
		print(f'Reading out csv {file}')
		df = pd.read_csv(file)

		error_trans = []
		amount_zero = 0

		for index, row in tqdm(df.iterrows(), total=len(df), desc=f'Load tnx'):
			# 获取每一列的值
			trans_hash = (row['hash'])
			# 交易地址
			from_address = row['from_address']
			to_address = row['to_address']
			# 这里原始的交易额度单位是Wei，1eth=10^18 Wei
			amount = int(row['value']) / (10 ** 18)
			# 块创建时间戳，实际上，某个交易并不具有确切的时间戳，仅用块的时间戳代表
			block_timestamp = int(row['block_timestamp'])
			# 时间窗口
			timewindow = 0
			# 方向，-1表示转出，1表示转入
			direction = 0

			# 滤除失败交易
			if from_address == "" or to_address == "":
				error_trans.append(trans_hash)
				continue

			# 滤除交易额为0的交易
			if amount == 0:
				amount_zero += 1
				continue

			# 	如果是in数据集
			if in_out == 1:
				direction = 1
				tnx = [amount, block_timestamp, direction, timewindow]
				try:
					accounts_dict[to_address]['all_trans'].append(tnx)
				# 如果地址不存在于字典中，则初始化该地址
				except:
					accounts_dict[to_address] = {
						'all_trans': [tnx],
						# 初始化时，地址按照数据集标签给
						'category': phisher_normal
					}
			elif in_out == -1:
				direction = -1
				tnx = [amount, block_timestamp, direction, timewindow]
				try:
					accounts_dict[from_address]['all_trans'].append(tnx)
				# 如果地址不存在于字典中，则初始化该地址
				except:
					accounts_dict[from_address] = {
						'all_trans': [tnx],
						# 初始化时，地址按照数据集标签给
						'category': phisher_normal
					}

		print(f'out_amount_zero:{amount_zero}')
		print(f'error_trans:{len(error_trans)}')

	load_data(phish_in, 1, 1)
	load_data(phish_out, -1, 1)

	load_data(normal_in, 1, 0)
	load_data(normal_out, -1, 0)

	def process_dict():
		for account, info in tqdm(accounts_dict.items(), desc="Processing accounts tnx"):
			info['all_trans'] = sorted(info['all_trans'], key=functools.cmp_to_key(cmp_time_ascending))
			info['all_cnt'] = len(info['all_trans'])

	process_dict()

	# 有一个地址是nan浮点类型，把它去除
	float_keys_dict = [key for key in accounts_dict.keys() if isinstance(key, float)]
	for key in float_keys_dict:
		del accounts_dict[key]

	def filter_account():
		# 筛选字典
		remove_account = []
		for account, attributes in accounts_dict.items():
			# 筛选交易数为5-500的地址
			if attributes['all_cnt'] < 5 or attributes['all_cnt'] > 500:
				remove_account.append(account)

		for rm_account in remove_account:
			del accounts_dict[rm_account]

	print(accounts_dict['0x634cdd7ebeccccf517cd8f4eca959474bc1b58cc'])

	with open('./conv_account_dict_classifer.pkl', 'wb') as f:
		pickle.dump(accounts_dict, f)

	return accounts_dict


def data_generate(type):
	phish_in_file = 'phisher_transaction_in.csv'
	phish_out_file = 'phisher_transaction_out.csv'
	normal_in_file = 'normal_eoa_transaction_in_slice_1000K.csv'
	normal_out_file = 'normal_eoa_transaction_out_slice_1000K.csv'
	if type == 1:
		accounts_data_generate(phish_in_file, phish_out_file, normal_in_file, normal_out_file)


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

	print_dict = {key: accounts_dict[key] for key in account_list[0:1]}
	print(print_dict)


# 优雅输出
# pprint.pprint(print_dict, indent=4, depth=4)


if __name__ == '__main__':
	# data_generate(1)

	# pkl_file='conv_account_dict.pkl'
	# pkl_file='conv_account_dict.pkl'
	pkl_file = 'conv_account_dict_classifer.pkl'
	account_pkl_read(pkl_file)
