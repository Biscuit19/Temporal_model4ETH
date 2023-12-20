import numpy as np
import pandas as pd
from datetime import datetime, timezone


# 读取恶意用户的交易信息
def read_scam_account():
	print('------读取恶意交易信息-------')
	data = pd.read_csv('Dataset.csv')

	# 取前1000行数据
	# data = data.head(10000)

	# 筛选from_scam或to_scam为1的条目
	data = data[((data['from_scam'] == 1) | (data['to_scam'] == 1)) & (data['value'] != 0)]

	# 创建一个空字典用于存储处理后的数据
	address_transactions = {}
	# 这个字典用于保证地址的顺序结构，使得每次遍历字典时的地址顺序是一样的，因为字典是无序的
	address_list = []

	# 遍历DataFrame中的每一行数据
	for index, row in data.iterrows():

		# 提取交易信息
		from_address = row['from_address']
		to_address = row['to_address']
		time_str = row['block_timestamp']

		try:
			# 处理两种时间字符串格式为datatime类型
			# 2018-02-04 12:37:01+00:00，2018-02-04 13:30:50 UTC
			# 尝试按照第一种格式解析
			time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S%z")
		except ValueError:
			# 如果解析失败，尝试按照第二种格式解析
			time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %Z")
			# 将时区处理为统一utc时区，统一格式，为了后面可以做时间的减法
			if time.tzinfo is None:
				time = time.replace(tzinfo=timezone.utc)

		amount = row['value'] / (10 ** 18)

		# 创建包含时间和交易额的交易信息列表
		# transaction = np.array([time, amount])
		transaction = [time, amount]

		if row['from_scam'] == 1:
			# 处理from_address的交易信息
			if from_address in address_transactions:
				address_transactions[from_address]['out'].append(transaction)
			else:
				address_list.append(from_address)
				address_transactions[from_address] = {'out': [transaction], 'in': []}

		if row['to_scam'] == 1:
			# 处理to_address的交易信息
			if to_address in address_transactions:
				address_transactions[to_address]['in'].append(transaction)
			else:
				address_list.append(to_address)
				address_transactions[to_address] = {'out': [], 'in': [transaction]}

	in_transactions_dict = {}
	in_address_list = []
	out_transactions_dict = {}
	out_address_list = []
	both_transactions_dict = {}
	both_address_list = []

	# 遍历原始字典中的地址信息
	for address in address_list:

		in_transactions = address_transactions[address]['in']
		out_transactions = address_transactions[address]['out']

		# 仅包含in交易，不包含out交易的子字典
		if len(in_transactions) > 0 and len(out_transactions) == 0:
			in_transactions_dict[address] = {'in': in_transactions, 'out': out_transactions}
			in_address_list.append(address)

		# 仅包含out交易，不包含in交易的子字典
		elif len(in_transactions) == 0 and len(out_transactions) > 0:
			out_transactions_dict[address] = {'in': in_transactions, 'out': out_transactions}
			out_address_list.append(address)


		# 包含in交易和out交易的子字典
		else:
			both_transactions_dict[address] = {'in': in_transactions, 'out': out_transactions}
			both_address_list.append(address)

	#
	# 打印处理后的数据
	print(f'总地址数：{len(address_transactions.keys())}')
	print(f'in地址数：{len(in_transactions_dict.keys())}')
	print(f'out地址数：{len(out_transactions_dict.keys())}')
	print(f'both地址数：{len(both_transactions_dict.keys())}')

	# print(address_transactions)
	# print(in_transactions_dict)
	# print(out_transactions_dict)
	# print(both_transactions_dict)

	return_dict = {
		'address_transactions': address_transactions,
		'address_list': address_list,
		'in_transactions': in_transactions_dict,
		'in_address_list': in_address_list,
		'out_transactions': out_transactions_dict,
		'out_address_list': out_address_list,
		'both_transactions': both_transactions_dict,
		'both_address_list': both_address_list
	}

	return return_dict


# 读取正常账户的交易信息
def read_normal_account():
	print('------读取正常账户交易信息-------')

	# 读取CSV文件，假设文件名为 transactions.csv
	data = pd.read_csv('Dataset.csv')

	# 取前1000行数据
	# data = data.head(10000)

	# 筛选from_scam或to_scam为0的条目
	# data = data[(data['from_scam'] == 0) & (data['to_scam'] == 0)]
	data = data[(data['from_scam'] == 0) & (data['to_scam'] == 0) & (data['value'] != '0')]

	# 创建一个空字典用于存储处理后的数据
	address_transactions = {}
	# 这个字典用于保证地址的顺序结构，使得每次遍历字典时的地址顺序是一样的，因为字典是无序的
	address_list = []

	# 遍历DataFrame中的每一行数据
	for index, row in data.iterrows():

		# 提取交易信息
		from_address = row['from_address']
		to_address = row['to_address']
		time_str = row['block_timestamp']

		try:
			# 处理两种时间字符串格式为datatime类型
			# 2018-02-04 12:37:01+00:00，2018-02-04 13:30:50 UTC
			# 尝试按照第一种格式解析
			time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S%z")
		except ValueError:
			# 如果解析失败，尝试按照第二种格式解析
			time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %Z")
			# 将时区处理为统一utc时区，统一格式，为了后面可以做时间的减法
			if time.tzinfo is None:
				time = time.replace(tzinfo=timezone.utc)

		amount = row['value'] / (10 ** 18)

		# 创建包含时间和交易额的交易信息列表
		# transaction = np.array([time, amount])
		transaction = [time, amount]

		# 处理from_address的交易信息
		if from_address in address_transactions:
			address_transactions[from_address]['out'].append(transaction)
		else:
			address_list.append(from_address)
			address_transactions[from_address] = {'out': [transaction], 'in': []}

		# 处理to_address的交易信息
		if to_address in address_transactions:
			address_transactions[to_address]['in'].append(transaction)
		else:
			address_list.append(to_address)
			address_transactions[to_address] = {'out': [], 'in': [transaction]}

	in_transactions_dict = {}
	in_address_list = []
	out_transactions_dict = {}
	out_address_list = []
	both_transactions_dict = {}
	both_address_list = []

	# 遍历原始字典中的地址信息
	for address in address_list:

		in_transactions = address_transactions[address]['in']
		out_transactions = address_transactions[address]['out']

		# 仅包含in交易，不包含out交易的子字典
		if len(in_transactions) > 0 and len(out_transactions) == 0:
			in_transactions_dict[address] = {'in': in_transactions, 'out': out_transactions}
			in_address_list.append(address)

		# 仅包含out交易，不包含in交易的子字典
		elif len(in_transactions) == 0 and len(out_transactions) > 0:
			out_transactions_dict[address] = {'in': in_transactions, 'out': out_transactions}
			out_address_list.append(address)


		# 包含in交易和out交易的子字典
		else:
			both_transactions_dict[address] = {'in': in_transactions, 'out': out_transactions}
			both_address_list.append(address)

	#
	# 打印处理后的数据
	print(f'总地址数：{len(address_transactions.keys())}')
	print(f'in地址数：{len(in_transactions_dict.keys())}')
	print(f'out地址数：{len(out_transactions_dict.keys())}')
	print(f'both地址数：{len(both_transactions_dict.keys())}')

	# print(address_transactions)
	# print(in_transactions_dict)
	# print(out_transactions_dict)
	# print(both_transactions_dict)

	return_dict = {
		'address_transactions': address_transactions,
		'address_list': address_list,
		'in_transactions': in_transactions_dict,
		'in_address_list': in_address_list,
		'out_transactions': out_transactions_dict,
		'out_address_list': out_address_list,
		'both_transactions': both_transactions_dict,
		'both_address_list': both_address_list
	}

	return return_dict


if __name__ == '__main__':
	read_scam_account()
	read_normal_account()
