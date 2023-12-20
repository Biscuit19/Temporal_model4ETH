# -*- codeing = utf-8 -*-
# 1.生成时间窗口值 2.合并正负样本集

import pickle
import random
from datetime import datetime

import beeprint


def time_window_generate(acct_dict):
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
	week_size = 2
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

	return acct_dict

# 增删某些特征属性
def feature_edit(acct_dict):
	del_att=['in_trans','out_trans']
	for acct, attributes in acct_dict.items():
		for att in del_att:
			del attributes[att]

		# print(attributes)
	return acct_dict

def normalize():

	return

def feature_engineer(pkl_file):
	with open(pkl_file, 'rb') as f:
		acct_dict = pickle.load(f)

	acct_dict = time_window_generate(acct_dict)

	# dump到pkl文件中
	dump_file = pkl_file.replace('../Dataset/', './').replace('data', 'featured_data')
	print(f'dumping in {dump_file}')
	with open(dump_file, 'wb') as f:
		pickle.dump(acct_dict, f)

	# beeprint.pp(acct_dict, sort_keys=False)

def merge_pkl_files(output_file):
	try:
		phs_pkl='./phish_account_featured_data.pkl'
		nor_pkl='./normal_account_featured_data.pkl'

		with open(phs_pkl, 'rb') as file1:
			phs_data = pickle.load(file1)

		with open(nor_pkl,'rb') as file2:
			nor_data = pickle.load(file2)

		# 随机抽样正常样本
		# ./phish_account_data.pkl have 2875 accounts
		nor_account_num=2875*10

		keys = list(nor_data.keys())
		# 随机抽取 nor_account_num 个键
		sampled_keys = random.sample(keys, nor_account_num)
		# 构建新字典，包含随机抽取的键值对
		sampled_nor_data = {key: nor_data[key] for key in sampled_keys}


		# 合并两个字典
		merged_data = {**phs_data, **sampled_nor_data}

		# 保存合并后的字典为新的pkl文件
		with open(output_file, 'wb') as merged_file:
			pickle.dump(merged_data, merged_file)

		print(f"合并完成并保存到 {output_file}")
	except FileNotFoundError:
		print("文件未找到，请检查文件路径。")
	except Exception as e:
		print(f"发生错误：{str(e)}")


if __name__ == '__main__':
	phs_pkl_file = '../Dataset/phish_account_data.pkl'
	nor_pkl_file = '../Dataset/normal_account_data.pkl'


	feature_engineer(phs_pkl_file)
	feature_engineer(nor_pkl_file)



	merge_pkl_files('./all_account_featured_data.pkl')


