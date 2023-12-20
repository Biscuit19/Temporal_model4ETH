# -*- codeing = utf-8 -*-
import pickle
from pprint import pprint
import random

import torch
from tqdm import tqdm



def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		accounts_dict = pickle.load(file)
	return accounts_dict


# tran[0]~tran[6]:
# amount, block_timestamp,direction, timewindow, from_address, to_address,tnx_hash

# 计算Time_Diff_Between_first_and_last(Mins)
def calculate_time_diff(transactions):
	first_transaction_time = transactions[0][1]
	last_transaction_time = transactions[-1][1]
	time_diff_minutes = (last_transaction_time - first_transaction_time) / 60
	return time_diff_minutes


# 计算Min_Val_Received
def calculate_min_val_received(transactions):
	val_received = [trans[0] for trans in transactions]
	if val_received:
		min_val_received = min(val_received)
		return min_val_received
	# 待处理，如果没有入交易，是否设置为0
	else:
		return 0


# 计算Min_Val_Sent
def calculate_min_val_sent(transactions):
	val_sent = [trans[0] for trans in transactions]
	if val_sent:
		min_val_sent = min(val_sent)
		return min_val_sent
	else:
		return 0


# 计算Total_Ether_Balance
def calculate_total_ether_balance(in_tran, out_tran):
	in_balance = sum(trans[0] for trans in in_tran)
	out_balance = sum(trans[0] for trans in out_tran)
	total_balance = in_balance - out_balance
	return total_balance


# 计算Avg_Val_Received
def calculate_avg_val_received(transactions):
	received_values = [trans[0] for trans in transactions]
	if received_values:
		avg_val_received = sum(received_values) / len(received_values)
		return avg_val_received
	else:
		return 0


# 计算Avg_Val_Sent
def calculate_avg_val_sent(transactions):
	sent_values = [trans[0] for trans in transactions]
	if sent_values:
		avg_val_sent = sum(sent_values) / len(sent_values)
		return avg_val_sent
	else:
		return 0


# 计算Avg_time_Between_received_tnx
def calculate_avg_time_between_received_transactions(transactions):
	if transactions:
		time_diffs = [transactions[i + 1][1] - trans[1] for i, trans in enumerate(transactions[:-1])]
		if time_diffs:
			avg_time = sum(time_diffs) / len(time_diffs) / 60
			return avg_time

	# 待处理，时间间隔0可能会代表时间间隔很小，但账户实际没有该性质
	return 0


# 计算Avg_time_Between_sent_tnx
def calculate_avg_time_between_sent_transactions(transactions):
	if transactions:
		time_diffs = [transactions[i + 1][1] - trans[1] for i, trans in enumerate(transactions[:-1])]
		if time_diffs:
			avg_time = sum(time_diffs) / len(time_diffs) / 60
			return avg_time

	# 待处理，时间间隔0可能会代表时间间隔很小，但账户实际没有该性质
	return 0


def calculate_max_val_received(transactions):
	if not transactions:
		return 0
	return max(trans[0] for trans in transactions)


# 计算Unique_Received_From_Addresses
def calculate_unique_received_addresses(transactions):
	if not transactions:
		return 0
	unique_received_addresses = set([trans[4] for trans in transactions])
	return len(unique_received_addresses)


# 计算Unique_Sent_To_Addresses20
def calculate_unique_sent_addresses(transactions):
	if not transactions:
		return 0
	unique_sent_addresses = set([trans[5] for trans in transactions])
	return len(unique_sent_addresses)


# 计算Total_Ether_Sent
def calculate_total_ether_sent(transactions):
	if not transactions:
		return 0
	total_ether_sent = sum([trans[0] for trans in transactions])
	return (total_ether_sent)


# 计算Total_Ether_Received
def calculate_total_ether_received(transactions):
	if not transactions:
		return 0
	total_ether_received = sum([trans[0] for trans in transactions])
	return total_ether_received


# 计算max_Time_Between_Rec_Tnx
def calculate_max_time_between_received_transactions(transactions):
	if transactions:
		time_diffs = [transactions[i + 1][1] - trans[1] for i, trans in enumerate(transactions[:-1])]
		if time_diffs:
			max_time = max(time_diffs) / 60
			return max_time
	return 0


# 计算ratioRecTotal
def calculate_ratio_received_total(received_transactions, total_transactions):
	if total_transactions > 0:
		ratio = received_transactions / total_transactions
		return ratio
	return 0

min_timestamp = 1498251906
max_timestamp = 1646092669
day_span = 60 * 60 * 24
day_size = 7
week_size = 8
# amount, block_timestamp, -1, timewindow, from_address, to_address, trans_hash
def time_window_generate(all_trans):

	# 时间窗口跨度
	time_window_span = day_span * day_size * week_size
	# 时间窗口数量
	time_window_num = int((max_timestamp - min_timestamp) / time_window_span)
	# print(f'0 - {time_window_num} time-windows totally')
	# 计算时间窗口值
	for tran in all_trans:
		# 为什么all_trans值改变了,out_trans和in_trans也会改变，这是因为它们都指向同一个列表。
		# 时间窗口从0开始
		tran[3] = int((tran[1] - min_timestamp) / time_window_span)

	# [0.05, 1611664539, -1, 93],
	# create window-num empty sub lists
	windows_trans = [[] for _ in range(time_window_num + 1)]
	for tran in all_trans:
		windows_trans[tran[3]].append(tran)

	return windows_trans


from AutoEncoder_ConvLSTM import AutoEncoder_ConvLSTM

# 全局变量，用于保存模型
global AECL_model

# 加载模型并保存在全局变量中
def load_model():
    global AECL_model
    AECL_model = AutoEncoder_ConvLSTM(input_channels=4, hidden_channels=64, kernel_size=(301, 1), num_layers=1,
                                      batch_first=True,
                                      bias=True, return_all_layers=False)
    AECL_model.load_state_dict(torch.load('AECL_model.pth'))
    print(f'Load AECL_model.pth')


def Conv_encode_pool(batch):
	global AECL_model
	output, h_c = AECL_model.encode(batch)
	h=h_c[0]
	encode_tensor=h[0]
	# print("output_size:",encode_tensor.size())

	# 对张量进行平均池化
	pooled_tensor = torch.mean(encode_tensor, dim=1)

	# 将张量形状变换为(64)/
	pooled_tensor = pooled_tensor.squeeze(-1)

	# 将张量转换为对应维度的列表
	pooled_list = pooled_tensor.tolist()

	return pooled_list


def create_hidden_dict(output_list,feature_index):
	# 使用字典推导式一次性创建新字典
	hidden_dict = {f'hidden_{i + 1}': value for i, value in enumerate(output_list)}
	feature_index.update(hidden_dict)
	return

from Data_Restruct import data_padding,list2tensor,data_normalize

def temporal_aggregation_extract(feature_index):
	all_trans= [trans[:-3] for trans in feature_index['all_trans']]

	windowed_trans=time_window_generate(all_trans)

	padding_trans = data_padding([windowed_trans])

	batch = list2tensor(padding_trans)

	batch = data_normalize(batch)

	pooled_list=Conv_encode_pool(batch)

	create_hidden_dict(pooled_list,feature_index)

	return

def feature_generate(pkl_file):
	print(f'feature_engineering {pkl_file}...')
	accounts_dict = read_pkl(pkl_file)
	load_model()
	for account, feature in tqdm(accounts_dict.items(), desc="Feature generating"):
		feature['Time_Diff_Between_first_and_last(Mins)'] = calculate_time_diff(feature['all_trans'])
		feature['Min_Val_Received'] = calculate_min_val_received(feature['in_trans'])
		feature['Min_Val_Sent'] = calculate_min_val_sent(feature['out_trans'])
		feature['Total_Ether_Balance'] = calculate_total_ether_balance(feature['in_trans'], feature['out_trans'])
		feature['Avg_Val_Received'] = calculate_avg_val_received(feature['in_trans'])
		feature['Avg_Val_Sent'] = calculate_avg_val_sent(feature['out_trans'])
		feature['Avg_time_Between_received_tnx'] = calculate_avg_time_between_received_transactions(feature['in_trans'])
		feature['Avg_time_Between_sent_tnx'] = calculate_avg_time_between_sent_transactions(feature['out_trans'])
		feature['Max_Val_Received'] = calculate_max_val_received(feature['in_trans'])
		feature['Unique_Received_From_Addresses'] = calculate_unique_received_addresses(feature['in_trans'])
		feature['Unique_Sent_To_Addresses'] = calculate_unique_sent_addresses(feature['out_trans'])
		feature['Total_Ether_Sent'] = calculate_total_ether_sent(feature['out_trans'])
		feature['Total_Ether_Received'] = calculate_total_ether_received(feature['in_trans'])
		feature['Sent_tnx'] = feature['out_cnt']
		feature['Received_tnx'] = feature['in_cnt']
		feature['max_Time_Between_RecTnx'] = calculate_max_time_between_received_transactions(feature['in_trans'])
		feature['ratio_Rec_Total'] = calculate_ratio_received_total(feature['Received_tnx'], feature['all_cnt'])
		temporal_aggregation_extract(feature)

	return accounts_dict




def feature_engineer():
	account_file = 'all_account_data.pkl'
	accounts_dict = feature_generate(account_file)

	pprint(accounts_dict['0x634cdd7ebeccccf517cd8f4eca959474bc1b58cc'])

	dump_file = './all_account_featured_data_GAT.pkl'

	with open(dump_file, 'wb') as f:
		pickle.dump(accounts_dict, f)

	print(f'dumping in {dump_file}')




def feature_view():
	account_file = 'all_account_data.pkl'
	accounts_dict=read_pkl(account_file)

	print_dict = {key: accounts_dict[key] for key in list(accounts_dict.keys())[0:2]}
	pprint(print_dict, indent=4, depth=4)


if __name__ == '__main__':

	# accounts_dict={'a':{'all_trans':[[0.05, 1611664539, -1, 93,'','',''],[0.05, 1611664539, -1, 93,'','','']]}}
	# for account, feature in (accounts_dict.items()):
	# 	temporal_aggregation_extract(feature)

	feature_engineer()
	# feature_view()
