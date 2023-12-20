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


# min_timestamp,max_timestamp = 1498251906,1646092669 # bert数据集时间跨度
min_timestamp,max_timestamp = 1438923669,1547885786 # zsu数据集时间跨度
day_span = 60 * 60 * 24
day_size = 7
week_size = 8
# 时间窗口跨度
time_window_span = day_span * day_size * week_size
# 时间窗口数量
time_window_num = int((max_timestamp - min_timestamp) / time_window_span)
print(f'time_window_num:{time_window_num+1}')


# amount, block_timestamp, -1, timewindow, from_address, to_address, trans_hash
def time_window_generate(all_trans):
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


# 加载模型并保存在全局变量中
def load_model():
	AECL_model = AutoEncoder_ConvLSTM(input_channels=4, hidden_channels=64, kernel_size=(301, 1), num_layers=1,
									  batch_first=True,
									  bias=True, return_all_layers=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	AECL_model.to(device)
	AECL_model.load_state_dict(torch.load('AECL_model.pth', map_location=device))
	print(f'Load AECL_model.pth')
	return AECL_model


def Conv_encode_pool(batch, model):

	output, h_c = model.encode(batch)
	h = h_c[0]
	encode_tensor = h[0]
	# print("output_size:",encode_tensor.size())

	# 对张量进行平均池化
	pooled_tensor = torch.mean(encode_tensor, dim=1)

	# 将张量形状变换为(64)/
	pooled_tensor = pooled_tensor.squeeze(-1)

	# 将张量转换为对应维度的列表
	pooled_list = pooled_tensor.tolist()

	return pooled_list


def create_hidden_dict(output_list, feature_index):
	# 使用字典推导式一次性创建新字典
	hidden_dict = {f'hidden_{i + 1}': value for i, value in enumerate(output_list)}
	feature_index.update(hidden_dict)
	return


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
		user_info["Total_Ether_Sent"],
		user_info["Total_Ether_Received"],
		user_info["Sent_tnx"],
		user_info["Received_tnx"],
		user_info["max_Time_Between_RecTnx"],
		user_info["ratio_Rec_Total"]
	]

	for key in user_info:
		if key.startswith("hidden_"):
			node_features.append(user_info[key])

	return node_features


from Data_Restruct import data_padding, list2tensor, data_normalize


def temporal_aggregation_extract(feature_index, model):
	all_trans = [trans[:-3] for trans in feature_index['all_trans']]

	windowed_trans = time_window_generate(all_trans)

	padding_trans = data_padding([windowed_trans])

	batch = list2tensor(padding_trans)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batch = data_normalize(batch).to(device)
	pooled_list = Conv_encode_pool(batch, model)

	create_hidden_dict(pooled_list, feature_index)

	return


def feature_generate(feature, model):
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
	temporal_aggregation_extract(feature, model)
	node_features = create_node_features(feature)

	return torch.tensor(node_features, dtype=torch.float32)


if __name__ == '__main__':
	accounts_dict = {
		'0x0000000000000000000000000000000000000000': {'account_address': '0x0000000000000000000000000000000000000000',
													   'out_trans': [], 'in_trans': [
				[1.0, 1515713137, 1, 0, '0xe47494379c1d48ee73454c251a6395fdd4f9eb43',
				 '0x0000000000000000000000000000000000000000',
				 '0x29e05339f1ee001490561c7721a7c82f9a281ea2b6aa179727002b5eb21f13a1'],
				[0.001, 1571204976, 1, 0, '0x00509732e3679d8a88565d38530970ca8f311adb',
				 '0x0000000000000000000000000000000000000000',
				 '0x51343bb466cc8e5274551fc73035d15d1a822d8243e3953685c4db029807f10a'],
				[0.008, 1574376905, 1, 0, '0xd4a45d8055ac000a4bbd7f3c6007d491d7f3c529',
				 '0x0000000000000000000000000000000000000000',
				 '0x0322ea8402138def81cb836a8c4a8047489436f427e2092a5fb0b0035424c050'],
				[0.00089516, 1598493568, 1, 0, '0x9da660fd34cfae0414dfcfb3bfbb1bbab9555416',
				 '0x0000000000000000000000000000000000000000',
				 '0x75f630c13441de0f52790418e819786a287d810ca64618e682bceaa3fb10dfb6'],
				[1e-08, 1598845801, 1, 0, '0x2f84f3e5f9aed03c8e3d88ffd5f633d9408da8b9',
				 '0x0000000000000000000000000000000000000000',
				 '0x0d3f3f129ff1871b651415c6b2f571f058f325796346cadf7743fe2e8793596e'],
				[1e-09, 1628239913, 1, 0, '0x440f1256224c351d30d7b723c99c99b38af4ff5f',
				 '0x0000000000000000000000000000000000000000',
				 '0xe5a0f594025b060492b4d7944e3347563ffe49bff1d21a5765a4231ea96c6920'],
				[0.0379230215339038, 1628309782, 1, 0, '0x56683ac80d019c9450413d692d35da145a08d682',
				 '0x0000000000000000000000000000000000000000',
				 '0xdd6e75604885c156c5c8a05ebb28fcd7eff8862de49c03c19df18b8ae698c370'],
				[0.032, 1630018043, 1, 0, '0x01a45f7fbf7453092e49b6db137568f9cfab8ead',
				 '0x0000000000000000000000000000000000000000',
				 '0x8fe5ae5a37565efb181e4dc914fb99a14b56ee0862a9022c94d872e0dcef06f4'],
				[0.24708309, 1631079459, 1, 0, '0x2a262ce2ec79f1904dded0c7068240fa77103bf1',
				 '0x0000000000000000000000000000000000000000',
				 '0x212a6026b57e39fb9376577e1a3543358f1483cf8b9b548da62b07e6d64b43ef'],
				[0.10069675, 1631079692, 1, 0, '0x4b339e6db6132cdeda1f5b985a96a73161b4d2a8',
				 '0x0000000000000000000000000000000000000000',
				 '0x9ed51e31f27fc26e2bff31ec194014a0978af0b98a6f87f46d75037f797d353b'],
				[0.087732, 1631079732, 1, 0, '0x1ea6b2566a8ddd092bba3ce3fa25d7373aa85d64',
				 '0x0000000000000000000000000000000000000000',
				 '0x13c0dfcfa929cfede98a2613010b3577d52bd609b5681077ddc95a9e890eff4d'],
				[0.377963, 1631079842, 1, 0, '0x6307796a4b2c83f296769fa28705a252ec4a1c1a',
				 '0x0000000000000000000000000000000000000000',
				 '0xe59c390301cff6691f972f8600c724d090c0548a0291cb822770d4b6de365746'],
				[0.137468, 1631079870, 1, 0, '0x5b0bc057d5823dd8fc3070cde5b3e4b5c88dee7d',
				 '0x0000000000000000000000000000000000000000',
				 '0x2c6f9288ffaa76951976c215ea989b0219445f83a58f586e41dff02e5e363e21'],
				[0.010141985710636, 1631079945, 1, 0, '0x410d7eec59c59f11b490ab4aad98a0b2954bc719',
				 '0x0000000000000000000000000000000000000000',
				 '0xe507941787976782cd908c3e08b8f55d5d2dad34e322211cf7eb328432ff9f69'],
				[0.01203217, 1631080197, 1, 0, '0x6f969ef595e76e72361ead46d66749db70aaad7b',
				 '0x0000000000000000000000000000000000000000',
				 '0xd7cdfb0c8d364a7b94760e7177195324dff724941163c91ff62a77aa93197793'],
				[0.467958, 1631080301, 1, 0, '0x3f881c681070115efbfb2157427b3980fb06c8e4',
				 '0x0000000000000000000000000000000000000000',
				 '0xe3b2626016464d6f074428e308e8dec765ebf699cef0b139939d2fc31e50d973'],
				[0.08047776, 1631081039, 1, 0, '0x72e953d5fa37df21fe5910d483fa7b8ecffaba8a',
				 '0x0000000000000000000000000000000000000000',
				 '0x0a622a4f2e09c4ddb93435556224a8838f344982066addfc1c792e0cb482f08a'],
				[0.299860315, 1631081201, 1, 0, '0x985d078976ac7b4b8ce2abe8b6b2cd7c2371130d',
				 '0x0000000000000000000000000000000000000000',
				 '0xcf197f0bbc694a6bb30ea171396352e5711ff9f3addad16b45b5241292cf6cfb'],
				[0.11975143, 1631082283, 1, 0, '0xbef0b6f5f9130b20ec2c0bdd1ecb4fe5901960e2',
				 '0x0000000000000000000000000000000000000000',
				 '0x72c2c99cccf6c16435acda9cd0281ac1532ca964302615fb8e3786cbf7726c67'],
				[0.139312, 1631082517, 1, 0, '0x4548425b40ba03d278154770c0fba223254ff869',
				 '0x0000000000000000000000000000000000000000',
				 '0xa10efebd53c1b2786076f0ab7d418560e9329ddf24e6c3c7dcac903538f0f7f5'],
				[0.171812, 1631082517, 1, 0, '0xcca36afd523fe30fa84e7cdc11ee16662a88cc82',
				 '0x0000000000000000000000000000000000000000',
				 '0x270cb1ca9b63592c60afe3f6e5f88bb8c23c34e2a27307607c3f0db192119e9e'],
				[0.6650715574961772, 1631082535, 1, 0, '0xd39f76f1b4ff3322399c66f332484317f7971af8',
				 '0x0000000000000000000000000000000000000000',
				 '0x56359c23184d1cc4b9b3ad27adccc7e57f3c2fd633d0ee9f0c2de8271d467a0d'],
				[0.355411958, 1631082608, 1, 0, '0xa96201f55fce493aa99e9ed383438d05892d1ec4',
				 '0x0000000000000000000000000000000000000000',
				 '0xb9738a7ff727ec0fdf03c04f3a3f8d81d42f710997271139d4b55720e6ecd986'],
				[6.9e-17, 1638937748, 1, 0, '0x172c015dab2ab790f5c639cdb996a29349a72086',
				 '0x0000000000000000000000000000000000000000',
				 '0x0000009ba56d7bd3197ef2c43e5941f13fdb310e7d9df53167156666fd125816'],
				[6.9e-17, 1639002858, 1, 0, '0x172c015dab2ab790f5c639cdb996a29349a72086',
				 '0x0000000000000000000000000000000000000000',
				 '0x000000073d55e593248cfce0416ffa7071314fb937f9b471ca0057e804bcb165'],
				[4.2e-16, 1639176849, 1, 0, '0x172c015dab2ab790f5c639cdb996a29349a72086',
				 '0x0000000000000000000000000000000000000000',
				 '0x00000035eb357b55c2b9e6c35f32e9e93741bc87ca3200d9f3eb9a4f89421867']], 'all_trans': [
				[1.0, 1515713137, 1, 0, '0xe47494379c1d48ee73454c251a6395fdd4f9eb43',
				 '0x0000000000000000000000000000000000000000',
				 '0x29e05339f1ee001490561c7721a7c82f9a281ea2b6aa179727002b5eb21f13a1'],
				[0.001, 1571204976, 1, 0, '0x00509732e3679d8a88565d38530970ca8f311adb',
				 '0x0000000000000000000000000000000000000000',
				 '0x51343bb466cc8e5274551fc73035d15d1a822d8243e3953685c4db029807f10a'],
				[0.008, 1574376905, 1, 0, '0xd4a45d8055ac000a4bbd7f3c6007d491d7f3c529',
				 '0x0000000000000000000000000000000000000000',
				 '0x0322ea8402138def81cb836a8c4a8047489436f427e2092a5fb0b0035424c050'],
				[0.00089516, 1598493568, 1, 0, '0x9da660fd34cfae0414dfcfb3bfbb1bbab9555416',
				 '0x0000000000000000000000000000000000000000',
				 '0x75f630c13441de0f52790418e819786a287d810ca64618e682bceaa3fb10dfb6'],
				[1e-08, 1598845801, 1, 0, '0x2f84f3e5f9aed03c8e3d88ffd5f633d9408da8b9',
				 '0x0000000000000000000000000000000000000000',
				 '0x0d3f3f129ff1871b651415c6b2f571f058f325796346cadf7743fe2e8793596e'],
				[1e-09, 1628239913, 1, 0, '0x440f1256224c351d30d7b723c99c99b38af4ff5f',
				 '0x0000000000000000000000000000000000000000',
				 '0xe5a0f594025b060492b4d7944e3347563ffe49bff1d21a5765a4231ea96c6920'],
				[0.0379230215339038, 1628309782, 1, 0, '0x56683ac80d019c9450413d692d35da145a08d682',
				 '0x0000000000000000000000000000000000000000',
				 '0xdd6e75604885c156c5c8a05ebb28fcd7eff8862de49c03c19df18b8ae698c370'],
				[0.032, 1630018043, 1, 0, '0x01a45f7fbf7453092e49b6db137568f9cfab8ead',
				 '0x0000000000000000000000000000000000000000',
				 '0x8fe5ae5a37565efb181e4dc914fb99a14b56ee0862a9022c94d872e0dcef06f4'],
				[0.24708309, 1631079459, 1, 0, '0x2a262ce2ec79f1904dded0c7068240fa77103bf1',
				 '0x0000000000000000000000000000000000000000',
				 '0x212a6026b57e39fb9376577e1a3543358f1483cf8b9b548da62b07e6d64b43ef'],
				[0.10069675, 1631079692, 1, 0, '0x4b339e6db6132cdeda1f5b985a96a73161b4d2a8',
				 '0x0000000000000000000000000000000000000000',
				 '0x9ed51e31f27fc26e2bff31ec194014a0978af0b98a6f87f46d75037f797d353b'],
				[0.087732, 1631079732, 1, 0, '0x1ea6b2566a8ddd092bba3ce3fa25d7373aa85d64',
				 '0x0000000000000000000000000000000000000000',
				 '0x13c0dfcfa929cfede98a2613010b3577d52bd609b5681077ddc95a9e890eff4d'],
				[0.377963, 1631079842, 1, 0, '0x6307796a4b2c83f296769fa28705a252ec4a1c1a',
				 '0x0000000000000000000000000000000000000000',
				 '0xe59c390301cff6691f972f8600c724d090c0548a0291cb822770d4b6de365746'],
				[0.137468, 1631079870, 1, 0, '0x5b0bc057d5823dd8fc3070cde5b3e4b5c88dee7d',
				 '0x0000000000000000000000000000000000000000',
				 '0x2c6f9288ffaa76951976c215ea989b0219445f83a58f586e41dff02e5e363e21'],
				[0.010141985710636, 1631079945, 1, 0, '0x410d7eec59c59f11b490ab4aad98a0b2954bc719',
				 '0x0000000000000000000000000000000000000000',
				 '0xe507941787976782cd908c3e08b8f55d5d2dad34e322211cf7eb328432ff9f69'],
				[0.01203217, 1631080197, 1, 0, '0x6f969ef595e76e72361ead46d66749db70aaad7b',
				 '0x0000000000000000000000000000000000000000',
				 '0xd7cdfb0c8d364a7b94760e7177195324dff724941163c91ff62a77aa93197793'],
				[0.467958, 1631080301, 1, 0, '0x3f881c681070115efbfb2157427b3980fb06c8e4',
				 '0x0000000000000000000000000000000000000000',
				 '0xe3b2626016464d6f074428e308e8dec765ebf699cef0b139939d2fc31e50d973'],
				[0.08047776, 1631081039, 1, 0, '0x72e953d5fa37df21fe5910d483fa7b8ecffaba8a',
				 '0x0000000000000000000000000000000000000000',
				 '0x0a622a4f2e09c4ddb93435556224a8838f344982066addfc1c792e0cb482f08a'],
				[0.299860315, 1631081201, 1, 0, '0x985d078976ac7b4b8ce2abe8b6b2cd7c2371130d',
				 '0x0000000000000000000000000000000000000000',
				 '0xcf197f0bbc694a6bb30ea171396352e5711ff9f3addad16b45b5241292cf6cfb'],
				[0.11975143, 1631082283, 1, 0, '0xbef0b6f5f9130b20ec2c0bdd1ecb4fe5901960e2',
				 '0x0000000000000000000000000000000000000000',
				 '0x72c2c99cccf6c16435acda9cd0281ac1532ca964302615fb8e3786cbf7726c67'],
				[0.139312, 1631082517, 1, 0, '0x4548425b40ba03d278154770c0fba223254ff869',
				 '0x0000000000000000000000000000000000000000',
				 '0xa10efebd53c1b2786076f0ab7d418560e9329ddf24e6c3c7dcac903538f0f7f5'],
				[0.171812, 1631082517, 1, 0, '0xcca36afd523fe30fa84e7cdc11ee16662a88cc82',
				 '0x0000000000000000000000000000000000000000',
				 '0x270cb1ca9b63592c60afe3f6e5f88bb8c23c34e2a27307607c3f0db192119e9e'],
				[0.6650715574961772, 1631082535, 1, 0, '0xd39f76f1b4ff3322399c66f332484317f7971af8',
				 '0x0000000000000000000000000000000000000000',
				 '0x56359c23184d1cc4b9b3ad27adccc7e57f3c2fd633d0ee9f0c2de8271d467a0d'],
				[0.355411958, 1631082608, 1, 0, '0xa96201f55fce493aa99e9ed383438d05892d1ec4',
				 '0x0000000000000000000000000000000000000000',
				 '0xb9738a7ff727ec0fdf03c04f3a3f8d81d42f710997271139d4b55720e6ecd986'],
				[6.9e-17, 1638937748, 1, 0, '0x172c015dab2ab790f5c639cdb996a29349a72086',
				 '0x0000000000000000000000000000000000000000',
				 '0x0000009ba56d7bd3197ef2c43e5941f13fdb310e7d9df53167156666fd125816'],
				[6.9e-17, 1639002858, 1, 0, '0x172c015dab2ab790f5c639cdb996a29349a72086',
				 '0x0000000000000000000000000000000000000000',
				 '0x000000073d55e593248cfce0416ffa7071314fb937f9b471ca0057e804bcb165'],
				[4.2e-16, 1639176849, 1, 0, '0x172c015dab2ab790f5c639cdb996a29349a72086',
				 '0x0000000000000000000000000000000000000000',
				 '0x00000035eb357b55c2b9e6c35f32e9e93741bc87ca3200d9f3eb9a4f89421867']], 'category': 0, 'out_cnt': 0,
													   'in_cnt': 26, 'all_cnt': 26}}
	model=load_model()
	feature = feature_generate(accounts_dict['0x0000000000000000000000000000000000000000'],model)
	print(feature.size())
