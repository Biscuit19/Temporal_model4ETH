import random
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

import dataProcess

Normalize_Set = True
# 筛选交易数至少为least_tran_num的账户
least_tran_num = 5
Time_diff_Set = True
# 设置tensor的输出格式
torch.set_printoptions(precision=4, sci_mode=False, threshold=10000000)
# 序列截断长度 (每条交易序列的交易数)
seq_set_len = 10


# 定义LSTM模型
class LSTM_Autoencoder(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(LSTM_Autoencoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

	# x是输入数据，即(batch_size, sequence_length, input_size)（batch_first=true）的张量
	def forward(self, x):
		encoded, _ = self.encoder(x)
		decoded, _ = self.decoder(encoded)
		return decoded

	def encode(self, x):
		encoded, _ = self.encoder(x)
		return encoded


def shuffle_tensor(tensor):
	indices = torch.randperm(tensor.size(0))
	shuffled_tensor = tensor[indices]
	return shuffled_tensor


def Normalize_data(data: np):
	# 对时间序列数据进行归一化处理，使得每一列的数据均值为0，方差为1。这样做可以使得不同列之间的数据尺度相近，有助于模型训练和优化过程

	# std_deviation = data.std(axis=0)
	#
	# # 查找标准差为0的列
	# cols_with_zero_std = np.where(std_deviation == 0)[0]
	#
	# 然后进行标准化处理
	epsilon = 1e-8  # 微小的修正值
	data -= data.mean(axis=0)
	data /= (data.std(axis=0)+epsilon)

	# min_vals = np.min(data, axis=0)
	# max_vals = np.max(data, axis=0)
	# epsilon = 1e-8  # 微小的修正值
	# data -= min_vals
	# data /= (max_vals - min_vals+epsilon)

	return


# [[datetime.datetime(2017, 10, 16, 5, 26, 53), 1.8e+18], [datetime.datetime(2017, 10, 16, 5, 57, 28), 2.86e+19], [datetime.datetime(2017, 10, 16, 5, 59, 57), 3.33e+17]
def calculate_time_diff(transaction_list):
	if not Time_diff_Set:
		for item in transaction_list:
			item[0] = int(item[0].timestamp())
		return

	prev_time = transaction_list[0][0]
	for i in range(len(transaction_list)):

		# 计算时间差
		if i != 0:
			current_time = transaction_list[i][0]
			# 计算本次和上次交易的时间间隔秒数
			time_diff = (current_time - prev_time).total_seconds()

			# 存储本次的时间戳，作为下一次计算的前个时间
			prev_time = transaction_list[i][0]

			transaction_list[i][0] = time_diff
		# print(type(transaction_list[i][0]))
		if i == 0:
			continue

	# 第一次交易的时间差设定为0
	transaction_list[0][0] = 0.0

	# 直接去除第一次交易,去除第一行
	del transaction_list[0]

	return transaction_list


def normal_data_load(address_num, tran_direct='out'):
	total_dict = dataProcess.read_normal_account()

	# 分类后的交易信息与对应的顺序地址
	# address_transactions = total_dict['address_transactions']
	# address_list = total_dict['address_list']
	# in_transactions = total_dict['in_transactions']
	# in_address_list = total_dict['in_address_list']

	# transactions = total_dict['out_transactions']
	# address_list = total_dict['out_address_list']

	transactions = total_dict['both_transactions']
	address_list = total_dict['both_address_list']

	add_tran_seq = []
	i = 0
	while len(add_tran_seq) < address_num and i < len(address_list):
		tran_seq = transactions[address_list[i]][tran_direct]
		i += 1
		# 筛选交易数至少为n的账户
		if not len(tran_seq) >= least_tran_num:
			continue
		# [[datetime.datetime(2017, 10, 16, 5, 26, 53), 1.8e+18], [datetime.datetime(2017, 10, 16, 5, 57, 28), 2.86e+19], [datetime.datetime(2017, 10, 16, 5, 59, 57), 3.33e+17]
		# print(f'Original seq{tran_seq}')

		calculate_time_diff(tran_seq)

		# 必须先算时间间隔再转np数组
		tran_seq = np.array(tran_seq)

		# print(f'time_diff type{type(tran_seq)} seq{tran_seq}')
		if Normalize_Set:
			Normalize_data(tran_seq)
		# print(f'Normalize seq{tran_seq}')
		tran_seq = torch.tensor(tran_seq, dtype=torch.float32)

		add_tran_seq.append(tran_seq)
	print(f'提取出{len(add_tran_seq)}个正常账户的交易序列')
	return add_tran_seq


# 加载num个恶意地址的交易序列
def scam_data_load(address_num, tran_direct='out'):
	total_dict = dataProcess.read_scam_account()

	# 分类后的交易信息与对应的顺序地址
	# address_transactions = total_dict['address_transactions']
	# address_list = total_dict['address_list']
	# in_transactions = total_dict['in_transactions']
	# in_address_list = total_dict['in_address_list']

	# transactions = total_dict['out_transactions']
	# address_list = total_dict['out_address_list']

	transactions = total_dict['both_transactions']
	address_list = total_dict['both_address_list']

	add_tran_seq = []
	i = 0
	while len(add_tran_seq) < address_num and i < len(address_list):
		tran_seq = transactions[address_list[i]][tran_direct]
		i += 1
		# 筛选交易数至少为n的账户
		if not len(tran_seq) >= least_tran_num:
			continue
		# [[datetime.datetime(2017, 10, 16, 5, 26, 53), 1.8e+18], [datetime.datetime(2017, 10, 16, 5, 57, 28), 2.86e+19], [datetime.datetime(2017, 10, 16, 5, 59, 57), 3.33e+17]
		# print(f'Original seq{tran_seq}')

		calculate_time_diff(tran_seq)

		# 必须先算时间间隔再转np数组
		tran_seq = np.array(tran_seq)

		# print(f'time_diff type{type(tran_seq)} seq{tran_seq}')

		if Normalize_Set:
			Normalize_data(tran_seq)
		# print(f'Normalize seq{tran_seq}')

		tran_seq = torch.tensor(tran_seq, dtype=torch.float32)

		add_tran_seq.append(tran_seq)
	print(f'提取出{len(add_tran_seq)}个交易数至少为{least_tran_num}的恶意账户的交易序列')
	return add_tran_seq


def model_train(load_num):
	scam_tran_seq = scam_data_load(load_num)
	normal_tran_seq = normal_data_load(load_num)
	# print(normal_tran_seq)

	# random.shuffle(scam_tran_seq)

	lengths = [len(seq) for seq in scam_tran_seq]

	# 找到最长的序列长度
	max_scam_seqlen = max(lengths)
	avg_scam_seqlen = int(sum(lengths) / len(lengths))

	print(f'序列平均长度：{avg_scam_seqlen}')

	# 返回值是张量，其中包含load_num个序列（对应某个地址的交易序列），
	# 地址的每个序列都被填充到了所有地址里最长的序列长度，序列里的每条数据有两个属性：时间差和交易额度
	scam_tran_seq = pad_sequence(scam_tran_seq, batch_first=True, padding_value=0)
	normal_tran_seq = pad_sequence(normal_tran_seq, batch_first=True, padding_value=0)

	# # 截断张量的第二列，截断成max_scam_seqlen的长度
	# scam_tran_seq = scam_tran_seq[:, :avg_scam_seqlen]
	scam_tran_seq = scam_tran_seq[:, :seq_set_len]

	normal_tran_seq = normal_tran_seq[:, :seq_set_len]
	#
	# # 最终的数据集
	combine_tran_seq = torch.cat([scam_tran_seq, normal_tran_seq])
	# print('-------最终带标签训练集---------')
	# print(combine_tran_seq)

	# 假设有一个输入序列的维度为 (batch_size, sequence_length, input_size)
	batch_size = scam_tran_seq.size()[0]
	sequence_length = scam_tran_seq.size()[1]
	# 数据维度
	# print(f'batch size:{batch_size}\n'
	# 	  f'seq_len:{sequence_length}')
	input_size = 2

	hidden_size = 8
	num_layers = 1

	# 创建模型
	# 假设你已经定义了 LSTM_Autoencoder 类
	LSTM_encoder = LSTM_Autoencoder(input_size, hidden_size, num_layers)

	# 定义均方差损失函数
	criterion = nn.MSELoss()

	# 优化器
	optimizer = torch.optim.Adam(LSTM_encoder.parameters(), lr=0.001)

	# 加入正常用户样本并打乱
	scam_tran_seq = shuffle_tensor(combine_tran_seq)


	split_idx = int(0.8 * len(scam_tran_seq))  # 80%的数据用于训练，20%用于验证
	train_seq = scam_tran_seq[:split_idx]
	eval_seq = scam_tran_seq[split_idx:]
	print(scam_tran_seq)
	print(scam_tran_seq.size())
	print(f'{train_seq.size()};{eval_seq.size()}')

	# 训练次数
	epochs_num = 5000
	best_validation_loss = float('inf')
	patience_max = 10  # 早停的耐心期
	patience = patience_max  # 当前耐心值

	for epoch in range(epochs_num):
		optimizer.zero_grad()
		output = LSTM_encoder(train_seq)
		loss = criterion(output, train_seq)  # 输入数据和重构数据之间的均方差作为损失
		loss.backward()
		optimizer.step()

		# 计算验证损失
		with torch.no_grad():
			eval_output = LSTM_encoder(eval_seq)
			eval_loss = criterion(eval_output, eval_seq)

		# 打印训练损失和验证损失
		if epoch % 100 == 0:
			print(f'Epoch [{epoch + 1}/{epochs_num}], Train Loss: {loss.item()}, Eval Loss: {eval_loss.item()}')

		# 判断过拟合
		if eval_loss < best_validation_loss:
			best_validation_loss = eval_loss
			patience = patience_max  # 重置耐心期
		else:
			# print(f"Best Loss:{best_validation_loss} >= Eval Loss: {eval_loss.item()}")
			patience -= 1
			if patience == 0:
				print("Early stopping triggered!")
				break

	path = "./lstm_encoder.pth"  # 替换为你想要保存模型的文件路径
	torch.save(LSTM_encoder.state_dict(), path)

	def lstm_eval():
		LSTM_encoder.eval()
		with torch.no_grad():
			eval_output = LSTM_encoder(eval_seq)
			eval_loss = criterion(eval_output, eval_seq)
			print(f"Eval_loss:{eval_loss.item()}")

	lstm_eval()


# 用编码器进行特征提取
def lstm_feature_extract(load_num):
	scam_tran_seq = scam_data_load(load_num)
	lengths = [len(seq) for seq in scam_tran_seq]
	avg_scam_seqlen = int(sum(lengths) / len(lengths))
	print(f'序列平均长度：{avg_scam_seqlen}')
	scam_tran_seq = pad_sequence(scam_tran_seq, batch_first=True, padding_value=0)
	scam_tran_seq = scam_tran_seq[:, :seq_set_len]

	input_size = 2
	hidden_size = 8
	num_layers = 1
	# 提取训练集
	split_idx = int(0.8 * len(scam_tran_seq))  # 80%的数据用于训练，20%用于验证
	eval_seq = scam_tran_seq[split_idx:]
	data = eval_seq
	LSTM_encoder = LSTM_Autoencoder(input_size, hidden_size, num_layers)
	LSTM_encoder.load_state_dict(torch.load('./lstm_encoder.pth'))
	with torch.no_grad():
		data_output = LSTM_encoder(data)
		features = LSTM_encoder.encode(data)
		print(f"features :{features}")
		print(f"data_output :{data_output}")


if __name__ == '__main__':
	# 是否开启标准化
	Normalize_Set = True
	# 筛选交易数至少为least_tran_num的账户
	least_tran_num = 10

	Time_diff_Set = True

	# 设置最长序列长度
	seq_set_len = 25

	# 尽可能提取的账户数量
	load_accounts_num = 10

	# lstm_feature_extract(load_accounts_num)
	model_train(load_accounts_num)
