import random
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

import dataProcess


Normalize_Set=True
# 筛选交易数至少为least_tran_num的账户
least_tran_num=5
Time_diff_Set=True
# 设置tensor的输出格式
torch.set_printoptions(precision=2, sci_mode=False)
# 序列截断长度 (每条交易序列的交易数)
seq_set_len = 10

# 定义LSTM模型
class LSTM_temporal_Extractor(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, linner_output_dim):
		super(LSTM_temporal_Extractor, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		# 全连接层的输入维度是lstm的隐藏状态维度hidden_size，输出维度是要分类的维度(需要的维度)num_classes
		self.fc = nn.Linear(hidden_size, linner_output_dim)

	# x是输入数据，即(batch_size, sequence_length, input_size)（batch_first=true）的张量
	def forward(self, x):
		# 自定义全零张量作为初始状态（初始隐藏状态和细胞状态）`x

		# out是输出值，不是hc
		output, (hn, cn) = self.lstm(x)
		# 使用最后一个时刻的隐藏状态作为序列的表示
		# 原始hn形状：(num_layers,batch_size,hidden_size)
		# print(f'原始：{hn}')
		hn = self.fc(hn)
		# print(f'fc后：{hn}')

		# 全连接层将每个batch的最后隐藏状态转化为了num_classes维度，进行了非线性变换
		# 经过全连接层形状：(num_layers,batch_size,linner_dim)

		return hn


def Normalize_data(data: np):
	# 对时间序列数据进行归一化处理，使得每一列的数据均值为0，方差为1。这样做可以使得不同列之间的数据尺度相近，有助于模型训练和优化过程
	# data -= data.mean(axis=0)
	# data /= data.std(axis=0)

	std_deviation = data.std(axis=0)

	# 查找标准差为0的列
	cols_with_zero_std = np.where(std_deviation == 0)[0]

	# 然后进行归一化处理
	data -= data.mean(axis=0)
	data /= data.std(axis=0)

	# 对标准差为0的列进行处理，例如替换为0或其他合理值
	# for col in cols_with_zero_std:
	# 	data[:, col] = 0


# 不能	data = (data - data.mean(axis=0)) / data.std(axis=0) 因为这样返回的data是一个局部变量，如果想在原始数组上操作就不能有局部变量


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
	print(f'提取出{len(add_tran_seq)}个恶意账户的交易序列')
	return add_tran_seq


def lstm_train(load_num):

	scam_tran_seq = scam_data_load(load_num)
	normal_tran_seq = normal_data_load(load_num)
	# print(scam_tran_seq)
	# print(normal_tran_seq)

	lengths = [len(seq) for seq in scam_tran_seq]

	# 找到最长的序列长度
	max_scam_seqlen = max(lengths)
	print(max_scam_seqlen)

	# 返回值是张量，其中包含load_num个序列（对应某个地址的交易序列），
	# 地址的每个序列都被填充到了所有地址里最长的序列长度，序列里的每条数据有两个属性：时间差和交易额度
	scam_tran_seq = pad_sequence(scam_tran_seq, batch_first=True, padding_value=0)
	normal_tran_seq = pad_sequence(normal_tran_seq, batch_first=True, padding_value=0)

	# 截断张量的第二列，截断成max_scam_seqlen的长度
	scam_tran_seq = scam_tran_seq[:, :seq_set_len]
	normal_tran_seq = normal_tran_seq[:, :seq_set_len]

	# 给恶意账户交易信息加标签值为 1
	scam_1_label = torch.ones(scam_tran_seq.size()[:-1] + (1,))
	scam_tran_seq = torch.cat([scam_tran_seq, scam_1_label], dim=2)
	print(scam_tran_seq)

	# print(scam_tran_seq)
	# scam_tran_seq = torch.cat([scam_tran_seq, scam_1_label], dim=2)

	# 给正常账户交易信息加标签值为 0
	normal_0_label = torch.zeros(normal_tran_seq.size()[:-1] + (1,))
	normal_tran_seq = torch.cat([normal_tran_seq, normal_0_label], dim=2)

	# 最终的数据集
	combine_tran_seq = torch.cat([scam_tran_seq, normal_tran_seq])
	print('-------最终带标签训练集---------')
	print(combine_tran_seq)

	print(scam_tran_seq.size())
	print(normal_tran_seq.size())
	print(combine_tran_seq.size())

	# 假设有一个输入序列的维度为 (batch_size, sequence_length, input_size)
	batch_size = combine_tran_seq.size()[0]
	sequence_length = combine_tran_seq.size()[1]
	print(f'batch size:{batch_size}\n'
		  f'seq_len:{sequence_length}')
	# 数据维度
	input_size = 2

	hidden_size = 8
	num_layers = 1
	# 全连接层的输出维度
	linner_output_dim = 1
	# 创建模型
	lstm_model = LSTM_temporal_Extractor(input_size, hidden_size, num_layers, linner_output_dim)

	# 定义损失函数和优化器
	criterion = nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失函数
	optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)


	# 提取训练集
	train_seq = combine_tran_seq[:, :, :2]
	print('-------最终无标签训练集---------')
	print(train_seq)
	# print(train_seq.size())

	# 提取标签集（目标值）
	label_target = combine_tran_seq[:, :, 2]  # ([20, 84])
	# label_target=label_target.unsqueeze(1)
	# 提取每行的第一个列的元素值，构成一个新张量
	label_target = label_target[:, 0]
	# 重构为(num_layers, batch_size, linner_output_dim)的张量形状
	label_target = label_target.reshape(num_layers, batch_size, linner_output_dim)

	# print(label_target)

	# 训练模型
	# 训练次数
	epochs_num = 10
	for epoch in range(epochs_num):
		# 经过全连接层形状：(num_layers,batch_size,linner_dim)
		output = lstm_model(train_seq)
		# print(f'output:{output},')
		loss = criterion(output, label_target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 2 == 0:
			# print(f'output:{output}')
			print(f"Epoch [{epoch + 1}/{epochs_num}], Loss: {loss.item():.4f}")

		path = "./lstm_1.pth"  # 替换为你想要保存模型的文件路径
		torch.save(lstm_model.state_dict(), path)




if __name__ == '__main__':
	# 是否开启标准化
	Normalize_Set = True
	# 筛选交易数至少为least_tran_num的账户
	least_tran_num = 5
	Time_diff_Set=True

	seq_set_len = 20
	# 提取的账户数量
	load_accounts_num = 40


	lstm_train(load_accounts_num)
