import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

"""
Github: Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN

The source of training data 
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""


def run_train_gru():
	inp_dim = 3
	out_dim = 1
	batch_size = 12 * 4

	'''load data'''
	data = load_data()
	data_x = data[:-1, :]
	data_y = data[+1:, 0]
	assert data_x.shape[1] == inp_dim

	train_size = int(len(data_x) * 0.75)

	train_x = data_x[:train_size]
	train_y = data_y[:train_size]
	train_x = train_x.reshape((train_size, inp_dim))
	train_y = train_y.reshape((train_size, out_dim))

	'''build model'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = RegGRU(inp_dim, out_dim, mod_dim=12, mid_layers=2).to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

	'''train'''
	var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
	var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

	batch_var_x = list()
	batch_var_y = list()

	for i in range(batch_size):
		j = train_size - i
		batch_var_x.append(var_x[j:])
		batch_var_y.append(var_y[j:])

	from torch.nn.utils.rnn import pad_sequence
	batch_var_x = pad_sequence(batch_var_x)
	batch_var_y = pad_sequence(batch_var_y)

	with torch.no_grad():
		weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
		weights = torch.tensor(weights, dtype=torch.float32, device=device)

	for e in range(256):
		out = net(batch_var_x)

		# loss = criterion(out, batch_var_y)
		loss = (out - batch_var_y) ** 2 * weights
		loss = loss.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if e % 100 == 0:
			print('Epoch: {}, Loss: {:.5f}'.format(e, loss.item()))

	'''eval'''
	net = net.eval()

	test_x = data_x.copy()
	test_x[train_size:, 0] = 0
	test_x = test_x[:, np.newaxis, :]
	test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
	for i in range(train_size, len(data) - 2):
		test_y = net(test_x[:i])
		test_x[i + 1, 0, 0] = test_y[-1]
	pred_y = test_x[1:, 0, 0]
	pred_y = pred_y.cpu().data.numpy()

	diff_y = pred_y[train_size:] - data_y[train_size:-1]
	l1_loss = np.mean(np.abs(diff_y))
	l2_loss = np.mean(diff_y ** 2)
	print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))
	plt.plot(pred_y, 'r', label='pred')
	plt.plot(data_y, 'b', label='real')
	plt.legend(loc='best')
	plt.pause(4)


def run_train_lstm():
	inp_dim = 3  # 输入数据的维度，由【客流量，年份，月份】三组数据构成，因此 inp_dim == 3
	out_dim = 1  # 需要预测的值只有客流量这一个值，因此 out_dim 为 1
	mid_dim = 8
	mid_layers = 1
	batch_size = 12 * 4  # 指定 batch size 大小
	mod_dir = '..'  # 模型保存路径

	# 载入数据 (144,3)
	# [乘客数，年份，月份]
	data = load_data()
	data_x = data[:-1, :]  # 数据集，去掉最后一行
	data_y = data[+1:, 0]  # 目标值数据，取第一列
	# assert就是测试表达式是否满足
	assert data_x.shape[1] == inp_dim  # 确保特征数据的维度与 inp_dim 一致

	train_size = int(len(data_x) * 0.75)  # 划分训练集大小

	train_x = data_x[:train_size]  # 训练集特征数据
	train_y = data_y[:train_size]  # 训练集目标值数据
	train_x = train_x.reshape((train_size, inp_dim))  # 调整训练集特征数据的形状
	train_y = train_y.reshape((train_size, out_dim))  # 调整训练集目标值数据的形状

	'''build model'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
	net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)  # 构建 LSTM 模型
	criterion = nn.MSELoss()  # 定义损失函数为均方误差（Mean Squared Error）
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)  # 使用Adam优化器进行参数优化

	'''train'''
	var_x = torch.tensor(train_x, dtype=torch.float32, device=device)  # 将训练集特征数据转换为张量，并移动到指定设备（GPU/CPU）
	var_y = torch.tensor(train_y, dtype=torch.float32, device=device)  # 将训练集目标值数据转换为张量，并移动到指定设备（GPU/CPU）

	# 生成每一批的特征序列和目标值序列,共需要生成batch size这么多条序列
	batch_var_x = list()
	batch_var_y = list()
	# 此时需要构建用于训练的序列，由于全部的数据仅有144条，
	# 我们设定了batch size为48,则我们需要构建48条序列,而我们现在仅有长度为144的一条序列
	# 为了最大化利用这条序列，要对其进行裁减，生成多条序列，类似于数据增广
	# 由于RNN的特性，需要在不同起始位点进行裁减，此处裁减了j到末尾的数据作为一条序列，共生成了48条序列
	for i in range(batch_size):
		j = batch_size - i
		batch_var_x.append(var_x[j:])
		batch_var_y.append(var_y[j:])
	# 这48条序列的起始裁剪位点都不一样（非常重要），因此在RNN内不会重复训练（非常重要）
	# print(f'weicaijian:{batch_var_x}\n')

	#
	from torch.nn.utils.rnn import pad_sequence
	batch_var_x = pad_sequence(batch_var_x)  # 将序列填充补0，使所有序列长度一致，可以传入网络训练
	batch_var_y = pad_sequence(batch_var_y)
	# print(f'{batch_var_x}\n')

	# 由于LSTM在输入序列长度不足的情况下进行预测可能会失误，因此我为loss设置了权重，输入序列的长度越短，其误差的权重就越小。
	# 这样，不同样本对于损失函数的贡献可以不同，从而在训练过程中调整样本的重要性。
	# with torch.no_grad():是上下文管理器，在这个代码块内部，任何计算都不会对模型的梯度产生影响。
	with torch.no_grad():
		# 生成从 0 到 len(train_y)-1 的整数序列,乘以 (np.e / len(train_y))
		# 通过np.tanh 函数进行处理，得到一个范围在 [-1, 1] 之间的权重向量。
		# 长度与train_y样本数相同，即每个序列对应一个权重值。
		weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
		weights = torch.tensor(weights, dtype=torch.float32, device=device)  # 将权重转换为张量，并移动到指定设备（GPU/CPU）
		print(f'weight: {weights}')


	print("Training Start")
	for e in range(500):
		out = net(batch_var_x)  # 进行前向传播，得到模型的预测输出

		# loss = criterion(out, batch_var_y)
		loss = (out - batch_var_y) ** 2 * weights  # 计算加权均方误差作为损失函数
		loss = loss.mean()  # 计算损失函数的平均值

		optimizer.zero_grad()  # 清空之前计算的梯度
		loss.backward()  # 进行反向传播，计算参数的梯度
		optimizer.step()  # 更新参数

		if e % 64 == 0:
			print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))  # 每隔一段训练迭代打印当前的损失函数值

	torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))  # 保存训练好的模型参数
	print("Save in:", '{}/net.pth'.format(mod_dir))

	'''eval'''
	net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
	net = net.eval()

	test_x = data_x.copy()
	test_x[train_size:, 0] = 0
	test_x = test_x[:, np.newaxis, :]
	test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

	'''simple way but no elegant'''
	# for i in range(train_size, len(data) - 2):
	#     test_y = net(test_x[:i])
	#     test_x[i, 0, 0] = test_y[-1]

	'''elegant way but slightly complicated'''
	eval_size = 1
	zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)
	test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
	test_x[train_size + 1, 0, 0] = test_y[-1]
	for i in range(train_size + 1, len(data) - 2):
		test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
		test_x[i + 1, 0, 0] = test_y[-1]
	pred_y = test_x[1:, 0, 0]
	pred_y = pred_y.cpu().data.numpy()

	diff_y = pred_y[train_size:] - data_y[train_size:-1]
	l1_loss = np.mean(np.abs(diff_y))
	l2_loss = np.mean(diff_y ** 2)
	print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

	plt.plot(pred_y, 'r', label='pred')
	plt.plot(data_y, 'b', label='real', alpha=0.3)
	plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
	plt.legend(loc='best')
	plt.savefig('lstm_reg.png')
	plt.pause(400)


def run_origin():
	inp_dim = 2
	out_dim = 1
	mod_dir = '..'

	'''load data'''
	data = load_data()  # axis1: number, year, month
	data_x = np.concatenate((data[:-2, 0:1], data[+1:-1, 0:1]), axis=1)
	data_y = data[2:, 0]

	train_size = int(len(data_x) * 0.75)
	train_x = data_x[:train_size]
	train_y = data_y[:train_size]

	train_x = train_x.reshape((-1, 1, inp_dim))
	train_y = train_y.reshape((-1, 1, out_dim))

	'''build model'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = RegLSTM(inp_dim, out_dim, mid_dim=4, mid_layers=2).to(device)
	criterion = nn.SmoothL1Loss()
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

	'''train'''
	var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
	var_y = torch.tensor(train_y, dtype=torch.float32, device=device)
	print('var_x.size():', var_x.size())
	print('var_y.size():', var_y.size())

	for e in range(512):
		out = net(var_x)
		loss = criterion(out, var_y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (e + 1) % 100 == 0:  # 每 100 次输出结果
			print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

	torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))

	'''eval'''
	# net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
	net = net.eval()  # 转换成测试模式

	"""
	inappropriate way of seq prediction: 
	use all real data to predict the number of next month
	"""
	test_x = data_x.reshape((-1, 1, inp_dim))
	var_data = torch.tensor(test_x, dtype=torch.float32, device=device)
	eval_y = net(var_data)  # 测试集的预测结果
	pred_y = eval_y.view(-1).cpu().data.numpy()

	plt.plot(pred_y[1:], 'r', label='pred inappr', alpha=0.3)
	plt.plot(data_y, 'b', label='real', alpha=0.3)
	plt.plot([train_size, train_size], [-1, 2], label='train | pred')

	"""
	appropriate way of seq prediction: 
	use real+pred data to predict the number of next 3 years.
	"""
	test_x = data_x.reshape((-1, 1, inp_dim))
	test_x[train_size:] = 0  # delete the data of next 3 years.
	test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
	for i in range(train_size, len(data) - 2):
		test_y = net(test_x[:i])
		test_x[i, 0, 0] = test_x[i - 1, 0, 1]
		test_x[i, 0, 1] = test_y[-1, 0]
	pred_y = test_x.cpu().data.numpy()
	pred_y = pred_y[:, 0, 0]
	plt.plot(pred_y[2:], 'g', label='pred appr')

	plt.legend(loc='best')
	plt.savefig('lstm_origin.png')
	plt.pause(4)


class RegLSTM(nn.Module):
	def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
		super(RegLSTM, self).__init__()

		self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
		# inp_dim 是LSTM输入张量的维度，我们已经根据我们的数据确定了这个值是3
		# mid_dim 是LSTM三个门 (gate) 的网络宽度，也是LSTM输出张量的维度这里是8
		# num_layers 使用2个LSTM（层数为2）对数据进行预测，然后将他们的输出堆叠起来。

		# 这个函数名是自己定的，并不是说模型特有的，这些函数都是自己写出来的，包括上面的rnn，
		# 只有nn.LSTM这种才是pytorch里自带的。
		self.reg = nn.Sequential(
			nn.Linear(mid_dim, mid_dim),
			nn.Tanh(),
			nn.Linear(mid_dim, out_dim),
		)  # regression

	def forward(self, x):
		y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

		seq_len, batch_size, hid_dim = y.shape
		y = y.view(-1, hid_dim)
		y = self.reg(y)
		y = y.view(seq_len, batch_size, -1)
		return y

	"""
	PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
	Examples::
		>>> rnn = nn.LSTM(10, 20, 2)
		>>> input = torch.randn(5, 3, 10)
		>>> h0 = torch.randn(2, 3, 20)
		>>> c0 = torch.randn(2, 3, 20)
		>>> output, (hn, cn) = rnn(input, (h0, c0))
	"""

	def output_y_hc(self, x, hc):
		# print(f'before rnn output x:{x} hc:{hc}')
		y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)
		# print(f'after rnn output y:{y} hc:{hc}')
		seq_len, batch_size, hid_dim = y.size()
		y = y.view(-1, hid_dim)
		y = self.reg(y)
		y = y.view(seq_len, batch_size, -1)
		# print(f'after reg output y:{y} hc:{hc}')

		return y, hc


class RegGRU(nn.Module):
	def __init__(self, inp_dim, out_dim, mod_dim, mid_layers):
		super(RegGRU, self).__init__()

		self.rnn = nn.GRU(inp_dim, mod_dim, mid_layers)
		self.reg = nn.Linear(mod_dim, out_dim)

	def forward(self, x):
		x, h = self.rnn(x)  # (seq, batch, hidden)

		seq_len, batch_size, hid_dim = x.shape
		x = x.view(-1, hid_dim)
		x = self.reg(x)
		x = x.view(seq_len, batch_size, -1)
		return x

	def output_y_h(self, x, h):
		y, h = self.rnn(x, h)

		seq_len, batch_size, hid_dim = y.size()
		y = y.view(-1, hid_dim)
		y = self.reg(y)
		y = y.view(seq_len, batch_size, -1)
		return y, h


def load_data():
	# passengers number of international airline , 1949-01 ~ 1960-12 per month
	# 初始化了一个np数组，1维数组，数据类型为float32，形状为(144, ) 144个元素的1维数组。
	seq_number = np.array(
		[112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
		 118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
		 114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
		 162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
		 209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
		 272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
		 302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
		 315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
		 318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
		 348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
		 362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
		 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
		 417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
		 432.], dtype=np.float32)
	# assert seq_number.shape == (144, )
	# plt.plot(seq_number)
	# plt.ion()
	# plt.pause(1)

	# 这一步是将1维数组变成一个2维数组，变成144*1的2维数组：[[],[],[]....]，
	# ‘:’表示选择所有行，np.newaxis表示在列维度上增加一个新的维度。
	# seq_number从原来的形状(144,)变成了新的形状(144, 1)，即有144行、1列的二维数组。
	seq_number = seq_number[:, np.newaxis]

	# print(repr(seq))
	# 1949~1960, 12 years, 12*12==144 month
	# 创建seq_year和seq_month数组，分别表示12年的年份和12个月的月份。np.arange(12)用于生成0到11的数组。
	# np.repeat和np.tile函数用于生成所有年份和月份的组合，得到一个形状为(144, 2)的数组seq_year_month
	# 表示从1949年1月到1960年12月的所有年份和月份组合。
	seq_year = np.arange(12)  # [0,1,2...11]
	seq_month = np.arange(12)
	seq_year_month = np.transpose(  # 合成144*2的二维数组 [[0,0],[0,1],[].....] (144,2)
		[np.repeat(seq_year, len(seq_month)),  # [0,0,(12个0)...1,1,1....11,11,11(都重复12次)] (144,)
		 np.tile(seq_month, len(seq_year))],  # [0,1,2...11,0,1,2...,11(重复12次)] (144,)
	)  # Cartesian Product

	# 将时间序列数据seq_number与年份月份数据seq_year_month在列方向上拼接，得到一个形状为(144, 3)的最终时间序列数据seq，
	# 其中第一列是乘客数量数据，第二列是年份数据，第三列是月份数据。
	seq = np.concatenate((seq_number, seq_year_month), axis=1)  # [[112.,0,0],[132,0,1]...]
	# normalization
	# 对时间序列数据进行归一化处理，使得每一列的数据均值为0，方差为1。这样做可以使得不同列之间的数据尺度相近，有助于模型训练和优化过程。
	seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
	# print(seq)
	# [乘客数，年份，月份]
	return seq


if __name__ == '__main__':
	run_train_lstm()
# run_train_gru()
# run_origin()
