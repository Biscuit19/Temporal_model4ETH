import pickle
import random
from torch.utils.data import DataLoader, TensorDataset
import torch,gc
import torch.nn as nn
from torch import optim, tensor
import os


# 设置PYTORCH_CUDA_ALLOC_CONF参数
print(torch.__version__)

"""
定义ConvLSTM每一层的、每个时间点的模型单元，及其计算。
"""
torch.set_printoptions(precision=4, sci_mode=False)


# torch.set_printoptions(precision=4, sci_mode=False, threshold=10000000)


class ConvLSTMCell(nn.Module):

	def __init__(self, input_dim, hidden_dim, kernel_size, bias):
		"""
		单元输入参数如下：
		input_dim: 输入张量对应的通道数，对于彩图为3，灰图为1。
		hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要“随意”设置。
		kernel_size: (int, int)，卷积核，并且卷积核通常都需要为奇数。
		bias: bool，单元计算时，是否加偏置，通常都要加，也就是True。
		"""
		super(ConvLSTMCell, self).__init__()  # self：实例化对象，__init__()定义时该函数就自动运行,
		# super()是实例self把ConvLSTMCell的父类nn.Modele的__init__()里的东西传到自己的__init__()里
		# 总之，这句是搭建神经网络结构必不可少的。
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.kernel_size = kernel_size
		self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # //表示除法后取整数，为使池化后图片依然对称，故这样操作。
		self.bias = bias

		# print(f'padding:{self.padding}')
		"""
		nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,dilation=1,groups=1,bias)
		二维的卷积神经网络
		"""
		self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,  # 每个单元的输入为上个单元的h和这个单元的x，
							  # 所以h和x要连接在一起，在x的通道数上与h的维度上相连。
							  out_channels=4 * self.hidden_dim,  # 输入门，遗忘门，输出门，激活门是LSTM的体现，
							  # 每个门的维度和隐藏层维度一样，这样才便于进行+和*的操作
							  # 输出了四个门，连接在一起，后面会想办法把门的输出单独分开，只要想要的。
							  kernel_size=self.kernel_size,
							  padding=self.padding,
							  bias=self.bias)

	def forward(self, input_tensor, cur_state):
		"""
		input_tensor:此时还是四维张量，还未考虑len_seq，[batch_size,channels,h,w]，[b,c,h,w]。
		cur_state:每个时间点单元内，包含两个状态张量：h和c。
		"""
		h_cur, c_cur = cur_state  # h_cur的size为[batch_size,hidden_dim,height,width],c_cur的size相同,也就是h和c的size与input_tensor相同

		combined = torch.cat([input_tensor, h_cur], dim=1)  # 把input_tensor与状态张量h,沿input_tensor通道维度(h的节点个数），串联。
		# combined:[batch_size,input_dim+hidden_dim,height,weight]

		combined_conv = self.conv(combined)  # Conv2d的输入，[batch_size,channels,height,width]
		# Conv2d的输出，[batch_size,output_dim,height,width]，这里output_dim=input_dim+hidden_dim

		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
											 dim=1)  # 将conv的输出combined_conv([batch_size,output_dim,height,width])
		# 分成output_dim这个维度去分块，每个块包含hidden_dim个节点信息
		# 四个块分别对于i,f,o,g四道门，每道门的size为[b,hidden_dim,h,w]
		i = torch.sigmoid(cc_i)  # 输入门
		f = torch.sigmoid(cc_f)  # 遗忘门
		o = torch.sigmoid(cc_o)  # 输出门
		g = torch.tanh(cc_g)  # 激活门

		# print(f"Shapes: f={f.shape}, c_cur={c_cur.shape}, i={i.shape}, g={g.shape}")

		c_next = f * c_cur + i * g  # 主线，遗忘门选择遗忘的+被激活一次的输入，更新长期记忆。
		h_next = o * torch.tanh(c_next)  # 短期记忆，通过主线的激活和输出门后，更新短期记忆（即每个单元的输出）。

		return h_next, c_next  # 输出当前时间点输出给下一个单元的，两个状态张量。

	def init_hidden(self, batch_size, image_size):
		"""
		初始状态张量的定义，也就是说定义还未开始时输入给单元的h和c。
		"""
		height, width = image_size
		init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)  # 初始输入0张量
		init_c = torch.zeros(batch_size, self.hidden_dim, height, width,
							 device=self.conv.weight.device)  # [b,hidden_dim,h,w]
		# self.conv.weight.device表示创建tensor存放的设备
		# 和conv2d进行的设备相同
		return (init_h, init_c)

class ConvLSTM(nn.Module):
	"""
	输入参数如下：
	input_dim:输入张量对应的通道数，对于彩图为3，灰图为1。
	hidden_dim:h,c两个状态张量的节点数，当多层的时候，可以是一个列表，表示每一层中状态张量的节点数。
	kernel_size:卷积核的尺寸，默认所有层的卷积核尺寸都是一样的，也可以设定不同的lstm层的卷积核尺寸不同。
	num_layers:lstm的层数，需要与len(hidden_dim)相等。
	batch_first:dimension 0位置是否是batch，是则True。
	bias:是否加偏置，通常都要加，也就是True。
	return_all_layers:是否返回所有lstm层的h状态。

	"""

	def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
				 batch_first=True, bias=True, return_all_layers=False):
		super(ConvLSTM, self).__init__()

		self._check_kernel_size_consistency(kernel_size)  # 后面def了的，检查卷积核是不是列表或元组。

		kernel_size = self._extend_for_multilayer(kernel_size, num_layers)  # 如果为多层，将卷积核以列表的形式分入多层，每层卷积核相同。
		hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)  # 如果为多层，将隐藏节点数以列表的形式分入多层，每层卷积核相同。

		if not len(kernel_size) == len(hidden_dim) == num_layers:  # 判断卷积层数和LSTM层数的一致性，若不同，则报错。
			raise ValueError('Inconsistent list length.')

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.kernel_size = kernel_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.bias = bias
		self.return_all_layers = return_all_layers  # 一般都为False。

		cell_list = []  # 每个ConvLSTMCell会存入该列表中。
		for i in range(0, self.num_layers):  # 当LSTM为多层，每一层的单元输入。
			if i == 0:
				cur_input_dim = self.input_dim  # 一层的时候，单元输入就为input_dim，多层的时候，单元第一层输入为input_dim。
			else:
				cur_input_dim = self.hidden_dim[i - 1]  # 多层的时候，单元输入为对应的，前一层的隐藏层节点情况。

			cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
										  hidden_dim=self.hidden_dim[i],
										  kernel_size=self.kernel_size[i],
										  bias=self.bias))

		self.cell_list = nn.ModuleList(cell_list)  # 把定义的多个LSTM层串联成网络模型，ModuleList中模型可以自动更新参数。

	def forward(self, input_tensor, hidden_state=None):
		"""
		input_tensor: 5D张量，[l, b, c, h, w] 或者 [b, l, c, h, w]
		hidden_state: 第一次输入为None，
		Returns：last_state_list, layer_output
		"""
		if not self.batch_first:
			input_tensor = input_tensor.permute(1, 0, 2, 3, 4)  # (t, b, c, h, w) -> (b, t, c, h, w)

		if hidden_state is not None:
			raise NotImplementedError()
		else:
			b, _, _, h, w = input_tensor.size()  # 自动获取 b,h,w信息。
			hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

		layer_output_list = []
		last_state_list = []

		seq_len = input_tensor.size(1)  # 根据输入张量获取lstm的长度。
		cur_layer_input = input_tensor  # 主线记忆的第一次输入为input_tensor。

		for layer_idx in range(self.num_layers):  # 逐层计算。

			h, c = hidden_state[layer_idx]  # 获取每一层的短期和主线记忆。
			output_inner = []
			for t in range(seq_len):  # 序列里逐个计算，然后更新。
				h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
				output_inner.append(h)  # 第layer_idx层的第t个stamp的输出状态。

			layer_output = torch.stack(output_inner, dim=1)  # 将第layer_idx层的所有stamp的输出状态串联起来。
			cur_layer_input = layer_output  # 准备第layer_idx+1层的输入张量，其实就是上一层的所有stamp的输出状态。

			layer_output_list.append(layer_output)  # 当前层(第layer_idx层）的所有timestamp的h状态的串联后，分层存入列表中。
			last_state_list.append([h, c])  # 当前层（第layer_idx层）的最后一个stamp的输出状态的[h,c]，存入列表中。

		if not self.return_all_layers:  # 当不返回所有层时
			layer_output_list = layer_output_list[-1:]  # 只取最后一层的所有timestamp的h状态。
			last_state_list = last_state_list[-1:]  # 只取最后一层的最后的stamp的输出状态[h,c]。

		return layer_output_list, last_state_list

	def _init_hidden(self, batch_size, image_size):
		"""
		所有lstm层的第一个时间点单元的输入状态。
		"""
		init_states = []
		for i in range(self.num_layers):
			init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))  # 每层初始单元，输入h和c，存为1个列表。
		return init_states

	@staticmethod  # 静态方法，不需要访问任何实例和属性，纯粹地通过传入参数并返回数据的功能性方法。
	def _check_kernel_size_consistency(kernel_size):
		"""
		检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
		"""
		if not (isinstance(kernel_size, tuple) or
				(isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
			raise ValueError('`kernel_size` must be tuple or list of tuples')

	@staticmethod
	def _extend_for_multilayer(param, num_layers):
		"""
		扩展到LSTM多层的情况
		"""
		if not isinstance(param, list):
			param = [param] * num_layers
		return param

# 定义自己的网络类MyConvLSTM，将ConvLSTM套在其中
class AutoEncoder_ConvLSTM(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, batch_first, bias, return_all_layers):
		super(AutoEncoder_ConvLSTM, self).__init__()
		self.convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers, batch_first, bias,
								 return_all_layers)

		# 编码器
		self.encoder = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers, batch_first, bias,
								return_all_layers=True)
		# 解码器
		self.decodkernel_sizeer = ConvLSTM(hidden_channels, input_channels, kernel_size, num_layers, batch_first, bias,
								return_all_layers)

	# def forward(self, x):
	#
	# 	output, h_c = self.convlstm(x)
	# 	# output=[tensor(output1,output2...),] output是一个列表，output[0]是一个张量（下标表示取哪一层，如果有两层且返回所有层，则有另一个LSTM层的数据比如output[1]）
	# 	# output[0]=(batch_size, sequence_length, hidden_dim, Height, Width)
	# 	# 包含着不同batch的所有时间步的隐藏状态
	#
	# 	# h_c=[[h,c]], h_c是一个列表，h_c[0]也是一个列表（下标表示取哪一层），h_c[0][0]是h张量,
	# 	# h_c[0][0]=(batch_size, hidden_dim, Height, Width) （h张量的维度，c也一样）
	# 	# 包含着不同batch的最后一个时间步的隐藏状态（所以时间步维度被省略了）
	# 	# [0][0]表示h隐藏状态，最后一层最后一个时间步的h状态
	#
	# 	# output=output[0][0]
	# 	# (h,c)=(h_c[0][0],h_c[0][1])
	#
	# 	return output, h_c

	def forward(self, x):
		encode_output, encode_h_c = self.encoder(x)
		# output=[tensor(output1,output2...),] output是一个列表，output[0]是一个张量（下标表示取哪一层，如果有两层且返回所有层，则有另一个LSTM层的数据比如output[1]）
		# output[0]=(batch_size, sequence_length, hidden_dim, Height, Width)
		# 包含着不同batch的所有时间步的隐藏状态

		# 因为要输入到解码器层，所以要取所有时间步，即取output而不是h隐藏状态
		encode_output = encode_output[0]
		#
		decoded_output_list, decode_h_c_list = self.decoder(encode_output)
		# 返回解码器生成，取最后一层
		decode_output = decoded_output_list[0]
		decode_h_c = decode_h_c_list[0]
		return decode_output, decode_h_c

	def encode(self, x):
		output, h_c = self.encoder(x)
		# 取最后一层
		encode_output = output[0]
		encode_h_c = h_c[0]

		return encode_output, encode_h_c


def data_load(pkl_file):
	with open(pkl_file, 'rb') as f:
		batch_data = pickle.load(f)

	return batch_data

# 张量取样
def random_sample_tensor(input_tensor, num_elements):
	"""
	随机截取输入张量的第一维度上的指定数量的元素，并返回新的张量。

	参数：
	input_tensor (torch.Tensor): 输入张量。
	num_elements (int): 要截取的元素数量。

	返回：
	torch.Tensor: 包含随机截取元素的新张量。
	"""
	# 确保 num_elements 不超过第一维度的大小
	num_elements = min(num_elements, input_tensor.shape[0])
	# 生成一个包含 num_elements 个不重复索引的列表
	indices = random.sample(range(input_tensor.shape[0]), num_elements)
	# 使用这些索引来截取输入张量的子集
	cropped_tensor = input_tensor[indices]

	return cropped_tensor


def model_train():
	tran_size = 300

	data_file = './conv_train_data.pkl'
	data_batch = data_load(data_file)

	# 测试训练：
	# sample_num=100
	# data_batch=random_sample_tensor(data_batch,sample_num)
	print('data size:', data_batch.size())

	# 划分训练集和验证集
	total_size = len(data_batch)
	val_size = int(0.1 * total_size)
	train_size = total_size - val_size

	train_data, val_data = torch.utils.data.random_split(data_batch, [train_size, val_size])

	# 获取原始数据集
	train_dataset = TensorDataset(train_data.dataset)
	val_dataset = TensorDataset(val_data.dataset)

	# 创建 DataLoader
	# 定义批次大小和创建 DataLoader，shuffle表示打乱
	batch_size = 256  # 设置批次大小
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	input_dim = 4
	hidden_dim = 64
	kernel_size = (tran_size+1, 1)
	num_layers = 1
	bias = True
	batch_first = True  # 是要batch first的，因为数据的batch在第一维度 (t, b, c, h, w) -> (b, t, c, h, w)
	return_all_layers = False
	lr=0.001
	# 将模型移到GPU上
	AECL_model = AutoEncoder_ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, bias, batch_first,
									  return_all_layers).to(device)
	print(f'(hidden_dim,kernel_size,num_layers,lr):{hidden_dim,kernel_size,num_layers,lr}')

	criterion = nn.MSELoss()
	criterion.to(device)
	optimizer = optim.Adam(AECL_model.parameters(), lr=lr)


	epoch_num = 20
	epoch_list = []
	loss_list = []

	for epoch in range(epoch_num):
		AECL_model.train()  # 切换到训练模式

		total_loss = 0.0
		for batch_data in train_dataloader:
			# print(type(batch_data))
			# 这里为什么取[0]，因为DataLoader的每个batch是一个元组，第一个元素是输入数据，第二个元素是对应的标签。
			# 如果没有标签，每个批次将会是一个包含一个元素的列表，所以要取[0]
			batch_data = batch_data[0].to(device)  # 从 DataLoader 中获取批次数据并移动到设备

			optimizer.zero_grad()
			decode_output, decode_h_c = AECL_model(batch_data)
			loss = criterion(decode_output, batch_data)
			loss.backward()

			optimizer.step()

			total_loss += loss.item()

		# 在训练循环内，添加过拟合判断
		if (epoch + 1) % 1 == 0:
			average_loss = total_loss / len(train_dataloader)
			loss_list.append(average_loss)
			epoch_list.append(epoch)
			print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {average_loss:.4f}')

		# 添加验证循环
		AECL_model.eval()  # 切换到评估模式
		with torch.no_grad():
			val_loss = 0.0
			for val_batch_data in val_dataloader:
				val_batch_data = val_batch_data[0].to(device)
				val_decode_output, _ = AECL_model(val_batch_data)
				val_loss += criterion(val_decode_output, val_batch_data).item()

			average_val_loss = val_loss / len(val_dataloader)
			print(f'Validation Loss: {average_val_loss:.4f}')

			# 判断是否过拟合
			patience = 4  # 允许的连续周期内验证集损失上升的次数
			if len(loss_list) > patience and all(average_val_loss > loss_list[-i - 1] for i in range(1, patience + 1)):
				print(
					f"Validation loss increased for {patience} consecutive epochs. Possible overfitting. Stopping training.")
				break  # 停止训练

	# 保存模型的权重和结构
	model_path = './AECL_model.pth'  # 指定保存的文件路径
	torch.save(AECL_model.state_dict(), model_path)

	# # 创建一个图表
	# plt.figure()
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.title('Loss Curve')
	# # 绘制 loss 值曲线
	# plt.plot(epoch_list, loss_list, marker='o')
	# # 显示图表
	# plt.show()



def test():
	x = torch.rand((70000, 31, 4, 300, 1))

	# (batch_size, sequence_length, Channels, Height, Width)
	# 创建自定义的ConvLSTM网络
	AECL_model = AutoEncoder_ConvLSTM(input_channels=4, hidden_channels=64, kernel_size=(301, 1), num_layers=1,
									   batch_first=True,
									   bias=True, return_all_layers=False)
	AECL_model.load_state_dict(torch.load('AECL_model.pth'))

	# 使用自定义的网络进行前向传播
	# output=tensor(output1,output2...) 不同output表示不同batch的所有时间步的h状态
	output, h_c = AECL_model.encode(x)

	# output[0]就是一个tensor，output这个列表里有不同层的tensor，设置只返回最后一层的tensor

	# output = output[0][2]
	# output_tensor=tensor(output[0])
	# h_c=[h,c] ，[0][0]表示h隐藏状态，最后一层最后一个时间步的h状态

	h = h_c[0]
	c = h_c[0]

	# output[0]=(sequence_length, hidden_dim, Height, Width)
	# output0表示输出第一个batch的output
	print("output_size:",output[0].size())

	# h0表示输出第一个batch的h
	# h[0]=(hidden_dim, Height, Width)
	print("hidden_state:", h[0].size())

	return output[0]

if __name__ == '__main__':
	# test()

	model_train()
