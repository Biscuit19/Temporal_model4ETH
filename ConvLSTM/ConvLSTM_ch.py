import torch
import torch.nn as nn

"""
定义ConvLSTM每一层的、每个时间点的模型单元，及其计算。
"""


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


"""
定义整个ConvLSTM按序列和按层数的结构和计算。
输入介绍:
        五维数据，[batch_size,len_seq,channels,height,width] or [l,b,c,h,w]。
输出介绍:
        输出两个列表：layer_output_list和last_state_list。
        列表0：layer_output_list--单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[b,l,hidden_dim,h,w]。
        列表1：last_state_list--双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个时间单元的输出状态[h,c],
               h.size=c.size=[b,hidden_dim,h,w]
使用示例:
        >> x = torch.rand((64, 20, 1, 64, 64))
        >> convlstm = ConvLSTM(1, 30, (3,3), 1, True, True, False)
        >> _,last_states = convlstm(x)
        >> h = last_states[0][0]  #第一个0表示要第1层的列表，第二个0表示要h的张量。
"""

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



if __name__ == '__main__':

	x = torch.rand((10, 31, 4, 300, 1))
	convlstm = ConvLSTM(4, 64, (301, 1), 1, True, True, False)
	_, last_states = convlstm(x)
	h = last_states[0][0]  # 第一个0表示要第一层的列表，第二个0表示要列表里第一个位置的h输出。

	print(h.size())