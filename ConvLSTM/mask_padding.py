import torch
import torch.nn as nn

# 虚拟训练数据，包含填充和实际数据
data = [
	[1, 2, 3, 0,8,1,1,1,1,1,7],
	[4, 5, 0, 0],
	[6, 7, 8, 9]
]

# 计算各序列的长度
lengths = torch.tensor([len(seq) for seq in data])

# 填充前的数据
print("填充前的数据：")
for seq in data:
	print(seq)

# 计算掩码
mask = torch.zeros(len(data), max(lengths)).bool()
for i, length in enumerate(lengths):
	mask[i, :length] = 1

# 填充后的数据
print("\n填充后的数据：")
padded_data = nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in data], batch_first=True)
print(padded_data)

# 打印掩码
print("\n生成的掩码：")
print(mask)


class RNNModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(RNNModel, self).__init__()

		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		packed_sequence = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
		packed_output, hidden = self.rnn(packed_sequence)
		output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
		output = output * mask.unsqueeze(-1).float()
		final_hidden = hidden[-1]
		output = self.fc(final_hidden)

		return output


# 假设每个序列的特征维度
input_size = 1
hidden_size = 2
num_layers = 1
output_size = 1

# 创建RNN模型
model = RNNModel(input_size, hidden_size, num_layers, output_size)

# 使用填充后的数据进行训练
output = model(padded_data)
print("\n模型输出：")
print(output)
