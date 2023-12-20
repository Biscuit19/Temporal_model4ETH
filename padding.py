import torch

# 示例序列数据
seq1 = torch.tensor([1, 2, 3])
seq2 = torch.tensor([4, 5])
seq3 = torch.tensor([6, 7, 8, 9, 10])

batch = list()
batch.append(seq1)
batch.append(seq2)
batch.append(seq3)
print(batch)

# 使用 pad_sequence 进行填充，默认填充值是0
batch = torch.nn.utils.rnn.pad_sequence(batch,batch_first=True)

print(batch)
print(batch.size())
# 输出：
# tensor([[1, 4, 6],
#         [2, 5, 7],
#         [3, 0, 8],
#         [0, 0, 9]])
