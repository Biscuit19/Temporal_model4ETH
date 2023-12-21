import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		data = pickle.load(file)
	return data


def dump_pkl(pkl_file, data):
	# 从pkl文件加载数据
	print(f'Dumping {pkl_file}...')
	with open(pkl_file, 'wb') as file:
		pickle.dump(data, file)
	return

def graph_view(graph):
	num_nodes = graph.num_nodes
	print("Number of nodes:", num_nodes)

	num_edges = graph.num_edges
	print("Number of edges:", num_edges)

	# 图的密度
	density = num_edges / (num_nodes * (num_nodes - 1))
	print("Density of the graph: {:.6f}".format(density))

	# 获取 y=1 和 y=0 的节点数量
	y_zero = (graph.y == 0).sum().item()
	y_one = (graph.y == 1).sum().item()

	print("Number of normal:", y_zero)
	print("Number of phisher", y_one)


class GCNWithClassifier(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim,num_classes=2):
		super(GCNWithClassifier, self).__init__()
		# 定义两层GCN卷积层
		# 将当前节点及其邻居的原始特征 聚合为当前节点的隐藏特征。
		self.conv1 = GCNConv(in_dim, hidden_dim)
		# 将当前节点及其邻居的 经过第一层卷积层转换后的隐藏特征 聚合为当前节点的隐藏特征。
		# 每一层GCN都在将邻居的信息 层数越高 当前节点越能融合高阶的邻居节点特征
		self.conv2 = GCNConv(hidden_dim, num_classes)
		# self.conv2 = GCNConv(hidden_dim, hidden_dim)


		# 定义分类器
		self.classifier = nn.Sequential(
			# 全连接层，将输入的维度从hidden_dim转换为1 二分类任务
			nn.Linear(num_classes, 1),
			# nn.Linear(hidden_dim, 1),

			# 使用Sigmoid函数将输出压缩到[0, 1]的范围内。
			nn.Sigmoid()
			# 损失函数使用二元交叉熵损失（BCELoss或BCEWithLogitsLoss）
		)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		# 第一层GCN卷积层：聚合当前节点及其邻居的原始特征，产生隐藏特征
		x = self.conv1(x, edge_index)
		x = F.relu(x)  # 使用ReLU激活函数
		x = F.dropout(x, training=self.training)  # 应用dropout，减少过拟合
		# 第二层GCN卷积层：进一步聚合经过第一层处理的特征
		x = self.conv2(x, edge_index)
		# 使用ReLU激活函数是为了引入非线性。非线性激活函数使得网络能够学习和建模更复杂的模式。
		# relu的作用是将所有负值置为0。
		x = F.relu(x)  # 再次使用ReLU激活函数
		# 分类器：将隐藏特征映射到二分类输出
		x = self.classifier(x)
		return x

def train_gcn_model(train_data, test_data, save_model=False):
	num_epochs = 500
	lr = 0.1

	print('--------------------Train Dataset-------------------------')
	# 假设 graph_view 是一个用于显示图信息的函数
	graph_view(train_data)
	print('--------------------Test Dataset-------------------------')
	graph_view(test_data)
	hidden_dim=4
	# 初始化模型
	model = GCNWithClassifier(in_dim=train_data.num_node_features, hidden_dim=hidden_dim)
	print(f'node_features size: {train_data.num_node_features}')
	criterion = nn.BCELoss()
	optimizer = Adam(model.parameters(), lr=lr)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	train_data = train_data.to(device)
	test_data = test_data.to(device)

	print(f'Start Training...')
	for epoch in range(num_epochs):
		model.train()
		optimizer.zero_grad()
		output = model(train_data).view(-1)
		loss = criterion(output, train_data.y.float())
		loss.backward()
		optimizer.step()

		if (epoch + 1) % 10 == 0:
			model.eval()
			with torch.no_grad():
				output = model(test_data).view(-1)
				# print(output)
				predictions = (output > 0.5).float()
				true_labels = test_data.y

				# 计算评估指标
				precision = precision_score(true_labels.cpu(), predictions.cpu(), zero_division=0)
				recall = recall_score(true_labels.cpu(), predictions.cpu())
				f1 = f1_score(true_labels.cpu(), predictions.cpu())

				# 计算FPR
				tn, fp, fn, tp = confusion_matrix(true_labels.cpu(), predictions.cpu()).ravel()
				fpr = fp / (fp + tn)

				accuracy = accuracy_score(true_labels.cpu(), predictions.cpu())
				print(
					f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}')

	if save_model:
		model_path = './GCN_model.pth'  # 指定保存的文件路径
		torch.save(model.state_dict(), model_path)

	return model

if __name__ == "__main__":
	# 训练模型
	train_data, test_data = read_pkl('train+test_data_embed_0.pkl')
	train_gcn_model(train_data, test_data,save_model=True)
	# 不带嵌入：
	train_data, test_data = read_pkl('train+test_data_no_embed_0.pkl')
	train_gcn_model(train_data, test_data,save_model=False)

	# # 测试模型
	# test_data = read_pkl('test_data_embed_0.pkl')
	# # train_data, test_data = read_pkl('train+test_data_embed_0.pkl')
	# gat_model_path = './GAT_model.pth'
	# test_gat_model(test_data, gat_model_path)


