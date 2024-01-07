import pickle

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def read_pkl(pkl_file):
	# 从pkl文件加载数据
	print(f'Reading {pkl_file}...')
	with open(pkl_file, 'rb') as file:
		accounts_dict = pickle.load(file)
	return accounts_dict


def train_and_evaluate_lgbm(train_dataset):
	train_data, label_data=train_dataset
	# random_state就是随机种子数，可以保证每次分割相同
	random_state=41
	X_train, X_test, y_train, y_test = train_test_split(
		train_data, label_data, test_size=0.2,random_state=random_state
	)

	# 确保数据是正确的格式
	X_train, X_test = np.array(X_train), np.array(X_test)
	y_train, y_test = np.array(y_train), np.array(y_test)

	# 设置训练参数
	params = {
		'objective': 'binary',  # 二元分类
		'metric': 'binary_logloss',  # 二元分类的损失函数
		'num_leaves': 31,  # 叶子节点数
		'learning_rate': 0.05,  # 学习率
		'feature_fraction': 0.9,  # 建树的特征选择比例
		'bagging_fraction': 0.8,  # 建树的样本采样比例
		'bagging_freq': 5,  # 每5轮进行一次bagging
		'verbose': -1  # 信息输出设置（-1表示不输出）
	}

	# 创建数据集
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

	# 训练模型
	print("Starting training...")
	gbm = lgb.train(params,
					lgb_train,
					num_boost_round=500,
					valid_sets=[lgb_eval])


	# 预测
	print("Starting predicting...")
	y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
	# 将概率值转换为二元输出
	y_pred_binary = np.where(y_pred > 0.5, 1, 0)

	# 评估模型
	accuracy = accuracy_score(y_test, y_pred_binary)
	precision = precision_score(y_test, y_pred_binary, zero_division=0)
	recall = recall_score(y_test, y_pred_binary)
	f1 = f1_score(y_test, y_pred_binary)

	print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

if __name__ == '__main__':
	train_data = read_pkl('mlp_data_all_feature.pkl')
	train_and_evaluate_lgbm(train_data)

	train_data = read_pkl('mlp_data_embed.pkl')
	train_and_evaluate_lgbm(train_data)

	train_data = read_pkl('mlp_data_static.pkl')
	train_and_evaluate_lgbm(train_data)