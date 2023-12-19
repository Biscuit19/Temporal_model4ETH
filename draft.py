import copy
import unittest
import random
from pprint import pprint

from torch import tensor, float32
from tqdm import tqdm


def data_padding(all_trans_list):
	print('[+]Padding data...')
	trans_size = 2
	padding_tran = [0, 0, 0, 0]
	print('padding tran:', padding_tran)

	# 处理整个数据集
	for all_windows in tqdm(all_trans_list, desc='Processing Data', unit='data'):

		for window in all_windows:
			# 计算要填充几次
			window_len = len(window)
			trans_lack = trans_size - window_len

			if trans_lack > 0:
				# 这里使用引用，减少占用空间
				window.extend([padding_tran] * trans_lack)
			elif trans_lack < 0:
				# 随机删去超过交易长度的元素，以满足交易长度
				del_indices = random.sample(range(window_len), -trans_lack)
				del_indices.sort(reverse=True)
				for i in del_indices:
					del window[i]

	return all_trans_list


if __name__ == '__main__':
	original_data = [
		[
			[[1, 2, 3, 3], [1, 2, 3, 3]],
			[[1, 2, 3, 3]],
			[]  # 空窗口
		],
		[
			[[1, 2, 3, 3], [1, 2, 3, 3]],
			[[1, 2, 3, 3]],
			[]  # 空窗口
		]
	]

	trans_list=data_padding(original_data)
	pprint(trans_list)

	batchs = tensor(trans_list, dtype=float32)

	print(batchs)

