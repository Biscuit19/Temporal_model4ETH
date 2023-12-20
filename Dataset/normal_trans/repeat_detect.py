import pandas as pd

# 读取四个CSV文件
file1 = "clean_phisher_transaction_in.csv"
file2 = "clean_phisher_transaction_out.csv"
file3 = "clean_normal_eoa_transaction_in_slice_1000K.csv"
file4 = "clean_normal_eoa_transaction_out_slice_1000K.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)



# 获取每个文件的哈希字段列
hash_column1 = df1['hash']
hash_column2 = df2['hash']
hash_column3 = df3['hash']
hash_column4 = df4['hash']

# 检查是否有相同的哈希值
common_hashes_1_3 = set(hash_column1) & set(hash_column3)
common_hashes_1_4 = set(hash_column1) & set(hash_column4)
common_hashes_2_3 = set(hash_column2) & set(hash_column3)
common_hashes_2_4 = set(hash_column2) & set(hash_column4)

if common_hashes_1_3:
	print(f"file1 和 file3 存在相同的哈希值:{len(common_hashes_1_3)}")
	df3 = df3[~df3['hash'].isin(common_hashes_1_3)]

if common_hashes_1_4:
	print(f"file1 和 file3 存在相同的哈希值:{len(common_hashes_1_4)}")
	df4 = df4[~df4['hash'].isin(common_hashes_1_4)]

if common_hashes_2_3:
	print(f"file1 和 file3 存在相同的哈希值:{len(common_hashes_2_3)}")
	df3 = df3[~df3['hash'].isin(common_hashes_2_3)]

if common_hashes_2_4:
	print(f"file1 和 file3 存在相同的哈希值:{len(common_hashes_2_4)}")
	df4 = df4[~df4['hash'].isin(common_hashes_2_4)]

df3.to_csv(file3, index=False)
df4.to_csv(file4, index=False)
print(f"{len(common_hashes_1_3)} 行已从 file3 中删除，并已保存为 {file3}.")
print(f"{len(common_hashes_1_4)} 行已从 file4 中删除，并已保存为 {file4}.")
print(f"{len(common_hashes_2_3)} 行已从 file3 中删除，并已保存为 {file3}.")
print(f"{len(common_hashes_2_4)} 行已从 file4 中删除，并已保存为 {file4}.")
# 删除第二个文件中重复的哈希值行
