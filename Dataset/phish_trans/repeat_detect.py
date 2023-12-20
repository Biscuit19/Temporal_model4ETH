import pandas as pd

# 读取两个CSV文件
file1 = "clean_phisher_transaction_in.csv"
file2 = "clean_phisher_transaction_out.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 获取两个文件中的哈希字段列
hash_column1 = df1['hash']
hash_column2 = df2['hash']

# 检查是否有相同的哈希值
common_hashes = set(hash_column1) & set(hash_column2)

if common_hashes:
    print(f"这两个文件中存在相同的哈希值: {len(common_hashes)} 个")
    # 删除file1中重复的行
    df1 = df1[~df1['hash'].isin(common_hashes)]
    # 保存修改后的文件
    df1.to_csv(f"clean_{file1}.csv", index=False)
else:
    print("这两个文件中没有相同的哈希值。")
