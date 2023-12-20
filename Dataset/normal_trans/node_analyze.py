import pandas as pd

# 读取四个CSV文件
# in表示这里的toaddress都是恶意账户
file1 = "clean_phisher_transaction_in.csv"
# out表示这里的from address都是恶意账户
file2 = "clean_phisher_transaction_out.csv"

# in表示这里的toaddress都是正常账户
file3 = "clean_normal_eoa_transaction_in_slice_1000K.csv"
file4 = "clean_normal_eoa_transaction_out_slice_1000K.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)


# 恶意交易中的恶意地址
phisher1 = set(df1['to_address'])
phisher2 = set(df2['from_address'])
phisher_address = phisher1 | phisher2
print(f'恶意地址数：{len(phisher_address)}')


# 恶意交易数据集中的非恶意地址
phisher_related1 = set(df1['from_address'])
phisher_related2 = set(df2['to_address'])
related_addr = (phisher_related1 | phisher_related2)
print(f'恶意交易的关联地址数：{len(related_addr)}')
related_phisher= phisher_address & related_addr
print(f'恶意交易数据集中的互相转账恶意地址数：{len(related_phisher)}')



related_normal = related_addr.difference(phisher_address)
print(f'恶意交易数据集中的非恶意地址数：{len(related_normal)}')


# 正常数据集中的所有地址
normal2 = set(df3['to_address'])
normal3 = set(df3['from_address'])
normal4 = set(df4['to_address'])
normal5 = set(df4['from_address'])
normal_data_address = normal2 | normal3 | normal4 | normal5
print(f'正常数据集中的所有地址数：{len(normal_data_address)}')

common_address = related_normal & normal_data_address
print(f"恶意数据集中的非恶意地址 与 正常数据集中的所有地址 的交集地址数:{len(common_address)}")

lack_address=related_normal.difference(normal_data_address)
print(f"恶意数据集中的非恶意地址 的缺失地址数:{len(lack_address)}")


# 指定要保存结果的文件名
output_file = "common_address.txt"
# 打开文件以写入地址
with open(output_file, 'w') as file:
	for address in common_address:
		file.write(str(address) + '\n')

# 指定要保存结果的文件名
output_file = "lack_addresses.txt"
# 打开文件以写入地址
with open(output_file, 'w') as file:
	for address in lack_address:
		file.write(str(address) + '\n')
