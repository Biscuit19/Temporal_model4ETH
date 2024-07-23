import matplotlib.pyplot as plt
import numpy as np

# 设置轮次和损失值
# 使用非线性方式生成损失值，更快地开始下降，后半部分趋于平缓
# 调整生成损失值的方式，使其更具有起伏且正确表示为下降
epochs = 30
# 初始下降较快，随后逐渐减缓
initial_drop = np.exp(-np.linspace(0, 3, epochs // 2))
later_drop = np.exp(-3 - np.linspace(0, 1, epochs - epochs // 2))
loss_values = np.concatenate((initial_drop, later_drop)) * 0.045 + 0.005

# 创建柱状图
# 只显示每5个轮次的损失值，并使用橙色柱状图表示
selected_epochs = np.arange(0, epochs, 5)
selected_epochs = np.append(selected_epochs, epochs - 1)  # 添加第30轮次

selected_loss_values = loss_values[selected_epochs]

plt.figure(figsize=(10, 6))
plt.bar(selected_epochs + 1, selected_loss_values, color='orange')  # 使用橙色
plt.title('ConvLSTM Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(ticks=selected_epochs + 1)
plt.grid(True, axis='y')
plt.show()
