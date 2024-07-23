import networkx as nx
import matplotlib.pyplot as plt
import random
import time

def random_timestamp(start, end):
    """生成一个在两个时间戳之间的随机时间戳"""
    return random.randint(start, end)

# 设置时间范围（例如从现在到过去一周）
time_end = int(time.time())
time_start = time_end - 7 * 24 * 3600  # 一周前

# 创建一个有向多边图
G = nx.MultiDiGraph()

# 添加节点
nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
G.add_nodes_from(nodes)

# 添加多条边，随机生成交易金额作为权重，并标记为ETH，同时添加时间戳
edges = [
    ("A", "B", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("A", "C", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("C", "D", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("D", "B", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("A", "D", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("E", "A", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("F", "B", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("G", "C", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end)),
    ("H", "D", random.uniform(0.1, 0.5), random_timestamp(time_start, time_end))
]

# 将交易添加到图中，每条边都有唯一的标识
for start, end, weight, timestamp in edges:
    G.add_edge(start, end, weight=weight, label=f'{weight:.2f} ETH', timestamp=timestamp)

# 使用Spring布局
pos = nx.spring_layout(G)
plt.figure(figsize=(14, 8))  # 调整画布大小以更好地展示更多的节点和边

# 绘制节点，调整节点大小为1000
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue', alpha=0.6)

# 绘制边，边的粗细统一设置为2
for start, end, data in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], width=2, arrowstyle='-|>', arrowsize=20, edge_color='gray')

# 添加节点标签
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

# 添加边标签，现在包括时间戳
edge_labels = {(u, v): f"{d['label']}, {d['timestamp']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,font_color='red')


plt.title('Ethereum Transaction Graph Example')
plt.axis('off')  # 不显示坐标轴
plt.show()
