import pickle 
import networkx as nx

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

G = load_pickle('./MulDiGraph.pkl')

print(nx.info(G))
# 获取节点数
num_nodes = G.number_of_nodes()

# 获取边数
num_edges = G.number_of_edges()

# 计算平均入度
total_in_degree = sum(d for n, d in G.in_degree())
average_in_degree = total_in_degree / num_nodes

print("节点数:", num_nodes)
print("边数:", num_edges)
print("平均入度:", average_in_degree)


# Traversal nodes:
for idx, nd in enumerate(nx.nodes(G)):
    # nd是节点地址0x1f1e784a61a8ca0a90250bcd2170696655b28a21
    print(nd)
    # 节点地址有一个属性{'isp': 0}，0表示正常账户，1表示钓鱼账户
    print(G.nodes[nd])
    break

for u, v, key, data in G.edges(keys=True, data=True):
    tag = G.nodes[u]['isp']
    print(f"源: {u},tag:{tag} 目标: {v}, 边的信息: {data},key:{key}")
    break

# Travelsal edges:
for ind, edge in enumerate(nx.edges(G)):
    # u->v的交易
    (u, v) = edge
    print(u,v)
    eg = G[u][v]
    print(eg)

    amo, tim = eg['amount'], eg['timestamp']
    print(amo, tim)
    break