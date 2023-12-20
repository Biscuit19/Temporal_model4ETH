# Temporal_model4ETH

## 测试模型

进入**ConvLSTM/GNN/Data_generate**文件夹下：

1. 数据重组：使用**中山大学数据集**（数据集为图结构，ZSU.py中指明了格式），执行**Account_generate_ZSU.py** 生成all_account_data_zsu.pkl和all_tnx_data_zsu.pkl

   - all_account_data.pkl格式：

     ```python
     accounts_dict[address] = {
         'account_address': address,
         'out_trans': [tnx,tnx2],
         'in_trans': [tnx...],
         'all_trans': [tnx...],
         'category': tag #1为正样本 0为负样本
     }
     ```

     其中的tnx格式为：

     ```
     [amount, block_timestamp, direction, timewindow, from_address, to_address, trans_hash]
     ```

     direction为1/-1 转入/出

     all_tnx_data.pkl格式：

     ```
     [tnx1,tnx2,tnx3...]
     ```

     其中tnx格式为：

     ```
     [amount, block_timestamp, timewindow, from_address, to_address]
     ```

   - 重组的数据格式需完全一致，否则无法运行后续的特征生成代码。

2. 图构建：运行**whole_graph_convert.py**生成whole_graph_data.pkl图文件和address_to_index.pkl文件

3. 子图提取和特征生成：运行**part_graph_convert.py**生成测试集test_data_embed_0.pkl

   - 导入了Data_Restruct.py，Feature_for_node.py，AutoEncoder_ConvLSTM.py，train_test_split.py，AECL_model.pth

   - 此时需要根据数据集交易的跨度调整时间窗口大小和值。在Feature_for_node.py文件中调整

4. 训练或测试模型：运行**GAT.py**

   - 测试：test_gat_model(test_data, gat_model_path)
   - 训练：train_gat_model()




- 由于内存问题，步骤需要分布执行，否则可能导致内存不足报错。