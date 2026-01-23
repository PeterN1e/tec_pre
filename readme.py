###############1.数据流转###################
"""
tec24: (batch_size,seq_length,71,73)
"""

"""
epoch：训练轮次，决定整体数据集看几遍
batch_size：每次喂给模型训练的样本数量
sqe_length：时间步数
(epoch,batch_size,sqe_length)


数据集: [样本0, 样本1, 样本2, ..., 样本9999]  (共10000个)

↓ 划分

Batch 0: [样本0, 样本1, ..., 样本31]         (batch_size=32)
Batch 1: [样本32, 样本33, ..., 样本63]
...
Batch 312: [样本9984, ..., 样本9999]          (最后一批可能不足32)

↓ 每个batch内的样本

样本0: [时间步0, 时间步1, ..., 时间步23]      (seq_length=24)

↓ 训练流程

For epoch in range(10):                      # 训练10轮
    For batch in 313个batches:               # 每轮313次迭代
        模型训练这一个batch

"""
#调整batch_size和epoch是调参重点，seq_length通常是数据固有属性