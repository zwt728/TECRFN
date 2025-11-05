Crossmodal Reinforced Transformer（CRT）替换跨模态层：用“交叉注意力 + 文本强化注意力”的双路结构取代原 Cross-modal TRM。
- 关键超参：nheads、layers、dst_feature_dims、attn_dropout、res_dropout、embed_dropout
相似度偏置门控：以 query 与文本序列的逐时刻余弦相似度作为 gate 偏置，强相关更信任强化路径，弱相关更信任交互路径，降低门控随机性。
- 关键超参：crt_gate_temp（门控温度）、crt_sim_scale（相似度偏置强度）
通道注意力（SE 重标定）：对门控后输出做通道级重加权，抑制噪声通道，稳住记忆层输入。
- 关键超参：crt_channel_attn_reduction（通道压缩比）
轻量残差稳态：输出叠加 α·query 残差，减小跨模态分布偏移导致的不稳。
- 关键超参：crt_residual_scale（残差系数）

文本去偏（Text Debias）
- 思路：学习一个“混杂词典”C（K×D），将文本编码 x 投影到该子空间的重建部分 recon，再用逐 token 门控做自适应剔除：x' = x − σ(Wx) ⊙ recon。目的在于抑制与情感无关/带有系统性偏见的文本方向，减少文本模态对融合的误导。
- 位置：放在文本编码（BERT 输出）之后、跨模态交互（CRT）之前；仅作用于文本流，不改变其它模态。
- 关键超参：
  - use_text_debias: true/false（开关）
  - debias_confounder_size: 词典大小 K（建议 32–64，MOSI 默认 50）
  - debias_model_dim: 线性门控隐藏维（默认 128）
  - debias_dropout: 去偏后 dropout（默认 0.1）
  - debias_num_heads: 去偏内部注意力头数（默认 8）
  - debias_num_layers: 去偏层数（默认 1–2）
  - use_kmeans_init: 词典是否用 KMeans 初始化（默认 false）