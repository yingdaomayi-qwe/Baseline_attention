import torch
import torch.nn as nn
import math


# 定义基于正弦和余弦的位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, D_embed, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建一个足够大的位置编码矩阵 (max_len, D_embed)
        pe = torch.zeros(max_len, D_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算每个维度的角频率
        div_term = torch.exp(torch.arange(0, D_embed, 2).float() * (-math.log(10000.0) / D_embed))

        # 交替使用 sin 和 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个批次维度，并作为不可训练参数
        self.pe = pe

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, D_embed)
        pe_expanded = self.pe[:x.size(1), :].unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)

        # 将位置编码与输入x相加
        output = x + pe_expanded
        return output


# 定义主模型
class RSRP_Prediction_Model(nn.Module):
    def __init__(self, D_embed, S_TF, K, H_TF, dropout_rate=0.1,num_tx=64):
        super(RSRP_Prediction_Model, self).__init__()

        # 参数定义
        self.D_embed = D_embed
        self.S_TF = S_TF
        self.K = K
        self.H_TF = H_TF

        # 可学习的预测token
        self.learnable_prediction_tokens = nn.Parameter(torch.randn(K - 1, D_embed))

        # 基于正弦和余弦的位置编码
        self.positional_encoding = PositionalEncoding(D_embed, max_len=S_TF)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_embed, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=H_TF)

        # 全连接嵌入层
        self.fc_embedding = nn.Linear(num_tx, D_embed)

        # FC Resizing Layer 1 and 2
        self.fc_resizing1 = nn.Linear(D_embed, 4 * D_embed)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.fc_bs_tx = nn.Linear(4 * D_embed, num_tx)  # BS Tx beam RSRP Prediction

    def forward(self, x):
        # 假设输入的x是 (batch_size, D_in) 的形状, 其中D_in根据M_Tx * S_TF，或者其它维度组合

        # 1. 生成嵌入token
        embed_tokens = self.fc_embedding(x)  # (batch_size, S_TF, D_embed)

        # 2. 添加基于正弦和余弦的位置编码
        embed_tokens = self.positional_encoding(embed_tokens)

        # 3. 将嵌入tokens和可学习的预测tokens拼接
        prediction_tokens = self.learnable_prediction_tokens.unsqueeze(0).repeat(embed_tokens.size(0), 1, 1)
        transformer_input = torch.cat([embed_tokens, prediction_tokens], dim=1)  # Concatenate in the 2nd dimension

        # 4. 通过Transformer编码器层
        transformer_output = self.transformer(transformer_input)  # (batch_size, S_TF + K - 1, D_embed)
        transformer_output = transformer_output[:,self.S_TF:,:]
        # 5. 全连接层resize
        resized_output = self.fc_resizing1(transformer_output)
        resized_output = self.relu(resized_output)
        resized_output = self.dropout(resized_output)  # Apply dropout

        # 6. 提取不同类型的预测结果
        bs_tx_rsrp = self.fc_bs_tx(resized_output)  # (batch_size, num_tx,K-1)

        return bs_tx_rsrp

