import  torch
import torch.nn as nn
import math
import numpy as np

class TecPreTransformer(nn.Module):
    """

    """
    def __init__(self,
                 in_dim,
                 history_len,
                 predict_len,
                 d_model,#模型维度
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,  #feedforward维度
                 activation="relu",#激活函数
                 dropout=0.1):
        super( ).__init__()
        self.predict_len = predict_len
        self.d_model = d_model
        self.input_projection = nn.Linear(in_dim,d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=5000)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation=activation,
            dtype=torch.float32
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.output_projection = nn.Linear(d_model, 1368)  #映射回所需长度
        #生成预测帧
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():#遍历模型内所有参数
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
    def forward(self,src):
        """
        :param src: 输入张量，形状为（batch_size, history_len, input_frame_size），（batch,24,500）
        :return: 输出张量，形状为（batch_size, predict_len, input_frame_size），（batch,1,500）
        """
        batch_size = src.shape[0]
        #print("原始输入前两步对比：", torch.allclose(src[0, 0, :], src[0, 1, :]))
        # 1. 输入映射: (batch, 24, 500) -> (batch, 24, d_model)
        src = self.input_projection(src)
        #print("投影后前两步对比：", torch.allclose(src[0, 0, :], src[0, 1, :]))
        # 2. 乘以sqrt(d_model)并加上位置编码
        src = src*math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        #print("加位置编码后前两步对比：", torch.allclose(src[0, 0, :], src[0, 1, :]))
        # 3. Transformer Encoder
        # 注意: transformer期望输入 (batch, seq, feature)
        #test=torch.randn_like(src)#随机生成一个0-1张量或-1-1都会触发错误
        memory = self.transformer_encoder(src)
        memory = memory.permute(1, 0, 2)

        #print("Encoder后前两步对比：", torch.allclose(memory[0,0,:], memory[0,1,:]))# (batch, 24, d_model)
        # 4. 时间维度压缩: 从24帧生成1帧
        # 先转置: (batch, d_model, 24)
        last_step_hidden = memory[:,-1,:]  #取出最后1步
        #取最后一个时间步
        # 5. 输出映射: (batch, 1, d_model) -> (batch, 1 , 500)
        output = self.output_projection(last_step_hidden)
        output = output.view(batch_size, self.predict_len, -1)  # 动态适配predict_len
        # output(24,1,1368)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self,x):
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)


class Test1(nn.Module):
    def __init__(self,
                 in_dim,
                 d_model,#模型维度
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,  #feedforward维度
                 activation="relu",#激活函数
                 dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.d_model = d_model
        self.input_projection = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=5000)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
    def _reset_parameters(self):
        for p in self.parameters():  # 遍历模型内所有参数
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        src = self.input_projection(src)
        # print("投影后前两步对比：", torch.allclose(src[0, 0, :], src[0, 1, :]))
        # 2. 乘以sqrt(d_model)并加上位置编码
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # print("加位置编码后前两步对比：", torch.allclose(src[0, 0, :], src[0, 1, :]))
        # 3. Transformer Encoder
        # 注意: transformer期望输入 (batch, seq, feature)
        # test=torch.rand_like(src)
        a = torch.randn_like(src)

        memory = self.transformer_encoder(a)
        return memory

def main_test():
    c =torch.rand(24,24,1024)
    test11 = Test1(d_model=512,in_dim=1024)
    b = test11(c)
    a = b
if __name__ == '__main__':
    # a = torch.randn(1,24,500)
    # Transformer = TecPreTransformer(in_dim = 500,history_len = 24,predict_len = 1,d_model = 512)
    # print(Transformer(a).shape)
    main_test()

