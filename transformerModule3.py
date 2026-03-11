import  torch
import torch.nn as nn
import math


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
                 dropout=0.1):
        super( ).__init__()
        self.predict_len = predict_len
        self.d_model = d_model
        self.input_projection = nn.Linear(in_dim,d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=history_len+10)

        encoder_layer= nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,    #dim_feedforward = 4 × d_model
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
            dtype=torch.float32
        )

        self.output_activation = nn.Tanh()
        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
                norm=nn.LayerNorm(d_model))

        self.output_projection = nn.Linear(d_model, 4104)  #映射回所需长度
        # self.output_projection = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, in_dim * predict_len)
        # )
    def forward(self,src):

        batch_size = src.shape[0]

        src = self.input_projection(src)
        #print("投影后前两步对比：", torch.allclose(src[0, 0, :], src[0, 1, :]))
        # 2. 乘以sqrt(d_model)并加上位置编码
        src = src*math.sqrt(self.d_model)

        src = self.pos_encoder(src)

        src = self.transformer_encoder(src)

        last_step_hidden = src[:,-1,:]
        # 5. 输出映射: (batch, 1, d_model) -> (batch , 500)
        #last_step_hidden = last_step_hidden.reshape(batch_size,self.predict_len,-1)
        output = self.output_projection(last_step_hidden)
        output = output.view(batch_size, 12,18,19)  # 动态适配predict_len
        # output(24,1,4104)
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
        if x.size(1)>self.pe.size(1):
            raise ValueError(f"输入序列长度{x.size(1)}超过位置编码最大长度{self.pe.size(1)}")
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)




if __name__ == '__main__':
     a = torch.randn(1,24,500)
    # Transformer = TecPreTransformer(in_dim = 500,history_len = 24,predict_len = 1,d_model = 512)
    # print(Transformer(a).shape)

