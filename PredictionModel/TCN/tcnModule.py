import  torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainConfig
class Chomp1d(nn.Module):
    def __init__(self,chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self,x):
        return x[:,:,:-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self,kernel_size,stride,n_inputs,n_outputs,padding,dilation,dropout):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(kernel_size=kernel_size,
                                  stride = stride,
                                  in_channels=n_inputs,
                                  out_channels=n_outputs,
                                  padding=padding,
                                  dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv1d_2 = nn.Conv1d(kernel_size=kernel_size,
                                  in_channels=n_outputs,
                                  out_channels=n_outputs,
                                  padding=padding,
                                  stride=stride,
                                  dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv_net = nn.Sequential(
            self.conv1d_1,self.chomp1,self.relu1,self.dropout1,
            self.conv1d_2,self.chomp2,self.relu2,self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs,n_outputs,1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._init_weights()
    def _init_weights(self):
        self.conv1d_1.weight.data.normal_(0, 0.01)
        self.conv1d_2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self,x):
        residual = x  # (batch, n_inputs, seq)

        # 主路径通过两个卷积层
        out = self.conv_net(x)  # (batch, n_outputs, seq)

        # 如果输入输出维度不同，对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(residual)  # (batch, n_inputs, seq) -> (batch, n_outputs, seq)

        # 残差连接：主路径输出 + 残差路径
        return self.relu(out + residual)

class TemporalConvNet(nn.Module):
    def __init__(self,num_inputs,num_channels,kernel_size=2,dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            padding = (kernel_size - 1)*dilation_size

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,  # 输入通道
                    n_outputs=out_channels,
                    kernel_size = kernel_size,
                    dropout = dropout,
                    dilation = dilation_size,
                    padding = padding,
                    stride = 1
                )
            )
        self.network = nn.Sequential(*layers)
    def forward(self,x):
        return self.network(x)


class TCNMiddlePredictor(nn.Module):

    def __init__(self,
                 predict_len = 1,
                 input_dim = 4104,
                 output_dim = 4104,
                 history_len = 12,
                 num_channels=None,
                 kernel_size = 3,
                 dropout = 0.2,
                 use_attention = False):
        super().__init__()
        if num_channels is None:
            num_channels = [256, 256, 256]
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.seq_length = history_len
        self.use_attention = use_attention
        self.predict_len = predict_len
        self.tcn = TemporalConvNet(
            num_inputs = input_dim,
            num_channels = num_channels,
            kernel_size = kernel_size,
            dropout=dropout
        )
        tcn_output_dim = num_channels[-1]

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(tcn_output_dim,64),
                nn.Tanh(),
                nn.Linear(64,1)
            )
            agg_dim = tcn_output_dim
        else:
            agg_dim = tcn_output_dim

        hidden_dim1 = (tcn_output_dim + output_dim)//2
        hidden_dim2 = (hidden_dim1 + output_dim)//2

        self.projection = nn.Sequential(
            # ----- 第一层：TCN输出 -> 中间维度1 -----
            nn.Linear(tcn_output_dim, hidden_dim1),  # 256 -> 2180
            nn.LayerNorm(hidden_dim1),  # 层归一化，稳定训练
            nn.ReLU(),  # 非线性激活
            nn.Dropout(dropout),  # 正则化

            # ----- 第二层：中间维度1 -> 中间维度2 -----
            nn.Linear(hidden_dim1, hidden_dim2),  # 2180 -> 3142
            nn.LayerNorm(hidden_dim2),  # 层归一化
            nn.ReLU(),  # 非线性激活
            nn.Dropout(dropout),  # 正则化

            # ----- 输出层：映射到目标维度 -----
            nn.Linear(hidden_dim2, output_dim)  # 3142 -> 4104
            # 注意：输出层不加激活函数，因为是回归任务（预测连续值）
            # 也不加LayerNorm，让解码器接收原始尺度的特征
        )
        # ==================== 残差连接 ====================
        # 如果TCN输出维度和目标维度差异大，可以用1x1线性投影做残差
        # 这里256->4104维度变化大，投影残差可能帮助梯度流动
        self.skip_connection = nn.Linear(tcn_output_dim, output_dim) \
            if tcn_output_dim != output_dim else None

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)

        tcn_out = self.tcn(x)

        if self.use_attention:
            tcn_out_perm=  tcn_out.permute(0,2,1)

            attn_scores = self.attention(tcn_out_perm)
            attn_weights = F.softmax(attn_scores, dim=1)
            context = torch.sum(attn_weights * tcn_out_perm, dim=1)

        else:

            context = tcn_out[:,:,-1]

        out = self.projection(context)

        if self.skip_connection is not None:
            residual = self.skip_connection(context)
            out = out + residual
        output = out.view(batch_size,self.predict_len, 12, 18, 19 )

        return output



if __name__ == '__main__':
    a = torch.randn(24, 12, 4104)
    b = TCNMiddlePredictor(input_dim=4104, output_dim=4104,predict_len =12, history_len=36)
    print(b(a).shape)
