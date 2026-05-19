import torch.nn as nn
import math
from config import device
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels =  input_dim+hidden_dim,
            out_channels = 4*input_dim,
            kernel_size=3,
            padding=1)
    def forward(self,x,h_prev,c_prev):
        combined = torch.cat([x,h_prev],dim = 1)
        combined_conv = self.conv(combined)
        cc_i,cc_f,cc_g,cc_o = torch.chunk(combined_conv,4,dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)

        c_next = c_prev*f +i*g
        h_next = o*torch.tanh(c_next)

        return h_next,c_next


class ConvLSTM(nn.Module):
    def __init__(self,in_channels,hidden_channels,pred_len=1):
        super().__init__()
        self.pred_len = pred_len
        self.conv_lstm_cell = ConvLSTMCell(input_dim=in_channels,hidden_dim=hidden_channels)

    def forward(self,x):
        batch, seq_len, _, h, w = x.shape
        hidden = torch.zeros(batch, 12, h, w, device=x.device)
        cell = torch.zeros(batch, 12, h, w, device=x.device)

        for i in range(seq_len):
            hidden,cell = self.conv_lstm_cell(x[:,i],hidden,cell)

        outputs = []
        for _ in range(self.pred_len):
            hidden,cell = self.conv_lstm_cell(hidden,hidden,cell)
            outputs.append(hidden.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs[:,0,:,:,:]

        return outputs

if __name__ == '__main__':
    conv_lstm_test = ConvLSTM(12,12)
    a = torch.randn(24,12,12,18,19)
    b= conv_lstm_test(a)
    print(b.shape)
