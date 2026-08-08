import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

class STLSTMCell(nn.Module):
    def __init__(self,input_channels,hidden_channels,kernel_size = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size//2

        #输入到状态的卷积
        self.conv_x = nn.Conv2d(input_channels,
                                hidden_channels * 4,
                                kernel_size,
                                padding = padding,
                                bias = True)
        #状态到状态的卷积
        self.conv_h = nn.Conv2d(hidden_channels ,
                                hidden_channels * 4,
                                kernel_size,
                                padding = padding,
                                bias = True)
        #时空记忆M的卷积
        self.conv_m = nn.Conv2d(hidden_channels * 2,
                                hidden_channels * 4,
                                kernel_size,
                                padding = padding,
                                bias = True)
        #输出门的额外卷积, 用于结合时空记忆M来生成最终隐藏状态
        self.conv_o = nn.Conv2d(hidden_channels * 2,
                                hidden_channels,
                                kernel_size,
                                padding = padding,
                                bias = True)
        self.norm = nn.LayerNorm([hidden_channels])

    def forward(self,x_t,h_prev,c_prev,m_prev):
        gates_x = self.conv_x(x_t)  # [B, 4*C, H, W]
        gates_h = self.conv_h(h_prev)# [B, 4*C, H, W]
        gates_m = self.conv_m(m_prev)# [B, 4*C, H, W]

        gates = gates_x + gates_h + gates_m

        i,f,g,o = torch.chunk(gates,4,dim = 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(o)

        c_t = f * c_prev + i * g
        m_t = f *m_prev +i * g
        h_t = o * torch.tanh(self.conv_o(torch.cat([c_t,m_t],dim = 1)))
        return h_t,c_t,m_t

class Predrnn(nn.Module):
    def __init__(self,input_channels,hidden_channels,input_length,output_length,num_layers):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_length = input_length
        self.num_layers = num_layers
        self.output_length = output_length

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.input_channels if i == 0 else self.hidden_channels
            self.cells.append(STLSTMCell(in_ch,hidden_channels))

            self.output_conv = nn.Conv2d(hidden_channels,
                                         self.output_length,
                                         kernel_size = 1,
                                         bias = True)
    def forward(self,input_seq,return_hidden = False):

        B,T,C,H,W = input_seq.shape
        h_list = [torch.zeros(B,self.hidden_channels,H,W,device=input_seq.device)
                  for _ in range(self.num_layers)]
        """
        # 列表推导式
        h_list = [torch.zeros(...) for _ in range(self.num_layers)]
        # 完全等价于下面普通for循环
        h_list = []
        for _ in range(self.num_layers):
            tensor = torch.zeros(B,self.hidden_channels,H,W,device=input_seq.device)
            h_list.append(tensor)
        """
        c_list = [torch.zeros(B, self.hidden_channels, H, W, device=input_seq.device)
                  for _ in range(self.num_layers)]
        m_list = [torch.zeros(B, self.hidden_channels, H, W, device=input_seq.device)
                  for _ in range(self.num_layers)]
        prediction = []
        all_hidden_states = []
        #编码阶段+解码阶段
        total_steps = self.input_length + self.output_length

        for t in range(total_steps):
            if t < self.input_length:
                # 处于编码阶段
                x_t = input_seq[:,t,:,:]  #[B,C,H,W]
            else:
                # 处于解码阶段
                x_t = pred
            current_input = x_t
            for layer_idx in range(self.num_layers):
                # " \ " 行连接符（续行符）
                h_list[layer_idx], c_list[layer_idx],m_list[layer_idx] = \
                    self.cells[layer_idx](current_input,h_list[layer_idx],m_list[layer_idx])

                current_input = h_list[layer_idx]
            last_hidden = h_list[-1]

            # 收集解码阶段的隐藏状态 (用于注意力模块)
            if return_hidden:
                all_hidden_states.append(last_hidden)

            #只在解码阶段生成预测
            if t >= self.input_length:
                pred = self.output_conv(last_hidden)
                prediction.append(pred)
        #堆叠预测结果
        predictions = torch.cat(prediction,dim = 1)

        if return_hidden:
            all_hidden_states = torch.cat(all_hidden_states,dim = 1)
            return predictions,all_hidden_states
        return predictions
class HaloAttention(nn.Module):
    def __init__(self,dim,block_size = 8,halo_size = 3,
                 num_heads= 4,dim_head = 32):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.halo_size = halo_size
        self.block_size = block_size
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        #QKV投影
        self.to_q = nn.Linear(dim,inner_dim,bias = False)
        self.to_kv = nn.Linear(dim,inner_dim*2,bias = False)
        self.to_out = nn.Linear(inner_dim,dim,bias = False)

        #位置编码 (相对位置偏置)
        self.rel_bias = nn.Parameter(torch.zeros(num_heads,
                                                 (block_size * 2 * halo_size)**2,
                                                 (block_size * 2 * halo_size)**2)
                                     )
        def forward(self,x):
            B,C,H,W = x.shape
            bs = self.block_size
            hs = self.halo_size
            padded_bs = bs * 2 * hs  #扩展后的块大小

            #1 Padding（确保，H，W可被block_size整除）
            pad_h = (bs - H % bs) % bs
            pad_w = (bs - W % bs) % bs
            x_padded = F.pad(x,(0,pad_w,0,pad_h))
            _,_,H_p,W_p = x_padded.shape
            #2: Halo Padding (光晕填充) ----
            x_halo = F.pad(x_padded,(hs,hs,hs,hs))
            #3: 提取所有扩展后的块
            num_blocks_h = H_p // bs
            num_blocks_w = W_p // bs
            # 使用unfold操作提取所有块
            # 每个块的大小为 padded_bs × padded_bs
            blocks = []
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # 计算在padded特征图中的起止坐标
                    h_start = i * bs
                    w_start = j * bs
                    #在halo-padded特征图中, 坐标偏移了hs
                    block = x_halo[:,:,
                                    h_start:h_start+padded_bs,
                                    w_start:w_start+padded_bs]
                    blocks.append(block)
            # [B, num_blocks, C, padded_bs, padded_bs]
            blocks = torch.stack(blocks,dim = 1)
            num_blocks = num_blocks_h * num_blocks_w

            #4 计算局部自注意力 ----
            # Reshape: [B * num_blocks, C, padded_bs^2]
            blocks_flat = blocks.reshape(B * num_blocks, C,-1).permute(0,2,1)
            # blocks_flat: [B*num_blocks, num_tokens, C]
            # QKV投影
            q = self.to_q(blocks_flat) # [B*N, T, inner_dim]
            kv = self.to_kv(blocks_flat)
            k,v = kv.chunk(2, dim = -1)
            #多头注意力
            T = blocks_flat.shape[1]
            q = q.reshape(B * num