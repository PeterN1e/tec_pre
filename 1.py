###################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple, Optional


class SimpleTransformerTECPredictor(nn.Module):
    def __init__(self,
                 input_dim: int = 512,  # 编码后的TEC特征维度
                 meta_dim: int = 3,  # 元数据维度 (f10.7, ssn, dst)
                 hidden_dim: int = 512,  # Transformer隐藏维度
                 num_heads: int = 8,  # 注意力头数
                 num_layers: int = 3,  # Transformer层数
                 dropout: float = 0.1,  # Dropout概率
                 seq_length: int = 24,  # 输入序列长度
                 pred_length: int = 6):  # 预测序列长度
        super().__init__()

        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.pred_length = pred_length

        # 输入投影层：将TEC特征和元数据映射到相同的隐藏空间
        self.tec_projection = nn.Linear(input_dim, hidden_dim)
        self.meta_projection = nn.Linear(meta_dim, hidden_dim)

        # 位置编码
        self.position_encoding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))

        # 使用官方Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # 输出预测头
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # 层标准化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                tec_features: torch.Tensor,
                meta_data: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            tec_features: [batch_size, seq_length, input_dim] 编码后的TEC特征
            meta_data: [batch_size, seq_length, meta_dim] 元数据 (f10.7, ssn, dst)
            src_mask: [seq_length, seq_length] 源序列掩码
            src_key_padding_mask: [batch_size, seq_length] 源序列填充掩码

        Returns:
            [batch_size, pred_length, input_dim] 预测的TEC特征序列
        """
        batch_size = tec_features.size(0)

        # 1. 特征投影
        tec_embedded = self.tec_projection(tec_features)  # [B, L, H]
        meta_embedded = self.meta_projection(meta_data)  # [B, L, H]

        # 2. 特征融合：将TEC特征和元数据相加
        combined_input = tec_embedded + meta_embedded  # [B, L, H]

        # 3. 添加位置编码
        combined_input = combined_input + self.position_encoding
        combined_input = self.dropout_layer(combined_input)
        combined_input = self.layer_norm(combined_input)

        # 4. Transformer编码
        memory = self.transformer_encoder(
            combined_input,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # [B, L, H]

        # 5. 预测未来序列（使用最后一个时间步的编码作为预测起点）
        predictions = []
        current_memory = memory[:, -1:, :]  # [B, 1, H]

        for i in range(self.pred_length):
            # 预测当前时间步
            pred = self.output_projection(current_memory)  # [B, 1, D]
            predictions.append(pred)

            # 更新记忆（可以使用多种策略）
            if i < self.pred_length - 1:
                # 策略1：使用当前预测作为下一个时间步的输入
                current_pred_embedded = self.tec_projection(pred)
                current_memory = current_pred_embedded

        # 6. 拼接所有预测结果
        predictions = torch.cat(predictions, dim=1)  # [B, P, D]

        return predictions

    def predict(self,
                tec_features: torch.Tensor,
                meta_data: torch.Tensor) -> torch.Tensor:
        """
        预测函数，用于推理

        Args:
            tec_features: 输入TEC特征
            meta_data: 输入元数据

        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(tec_features, meta_data)


class AdvancedTransformerTECPredictor(nn.Module):
    """高级版本，支持更复杂的特征融合和预测策略"""

    def __init__(self,
                 input_dim: int = 512,
                 meta_dim: int = 3,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 seq_length: int = 24,
                 pred_length: int = 6):
        super().__init__()

        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.pred_length = pred_length

        # 输入投影层
        self.tec_projection = nn.Linear(input_dim, hidden_dim)
        self.meta_projection = nn.Linear(meta_dim, hidden_dim)

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        # 位置编码
        self.position_encoding = nn.Parameter(torch.zeros(1, seq_length + pred_length, hidden_dim))

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出预测头
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # 预测位置的嵌入
        self.pred_position_embedding = nn.Parameter(torch.zeros(1, pred_length, hidden_dim))

        # Dropout和层标准化
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                tec_features: torch.Tensor,
                meta_data: torch.Tensor,
                future_meta: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            tec_features: [batch_size, seq_length, input_dim] 编码后的TEC特征
            meta_data: [batch_size, seq_length, meta_dim] 历史元数据
            future_meta: [batch_size, pred_length, meta_dim] 未来元数据（可选）
            src_mask: 源序列掩码
            src_key_padding_mask: 源序列填充掩码

        Returns:
            [batch_size, pred_length, input_dim] 预测的TEC特征序列
        """
        batch_size = tec_features.size(0)

        # 1. 特征投影
        tec_embedded = self.tec_projection(tec_features)  # [B, L, H]
        meta_embedded = self.meta_projection(meta_data)  # [B, L, H]

        # 2. 特征融合
        combined_input = self.fusion_layer(
            torch.cat([tec_embedded, meta_embedded], dim=-1)
        )  # [B, L, H]

        # 3. 处理未来元数据（如果提供）
        if future_meta is not None:
            future_meta_embedded = self.meta_projection(future_meta)  # [B, P, H]
            # 创建预测位置的嵌入
            pred_embeddings = future_meta_embedded + self.pred_position_embedding
            # 组合历史和预测输入
            combined_sequence = torch.cat([combined_input, pred_embeddings], dim=1)  # [B, L+P, H]
        else:
            # 如果没有未来元数据，只使用历史数据
            combined_sequence = combined_input  # [B, L, H]

        # 4. 添加位置编码
        pos_encoding = self.position_encoding[:, :combined_sequence.size(1), :]
        combined_sequence = combined_sequence + pos_encoding
        combined_sequence = self.dropout_layer(combined_sequence)
        combined_sequence = self.layer_norm(combined_sequence)

        # 5. Transformer编码
        memory = self.transformer_encoder(
            combined_sequence,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # [B, L+P, H] 或 [B, L, H]

        # 6. 获取预测结果
        if future_meta is not None:
            # 如果使用了未来元数据，直接取预测部分的输出
            pred_memory = memory[:, -self.pred_length:, :]  # [B, P, H]
        else:
            # 否则使用自回归方式预测
            pred_memory = []
            current_memory = memory[:, -1:, :]  # [B, 1, H]

            for i in range(self.pred_length):
                pred_embedded = self.output_projection(current_memory)  # [B, 1, D]
                pred_embedded = self.tec_projection(pred_embedded)  # [B, 1, H]

                if i < self.pred_length - 1:
                    pred_memory.append(current_memory)
                    current_memory = pred_embedded

            pred_memory = torch.cat(pred_memory, dim=1)  # [B, P, H]

        # 7. 投影到输出维度
        predictions = self.output_projection(pred_memory)  # [B, P, D]

        return predictions

    def predict(self,
                tec_features: torch.Tensor,
                meta_data: torch.Tensor,
                future_meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测函数

        Args:
            tec_features: 输入TEC特征
            meta_data: 历史元数据
            future_meta: 未来元数据（可选）

        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(tec_features, meta_data, future_meta)


# 工具函数
def create_transformer_mask(seq_length: int, pred_length: int = 0,
                            device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    创建Transformer掩码

    Args:
        seq_length: 序列长度
        pred_length: 预测长度
        device: 设备

    Returns:
        掩码张量
    """
    total_length = seq_length + pred_length
    mask = torch.zeros(total_length, total_length, device=device)

    # 为预测部分创建后续掩码
    if pred_length > 0:
        for i in range(seq_length, total_length):
            mask[i, :i] = float('-inf')

    return mask


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transformer_model(config: dict) -> nn.Module:
    """
    根据配置创建Transformer模型

    Args:
        config: 模型配置字典

    Returns:
        Transformer模型
    """
    model_type = config.get('model_type', 'simple')

    if model_type == 'advanced':
        model = AdvancedTransformerTECPredictor(
            input_dim=config.get('input_dim', 512),
            meta_dim=config.get('meta_dim', 3),
            hidden_dim=config.get('hidden_dim', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            seq_length=config.get('seq_length', 24),
            pred_length=config.get('pred_length', 6)
        )
    else:
        model = SimpleTransformerTECPredictor(
            input_dim=config.get('input_dim', 512),
            meta_dim=config.get('meta_dim', 3),
            hidden_dim=config.get('hidden_dim', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            seq_length=config.get('seq_length', 24),
            pred_length=config.get('pred_length', 6)
        )

    return model


if __name__ == '__main__':
    # 示例用法
    config = {
        'input_dim': 512,
        'meta_dim': 3,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'seq_length': 24,
        'pred_length': 6,
        'model_type': 'simple'
    }

    model = get_transformer_model(config)

    # 示例输入
    batch_size = 32
    tec_features = torch.randn(batch_size, config['seq_length'], config['input_dim'])
    meta_data = torch.randn(batch_size, config['seq_length'], config['meta_dim'])

    # 前向传播
    output = model(tec_features, meta_data)

    print(f"Input shape: {tec_features.shape}")
    print(f"Meta data shape: {meta_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {count_parameters(model):,}")