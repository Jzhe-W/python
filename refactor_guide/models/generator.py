# models/generator.py - 生成器模型定义

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
            nn.Linear(out_features, out_features)
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.block(x) + self.shortcut(x))


class ConditionalGenerator(nn.Module):
    """条件生成器：接收噪声z、季节条件c和预测数据forecast"""

    def __init__(self, z_dim=100, season_dim=4, forecast_dim=96, hidden=512):
        super().__init__()
        self.z_dim = z_dim
        self.season_dim = season_dim
        self.forecast_dim = forecast_dim
        self.hidden = hidden

        # 季节嵌入层
        self.season_embedding = nn.Embedding(season_dim, hidden // 4)

        # 预测数据处理层
        self.forecast_proj = nn.Sequential(
            nn.Linear(forecast_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.LeakyReLU(0.2)
        )

        # 噪声和条件融合
        self.input = nn.Linear(z_dim + hidden // 4 + hidden // 2, hidden)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden, hidden, dropout=0.0) for _ in range(3)
        ])

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden, 96),
            nn.Tanh()
        )

        # 季节特定的输出调整
        self.season_output_adjust = nn.ModuleList([
            nn.Linear(96, 96) for _ in range(4)
        ])

        # 季节特定的输出范围调整
        self.season_range_adjust = nn.Parameter(torch.tensor([
            [0.8, 1.2],  # Spring
            [0.2, 0.5],  # Summer: 低风期
            [0.8, 1.2],  # Autumn
            [0.6, 0.9]   # Winter
        ], dtype=torch.float32))

    def forward(self, z, c, forecast, training_step=0):
        batch_size = z.size(0)
        season_idx = c.argmax(dim=1)
        season_emb = self.season_embedding(season_idx)

        # 处理预测数据
        forecast_emb = self.forecast_proj(forecast)

        # 融合所有条件
        x = torch.cat([z, season_emb, forecast_emb], dim=1)
        x = self.input(x)

        # 残差处理
        for res_block in self.res_blocks:
            x = res_block(x)

        # 生成基础输出
        x = self.output(x)

        # 季节特定的输出调整 - 向量化版本
        season_adjustments = torch.stack([self.season_output_adjust[i].weight for i in range(4)], dim=0)
        season_biases = torch.stack([self.season_output_adjust[i].bias for i in range(4)], dim=0)

        selected_weights = season_adjustments[season_idx]
        selected_biases = season_biases[season_idx]

        adjusted_outputs = torch.einsum('bi,bij->bj', x, selected_weights) + selected_biases

        # 应用季节特定的范围调整
        season_range_factors = self.season_range_adjust[season_idx]
        min_factor = season_range_factors[:, 0:1]
        max_factor = season_range_factors[:, 1:2]

        adjusted_outputs = adjusted_outputs * (max_factor - min_factor) / 2 + (max_factor + min_factor) / 2

        return torch.clamp(adjusted_outputs, -1, 1)


class Generator(nn.Module):
    """基础生成器（不使用预测数据）"""
    
    def __init__(self, z_dim=100, season_dim=4, hidden=512):
        super().__init__()
        self.z_dim = z_dim
        self.season_dim = season_dim
        self.hidden = hidden

        self.season_embedding = nn.Embedding(season_dim, hidden)
        self.input = nn.Linear(z_dim + hidden, hidden)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden, hidden, dropout=0.0) for _ in range(3)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden, 96),
            nn.Tanh()
        )

        self.season_output_adjust = nn.ModuleList([
            nn.Linear(96, 96) for _ in range(4)
        ])

    def forward(self, z, c, training_step=0):
        batch_size = z.size(0)
        season_idx = c.argmax(dim=1)
        season_emb = self.season_embedding(season_idx)

        x = torch.cat([z, season_emb], dim=1)
        x = self.input(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.output(x)

        # 季节特定的输出调整
        season_adjustments = torch.stack([self.season_output_adjust[i].weight for i in range(4)], dim=0)
        season_biases = torch.stack([self.season_output_adjust[i].bias for i in range(4)], dim=0)

        selected_weights = season_adjustments[season_idx]
        selected_biases = season_biases[season_idx]

        adjusted_outputs = torch.einsum('bi,bij->bj', x, selected_weights) + selected_biases

        return torch.clamp(adjusted_outputs, -1, 1)
