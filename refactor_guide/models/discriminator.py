# models/discriminator.py - 判别器模型定义

import torch.nn as nn

class ResidualBlockCond(nn.Module):
    """条件残差块"""
    def __init__(self, in_features, cond_dim, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features + cond_dim, out_features)
        self.relu = nn.ReLU(True)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.relu(self.linear(x))


class ConditionalDiscriminator(nn.Module):
    """条件判别器：接收曲线curve、季节条件c和预测数据forecast"""

    def __init__(self, season_dim=4, forecast_dim=96, hidden=256):
        super().__init__()
        # 输入维度：曲线(96) + 季节(4) + 预测(96)
        self.input = nn.Linear(96 + season_dim + forecast_dim, hidden)
        self.res1 = ResidualBlockCond(hidden, season_dim, hidden)
        self.res2 = ResidualBlockCond(hidden, season_dim, hidden)
        self.output = nn.Linear(hidden, 1)
        self.aux_cls = nn.Linear(hidden, season_dim)

    def forward(self, curve, c, forecast):
        x = torch.cat([curve, c, forecast], dim=1)
        x = self.input(x)
        x = self.res1(x, c)
        x = self.res2(x, c)
        adv_out = self.output(x)
        aux_out = self.aux_cls(x)
        return adv_out, aux_out


class Discriminator(nn.Module):
    """基础判别器（不使用预测数据）"""
    
    def __init__(self, season_dim=4, hidden=256):
        super().__init__()
        self.input = nn.Linear(96 + season_dim, hidden)
        self.res1 = ResidualBlockCond(hidden, season_dim, hidden)
        self.res2 = ResidualBlockCond(hidden, season_dim, hidden)
        self.output = nn.Linear(hidden, 1)
        self.aux_cls = nn.Linear(hidden, season_dim)

    def forward(self, curve, c):
        x = torch.cat([curve, c], dim=1)
        x = self.input(x)
        x = self.res1(x, c)
        x = self.res2(x, c)
        adv_out = self.output(x)
        aux_out = self.aux_cls(x)
        return adv_out, aux_out
