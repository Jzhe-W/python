# 快速修复版本2：统一季节处理策略
# 替换原代码中的季节特殊处理部分

# ========== 移除过度的季节特殊处理 ==========

def unified_season_noise(batch_size, z_dim, noise_sigma, epoch, total_epochs, device):
    """
    统一的季节噪声生成策略
    
    关键改进：
    1. 移除季节特殊处理（Summer 2.5x, Winter 2.0x等）
    2. 使用动态噪声调整（基于训练进度）
    3. 所有季节使用统一策略
    """
    # 动态噪声调整：早期高噪声，后期低噪声
    progress = epoch / total_epochs
    # 使用余弦退火策略
    dynamic_noise = noise_sigma * (0.5 + 0.5 * np.cos(np.pi * progress))
    
    # 生成噪声
    z = torch.randn(batch_size, z_dim, device=device) * dynamic_noise
    
    return z


def remove_excessive_season_constraints(gen_curves, season_vec, weight=2.0):
    """
    简化的季节约束（移除过度的硬编码约束）
    
    只保留最基本的物理约束：
    1. 出力范围 [0, 1]
    2. Summer整体偏低（但不强制）
    """
    season_idx = season_vec.argmax(dim=1)
    
    # 1. 基本出力范围约束
    range_penalty = torch.relu(-gen_curves) + torch.relu(gen_curves - 1.0)
    loss = range_penalty.mean()
    
    # 2. 温和的Summer约束（不强制）
    summer_mask = (season_idx == 1)
    if summer_mask.any():
        summer_curves = gen_curves[summer_mask]
        # 只约束均值，不约束峰值和低出力比例
        mean_output = summer_curves.mean()
        # 温和约束：只在均值>0.5时才惩罚
        mean_penalty = torch.relu(mean_output - 0.5) * 0.5  # 降低权重
        loss += mean_penalty
    
    return weight * loss


# ========== 使用方法 ==========
"""
# 1. 替换原代码中的噪声生成（约第1150行）：

# 原代码：
season_idx = season_vec.argmax(dim=1)
summer_mask = (season_idx == 1)
if summer_mask.any():
    z = torch.randn(...) * (noise_sigma * 2.5) + noise_mu
elif winter_mask.any():
    z = torch.randn(...) * (noise_sigma * 2.0) + noise_mu
else:
    z = torch.randn(...) * (noise_sigma * 0.8) + noise_mu

# 替换为：
z = unified_season_noise(batch_size, z_dim, noise_sigma, epoch, epochs, device)


# 2. 替换原代码中的季节约束（约第1300行）：

# 原代码：
wind_physical_loss_v2 = wind_power_physical_loss(gen_curves, season_vec, weight=5.0)
wind_constraint = wind_summer_constraint(gen_curves, season_vec)
total_loss_G += 5.0 * wind_physical_loss_v2 + 10.0 * wind_constraint

# 替换为：
simple_constraint = remove_excessive_season_constraints(gen_curves, season_vec, weight=2.0)
total_loss_G += simple_constraint
"""


# ========== 完整的简化训练循环示例 ==========
"""
for epoch in range(1, epochs + 1):
    for batch_idx, (real_curves, forecast_curves, season_vec) in enumerate(dataloader):
        real_curves, forecast_curves, season_vec = real_curves.to(device), forecast_curves.to(device), season_vec.to(device)
        batch_size = real_curves.size(0)
        
        # === 更新判别器 ===
        for _ in range(n_critic):
            # 统一噪声策略
            z = unified_season_noise(batch_size, z_dim, noise_sigma, epoch, epochs, device)
            fake_curves = G(z, season_vec, forecast_curves, 0).detach()
            
            d_real, aux_real = D(real_curves, season_vec, forecast_curves)
            d_fake, aux_fake = D(fake_curves, season_vec, forecast_curves)
            
            loss_D = d_fake.mean() - d_real.mean()
            gp = gradient_penalty(D, real_curves, fake_curves, season_vec, forecast_curves)
            aux_loss = F.cross_entropy(aux_real, season_vec.argmax(dim=1))
            
            opt_D.zero_grad()
            (loss_D + lambda_gp * gp + aux_loss).backward()
            opt_D.step()
        
        # === 更新生成器 ===
        # 统一噪声策略
        z = unified_season_noise(batch_size, z_dim, noise_sigma, epoch, epochs, device)
        gen_curves = G(z, season_vec, forecast_curves, 0)
        
        d_gen, aux_gen = D(gen_curves, season_vec, forecast_curves)
        loss_G = -d_gen.mean()
        aux_loss_G = F.cross_entropy(aux_gen, season_vec.argmax(dim=1))
        
        # 使用平衡的损失函数（来自quick_fix_v1.py）
        total_loss_G = balanced_loss_function(
            gen_curves, real_curves, season_vec,
            loss_G, aux_loss_G, d_gen, aux_gen,
            USE_LIGHTWEIGHT_ACF=True
        )
        
        opt_G.zero_grad()
        total_loss_G.backward()
        opt_G.step()
"""
