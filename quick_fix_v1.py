# 快速修复版本1：重新平衡损失权重
# 将这段代码替换原代码中的"total_loss_G"部分（约第1200行附近）

# ========== 优化后的损失函数（平衡版） ==========
def balanced_loss_function(gen_curves, real_curves, season_vec, 
                          loss_G, aux_loss_G, d_gen, aux_gen,
                          USE_LIGHTWEIGHT_ACF=True):
    """
    平衡的损失函数：解决过度追求相似性的问题
    
    关键改进：
    1. 降低相似性权重（5.0 → 2.0）
    2. 提高多样性权重（0.5 → 1.5）
    3. 恢复轻量级ACF约束
    4. 平衡各类损失的贡献
    """
    
    # === 核心WGAN损失 ===
    total_loss = loss_G + aux_loss_G
    
    # === 相似性损失（适度降低权重） ===
    # 1. 逐点MSE
    pointwise_loss = F.mse_loss(gen_curves, real_curves)
    total_loss += 1.5 * pointwise_loss  # 3.0 → 1.5
    
    # 2. 增强相似性
    mse_loss = F.mse_loss(gen_curves, real_curves)
    total_loss += 2.0 * mse_loss  # 5.0 → 2.0
    
    # 3. 量级匹配
    fake_mean = torch.mean(gen_curves)
    fake_std = torch.std(gen_curves)
    real_mean = torch.mean(real_curves)
    real_std = torch.std(real_curves)
    mag_loss = torch.abs(fake_mean - real_mean) / (real_mean + 1e-8) + \
               torch.abs(fake_std - real_std) / (real_std + 1e-8)
    total_loss += 2.0 * mag_loss  # 5.0 → 2.0
    
    # === 多样性损失（适度提高权重） ===
    # 1. 季节间多样性
    season_idx = season_vec.argmax(dim=1)
    season_means = torch.zeros(4, gen_curves.size(1), device=gen_curves.device)
    season_counts = torch.zeros(4, device=gen_curves.device)
    
    for season in range(4):
        mask = (season_idx == season)
        if mask.any():
            season_means[season] = gen_curves[mask].mean(dim=0)
            season_counts[season] = mask.sum().float()
    
    valid_seasons = season_counts > 0
    if valid_seasons.sum() >= 2:
        valid_means = season_means[valid_seasons]
        diff_matrix = valid_means.unsqueeze(1) - valid_means.unsqueeze(0)
        l1_diffs = torch.mean(torch.abs(diff_matrix), dim=2)
        upper_tri_mask = torch.triu(torch.ones_like(l1_diffs), diagonal=1).bool()
        div_loss = -l1_diffs[upper_tri_mask].mean()
        total_loss += 1.5 * div_loss  # 0.5 → 1.5
    
    # 2. 季节内多样性
    intra_div_loss = torch.tensor(0.0, device=gen_curves.device)
    for season in range(4):
        mask = (season_idx == season)
        if mask.sum() > 1:
            season_curves = gen_curves[mask]
            diff_matrix = season_curves.unsqueeze(1) - season_curves.unsqueeze(0)
            l1_diffs = torch.mean(torch.abs(diff_matrix), dim=2)
            upper_tri_mask = torch.triu(torch.ones_like(l1_diffs), diagonal=1).bool()
            if upper_tri_mask.any():
                intra_div_loss += -l1_diffs[upper_tri_mask].mean()
    total_loss += 2.0 * intra_div_loss / 4  # 0.8 → 2.0
    
    # === 时序损失（适度恢复） ===
    # 1. 轻量级ACF（只约束前5个lag）
    if USE_LIGHTWEIGHT_ACF:
        acf_loss = torch.tensor(0.0, device=gen_curves.device)
        for lag in range(1, 6):  # 只约束前5个lag
            curve_t = gen_curves[:, :-lag]
            curve_t_lag = gen_curves[:, lag:]
            
            # 中心化
            curve_t_centered = curve_t - curve_t.mean(dim=1, keepdim=True)
            curve_t_lag_centered = curve_t_lag - curve_t_lag.mean(dim=1, keepdim=True)
            
            # 计算相关系数
            numerator = torch.sum(curve_t_centered * curve_t_lag_centered, dim=1)
            var_t = torch.sum(curve_t_centered ** 2, dim=1)
            var_t_lag = torch.sum(curve_t_lag_centered ** 2, dim=1)
            denominator = torch.sqrt(torch.clamp(var_t * var_t_lag, min=1e-8))
            corr = (numerator / denominator).mean()
            
            # 期望的衰减模式
            expected = torch.exp(torch.tensor(-0.1 * lag, device=gen_curves.device))
            acf_loss += torch.abs(corr - expected)
        
        total_loss += 0.5 * acf_loss / 5  # 轻量级ACF
    
    # 2. 时序相关性
    adjacent_diffs = torch.abs(gen_curves[:, 1:] - gen_curves[:, :-1])
    temporal_loss = torch.mean(adjacent_diffs)
    total_loss += 0.3 * temporal_loss  # 0.02 → 0.3
    
    # 3. 频域损失
    fft = torch.fft.fft(gen_curves, dim=1)
    magnitude = torch.abs(fft)
    low_freq_ratio = magnitude[:, :magnitude.size(1)//4].sum(dim=1) / magnitude.sum(dim=1)
    freq_loss = -low_freq_ratio.mean()
    total_loss += 0.2 * freq_loss  # 0.02 → 0.2
    
    # === 风电物理约束（保持） ===
    # 1. 出力范围约束
    range_penalty = torch.relu(-gen_curves) + torch.relu(gen_curves - 1.0)
    total_loss += 2.0 * range_penalty.mean()
    
    # 2. Summer低出力约束
    summer_mask = (season_idx == 1)
    if summer_mask.any():
        summer_curves = gen_curves[summer_mask]
        mean_output = summer_curves.mean()
        mean_penalty = torch.relu(mean_output - 0.35)
        total_loss += 5.0 * mean_penalty
    
    return total_loss


# ========== 使用方法 ==========
# 在训练循环中替换原来的total_loss_G计算：
"""
# 原代码（第1200行附近）：
total_loss_G = (loss_G + aux_loss_G + 5.0 * similarity_loss + ...)

# 替换为：
total_loss_G = balanced_loss_function(
    gen_curves, real_curves, season_vec,
    loss_G, aux_loss_G, d_gen, aux_gen,
    USE_LIGHTWEIGHT_ACF=True
)
"""
