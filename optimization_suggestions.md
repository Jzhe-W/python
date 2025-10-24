# CWGAN-GP 风电数据生成优化建议

## 📊 主要问题分析

### 1. **超参数优化不足** ⚠️
```python
# 当前问题
n_trials=2  # 仅2次试验
ultra_fast_epochs=5  # 仅5轮训练
```
**影响**：无法找到真正的最优参数组合

### 2. **损失权重失衡** ⚠️
```python
# 阶段1+2权重调整可能矫枉过正
5.0 * similarity_loss   # 提高10倍
5.0 * mag_loss          # 提高2.5倍
0.5 * div_loss          # 降低4倍
```
**影响**：模型过度追求相似性，牺牲了多样性和泛化能力

### 3. **季节特殊处理过度** ⚠️
- Summer使用2.5倍噪声
- Winter使用2.0倍噪声
- 多个硬编码的季节约束
**影响**：破坏了模型的统一性，难以协调

### 4. **双重归一化问题** ⚠️
```python
# Z-score → MinMax → [0,1]
# 反归一化时容易出错
```
**影响**：信息损失，反归一化不准确

### 5. **ACF损失被完全禁用** ⚠️
```python
acf_reg_loss = torch.tensor(0.0, device=gen_curves.device)
```
**影响**：丢失了时序相关性特征

---

## 🚀 优化方案

### 方案A：渐进式优化（推荐）

#### Step 1: 增加超参数优化强度
```python
# 修改超参数优化配置
optimizer = SimpleHyperparameterOptimizer(
    train_dataset, val_dataset, 
    n_trials=10,              # 2 → 10
    ultra_fast_mode=False     # 关闭极速模式
)
optimizer.ultra_fast_epochs = 30  # 5 → 30
```

#### Step 2: 重新平衡损失权重
```python
# 建议的平衡权重（基于WGAN-GP最佳实践）
total_loss_G = (
    loss_G + aux_loss_G +
    
    # 相似性（适度降低）
    2.0 * similarity_loss +      # 5.0 → 2.0
    2.0 * mag_loss +             # 5.0 → 2.0
    
    # 多样性（适度提高）
    1.5 * div_loss +             # 0.5 → 1.5
    2.0 * intra_div_loss +       # 0.8 → 2.0
    
    # 阶段2损失（适度降低）
    1.5 * pointwise_loss +       # 3.0 → 1.5
    1.0 * segment_loss +         # 2.0 → 1.0
    1.0 * peak_valley_loss +     # 2.0 → 1.0
    1.0 * dist_loss +            # 1.5 → 1.0
    
    # 时序（适度恢复）
    0.3 * temporal_loss +        # 0.02 → 0.3
    0.2 * freq_loss +            # 0.02 → 0.2
    
    # 风电约束（保持）
    2.0 * wind_physical_loss +
    5.0 * wind_physical_loss_v2 +
    10.0 * wind_constraint
)
```

#### Step 3: 统一季节噪声策略
```python
# 移除过度的季节特殊处理
# 统一使用动态噪声调整
progress = epoch / epochs
base_noise = noise_sigma * (1.0 + 0.5 * np.sin(2 * np.pi * progress))

# 所有季节使用统一策略
z = torch.randn(batch_size, z_dim, device=device) * base_noise + noise_mu
```

#### Step 4: 简化归一化方案
```python
# 只使用MinMax归一化（移除Z-score）
scaler = MinMaxScaler(feature_range=(0, 1))
data_norm = scaler.fit_transform(data.values.T).T

# 反归一化也更简单
fake_denorm = scaler.inverse_transform(fake_all.T).T
```

#### Step 5: 恢复轻量级ACF约束
```python
# 使用轻量级ACF损失（只约束前5个lag）
def lightweight_acf_loss(fake_curves, max_lag=5):
    acf_loss = 0
    for lag in range(1, max_lag + 1):
        curve_t = fake_curves[:, :-lag]
        curve_t_lag = fake_curves[:, lag:]
        corr = torch.mean(curve_t * curve_t_lag)
        expected = torch.exp(torch.tensor(-0.1 * lag, device=fake_curves.device))
        acf_loss += torch.abs(corr - expected)
    return acf_loss / max_lag

# 添加到总损失
total_loss_G += 0.5 * lightweight_acf_loss(gen_curves)
```

---

### 方案B：激进式优化（快速实验）

#### 完全重构损失函数
```python
# 极简化的损失函数（基于Vanilla WGAN-GP）
total_loss_G = (
    loss_G +                          # WGAN核心损失
    aux_loss_G +                      # 辅助分类损失
    1.0 * F.mse_loss(gen_curves, real_curves) +  # 简单MSE
    0.5 * seasonal_diversity_loss(gen_curves, season_vec)  # 最小多样性
)
```

#### 移除所有季节特殊处理
```python
# 统一噪声策略
z = torch.randn(batch_size, z_dim, device=device) * noise_sigma

# 移除所有if summer_mask / winter_mask分支
```

---

### 方案C：数据驱动优化

#### 1. 分析真实数据特征
```python
# 添加真实数据特征分析
def analyze_real_data(data, season_labels):
    """分析真实数据的统计特征"""
    for season in range(4):
        season_data = data[season_labels[:, season] == 1]
        print(f"{season_names[season]}:")
        print(f"  均值: {season_data.mean():.4f}")
        print(f"  标准差: {season_data.std():.4f}")
        print(f"  峰值: {season_data.max():.4f}")
        print(f"  谷值: {season_data.min():.4f}")
        print(f"  变化率: {np.diff(season_data).std():.4f}")

# 在训练前调用
analyze_real_data(train_data.values.T, train_labels)
```

#### 2. 基于分析调整约束
根据真实数据的实际统计特征，动态设置约束目标

---

## 🎯 快速诊断清单

运行以下代码诊断当前模型状态：

```python
# 添加到训练循环中
if epoch % 50 == 0:
    print("\n=== 模型诊断 ===")
    print(f"1. W距离: {avg_wasserstein:.4f} (目标: <0.1)")
    print(f"2. 季节差异: {avg_season_diff:.4f} (目标: >0.15)")
    print(f"3. 生成器损失: {avg_loss_G:.4f}")
    print(f"4. 判别器损失: {avg_loss_D:.4f}")
    
    # 生成样本检查
    with torch.no_grad():
        z_test = torch.randn(16, z_dim, device=device) * noise_sigma
        season_test = torch.eye(4, device=device).repeat(4, 1)
        fake_test = G(z_test, season_test, 0).detach().cpu().numpy()
        
        print(f"5. 生成数据范围: [{fake_test.min():.4f}, {fake_test.max():.4f}]")
        print(f"6. 生成数据均值: {fake_test.mean():.4f}")
        print(f"7. 生成数据标准差: {fake_test.std():.4f}")
```

---

## 📋 推荐执行顺序

### 阶段1：快速验证（1-2小时）
1. ✅ 使用方案B的极简损失函数
2. ✅ 训练200轮，观察W距离和MAPE
3. ✅ 如果MAPE显著降低（<200%），说明过度复杂化是主要问题

### 阶段2：精细调优（3-5小时）
1. ✅ 实施方案A的Step 1-3
2. ✅ 训练500轮，记录最佳模型
3. ✅ 对比优化前后的指标

### 阶段3：深度优化（可选）
1. ✅ 实施方案C的数据分析
2. ✅ 基于分析结果微调约束
3. ✅ 训练完整的2000轮

---

## 🔧 关键参数建议值

```python
# 超参数优化
n_trials = 10
ultra_fast_epochs = 30

# 训练参数
epochs = 1000  # 2000 → 1000（先看效果）
batch_size = 32  # 64 → 32（提高稳定性）
n_critic = 3  # 5 → 3（加快G更新）

# 学习率
lr_G = 2e-4  # 保持
lr_D = 1e-4  # 保持

# 噪声
noise_sigma = 0.2  # 统一使用，不分季节

# 早停
early_stopping_patience = 100  # 50 → 100
```

---

## 🎯 预期改善

实施优化后的预期指标：
- **Summer MAPE**: 400-500% → **150-250%**
- **整体MAPE**: 200-300% → **100-150%**
- **W距离**: >0.5 → **<0.1**
- **训练稳定性**: 显著提升
- **收敛速度**: 提高2-3倍

