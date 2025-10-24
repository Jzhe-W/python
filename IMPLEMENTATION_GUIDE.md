# 🚀 CWGAN-GP 风电数据生成优化实施指南

## 📋 快速诊断：找出核心问题

### Step 0: 运行诊断代码（5分钟）

在你的代码中添加这段诊断代码，了解当前模型的问题：

```python
# 在训练开始前添加
print("\n" + "="*60)
print("🔍 模型诊断分析")
print("="*60)

# 1. 数据质量检查
print("\n1. 数据质量:")
print(f"  真实数据范围: [{data.min().min():.2f}, {data.max().max():.2f}]")
print(f"  归一化后范围: [{data_norm.min().min():.4f}, {data_norm.max().max():.4f}]")
print(f"  预测数据范围: [{forecast_data.min().min():.2f}, {forecast_data.max().max():.2f}]")

# 2. 损失权重检查
print("\n2. 当前损失权重:")
print(f"  相似性权重: 5.0 (可能过高)")
print(f"  量级匹配权重: 5.0 (可能过高)")
print(f"  多样性权重: 0.5 (可能过低)")
print(f"  季节内多样性权重: 0.8 (可能过低)")

# 3. 超参数检查
print("\n3. 超参数优化配置:")
print(f"  优化试验次数: {optimizer.n_trials} (建议≥10)")
print(f"  极速模式训练轮数: {optimizer.ultra_fast_epochs} (建议≥30)")
print(f"  早停patience: {early_stopping_patience} (建议≥100)")

# 4. 季节处理策略
print("\n4. 季节处理策略:")
print(f"  Summer噪声系数: 2.5x (可能过高)")
print(f"  Winter噪声系数: 2.0x (可能过高)")
print(f"  其他季节噪声系数: 0.8x")
print(f"  ⚠️  建议：统一噪声策略，避免过度差异化")

print("="*60 + "\n")
```

### 诊断结果解读

根据输出判断主要问题：

| 指标 | 正常范围 | 如果超出 |
|------|---------|---------|
| W距离 | < 0.1 | 权重失衡或收敛困难 |
| Summer MAPE | < 200% | 季节约束过度或归一化问题 |
| 生成数据范围 | 接近真实数据 | 反归一化错误 |
| 季节差异 | > 0.15 | 多样性不足 |

---

## 🎯 推荐的优化路径

### 路径1：快速修复（1-2小时，推荐）

**适用场景**：你想快速看到改善效果

#### 1.1 重新平衡损失权重（最重要！）

找到训练循环中的`total_loss_G`计算部分（约1200行），替换为：

```python
# 导入修复代码
from quick_fix_v1 import balanced_loss_function

# 替换原来的复杂损失计算
total_loss_G = balanced_loss_function(
    gen_curves, real_curves, season_vec,
    loss_G, aux_loss_G, d_gen, aux_gen,
    USE_LIGHTWEIGHT_ACF=True
)
```

**预期改善**：Summer MAPE 400-500% → 200-300%

#### 1.2 统一季节处理策略

找到噪声生成部分（约1150行），替换为：

```python
# 导入修复代码
from quick_fix_v2 import unified_season_noise

# 替换原来的季节特殊处理
z = unified_season_noise(batch_size, z_dim, noise_sigma, epoch, epochs, device)
```

**预期改善**：训练稳定性提升，各季节性能更均衡

#### 1.3 运行快速测试

```python
# 修改训练参数
epochs = 500  # 先跑500轮看效果
batch_size = 32  # 降低批次大小

# 开始训练
# 观察前100轮的W距离变化
```

**判断标准**：
- ✅ W距离在100轮内降到<0.3：继续训练
- ❌ W距离徘徊在>0.5：进入路径2深度优化

---

### 路径2：深度优化（3-5小时）

**适用场景**：路径1效果不明显，或需要最佳性能

#### 2.1 简化归一化方案

```python
# 导入修复代码
from quick_fix_v4_simplified_normalization import simplified_normalization, simplified_denormalization

# 替换原来的双重归一化
data_norm, forecast_data_norm, scaler = simplified_normalization(data, forecast_data)

# 训练后反归一化
fake_denorm = simplified_denormalization(fake_all, scaler)
```

**预期改善**：消除反归一化错误，数据范围更准确

#### 2.2 改进超参数优化

```python
# 导入修复代码
from quick_fix_v3 import improved_objective_function

# 增加优化强度
if OPTUNA_AVAILABLE:
    optimizer = SimpleHyperparameterOptimizer(
        train_dataset, val_dataset,
        n_trials=10,              # 2 → 10
        ultra_fast_mode=False     # 关闭极速模式
    )
    optimizer.ultra_fast_epochs = 30  # 5 → 30
    best_params = optimizer.optimize()
```

**预期改善**：找到更优的超参数组合

#### 2.3 添加动态学习率和改进早停

```python
# 导入修复代码
from quick_fix_v3 import dynamic_lr_adjustment, ImprovedEarlyStopping

# 创建早停对象
early_stopping = ImprovedEarlyStopping(patience=100, min_delta=0.001)

# 在训练循环中
if epoch % 20 == 0:
    # 动态学习率调整
    dynamic_lr_adjustment(opt_G, opt_D, avg_wasserstein, epoch)
    
    # 早停检查
    if early_stopping(avg_wasserstein):
        break
```

**预期改善**：训练更稳定，自动找到最佳停止点

#### 2.4 完整训练

```python
epochs = 1000  # 完整训练1000轮
# 观察W距离、MAPE、季节差异等指标
```

---

### 路径3：激进重构（仅在路径1/2都失败时）

**适用场景**：模型完全不收敛，或Summer MAPE>500%

#### 3.1 使用极简损失函数

```python
# 完全移除所有复杂约束，回归WGAN-GP基础
total_loss_G = (
    loss_G +                                      # WGAN核心
    aux_loss_G +                                  # 辅助分类
    2.0 * F.mse_loss(gen_curves, real_curves)   # 简单MSE
)
```

#### 3.2 移除所有季节特殊处理

```python
# 统一噪声
z = torch.randn(batch_size, z_dim, device=device) * noise_sigma

# 移除所有if summer_mask/winter_mask分支
```

#### 3.3 短期快速测试

```python
epochs = 200  # 只跑200轮
# 如果W距离能降到<0.2，说明简化有效，再逐步添加约束
```

---

## 📊 实施检查表

### 开始前（必做）

- [ ] 备份当前代码
- [ ] 运行Step 0诊断代码
- [ ] 记录当前的MAPE、W距离等基准指标
- [ ] 确认数据路径正确

### 路径1实施

- [ ] 下载所有quick_fix_vX.py文件
- [ ] 替换损失函数（quick_fix_v1）
- [ ] 统一季节策略（quick_fix_v2）
- [ ] 运行500轮测试
- [ ] 对比优化前后指标

### 路径2实施（如果路径1不够）

- [ ] 简化归一化（quick_fix_v4）
- [ ] 改进超参数优化（quick_fix_v3）
- [ ] 添加动态学习率和早停
- [ ] 运行1000轮完整训练
- [ ] 分析最终结果

### 路径3实施（紧急救援）

- [ ] 使用极简损失函数
- [ ] 移除所有特殊处理
- [ ] 短期测试200轮
- [ ] 判断是否继续

---

## 🎯 预期时间线

| 路径 | 准备 | 训练 | 分析 | 总计 |
|-----|------|------|------|------|
| 路径1 | 10分钟 | 1小时 | 20分钟 | ~1.5小时 |
| 路径2 | 20分钟 | 3小时 | 30分钟 | ~4小时 |
| 路径3 | 5分钟 | 30分钟 | 10分钟 | ~45分钟 |

---

## 📈 成功标准

### 最低标准（可接受）

- Summer MAPE < 300%
- 整体MAPE < 200%
- W距离 < 0.2
- 各季节多样性评分 > 0.4

### 良好标准（推荐目标）

- Summer MAPE < 200%
- 整体MAPE < 150%
- W距离 < 0.1
- 各季节多样性评分 > 0.6

### 优秀标准（理想状态）

- Summer MAPE < 150%
- 整体MAPE < 100%
- W距离 < 0.05
- 各季节多样性评分 > 0.8

---

## 🔧 常见问题排查

### Q1: 修改后代码报错

**解决方案**：
```python
# 确保导入所有依赖
import torch
import torch.nn.functional as F
import numpy as np

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Q2: W距离不下降

**可能原因**：
1. 学习率过高 → 降低到1e-4
2. 判别器过强 → 降低n_critic到3
3. 权重失衡 → 使用quick_fix_v1

### Q3: Summer季节MAPE爆炸

**可能原因**：
1. 归一化错误 → 使用quick_fix_v4
2. 约束过度 → 使用quick_fix_v2
3. 数据质量问题 → 检查原始数据

### Q4: GPU内存不足

**解决方案**：
```python
batch_size = 16  # 降低批次大小
n_samples_for_interval = 100  # 降低评估样本数

# 添加内存清理
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## 📞 下一步行动

### 立即执行（5分钟内）

1. 运行Step 0诊断代码
2. 确定主要问题类型
3. 选择优化路径（路径1/2/3）

### 今天完成

1. 实施选定路径的优化
2. 运行测试训练
3. 记录关键指标变化

### 后续跟进

1. 分析优化效果
2. 微调剩余参数
3. 完整训练并验证

---

## 📝 记录模板

```markdown
## 优化实验记录

**日期**: 2025-XX-XX
**选择路径**: 路径1 / 路径2 / 路径3
**训练轮数**: XXX

### 优化前基准
- Summer MAPE: XXX%
- 整体MAPE: XXX%
- W距离: XXX
- 训练时间: XXX小时

### 实施的改动
1. [ ] 改动1：XXX
2. [ ] 改动2：XXX
3. [ ] 改动3：XXX

### 优化后结果
- Summer MAPE: XXX% (改善: ±XX%)
- 整体MAPE: XXX% (改善: ±XX%)
- W距离: XXX (改善: ±XX)
- 训练时间: XXX小时

### 结论
- [ ] 达到最低标准
- [ ] 达到良好标准
- [ ] 达到优秀标准
- [ ] 需要进一步优化

### 下一步计划
XXX
```

---

## 🎉 祝优化顺利！

有问题随时反馈，我会继续协助调整策略。
