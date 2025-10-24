# ⚡ 30分钟快速优化指南

> **目标**：用最少的改动，最快看到效果

## 📋 你需要做的（3步）

### Step 1: 下载优化文件（1分钟）

下载这两个文件：
- `quick_fix_v1.py` - 损失函数优化
- `quick_fix_v2.py` - 季节策略优化

放到你的代码同目录下。

---

### Step 2: 修改你的代码（10分钟）

#### 改动1：替换损失函数（约1200行）

**找到这段代码**：
```python
total_loss_G = (loss_G + aux_loss_G +
                5.0 * similarity_loss +
                5.0 * mag_loss +
                ...)
```

**替换为**：
```python
# 在文件开头添加导入
from quick_fix_v1 import balanced_loss_function

# 替换损失计算
total_loss_G = balanced_loss_function(
    gen_curves, real_curves, season_vec,
    loss_G, aux_loss_G, d_gen, aux_gen,
    USE_LIGHTWEIGHT_ACF=True
)
```

#### 改动2：统一噪声策略（约1150行）

**找到这段代码**：
```python
season_idx = season_vec.argmax(dim=1)
summer_mask = (season_idx == 1)
if summer_mask.any():
    z = torch.randn(...) * (noise_sigma * 2.5) + noise_mu
elif winter_mask.any():
    z = torch.randn(...) * (noise_sigma * 2.0) + noise_mu
else:
    z = torch.randn(...) * (noise_sigma * 0.8) + noise_mu
```

**替换为**：
```python
# 在文件开头添加导入
from quick_fix_v2 import unified_season_noise

# 替换噪声生成
z = unified_season_noise(batch_size, z_dim, noise_sigma, epoch, epochs, device)
```

#### 改动3：调整训练参数（约第1000行）

**找到这段代码**：
```python
epochs = 2000
batch_size = 64
n_critic = 5
```

**替换为**：
```python
epochs = 500       # 先跑500轮快速测试
batch_size = 32    # 降低批次大小，提高稳定性
n_critic = 3       # 减少判别器更新次数
```

---

### Step 3: 运行并观察（20分钟）

```bash
python your_script.py
```

**观察前100轮的输出**：

#### ✅ 成功的信号：
```
Epoch 20: W_dist=0.45, ...  # 开始下降
Epoch 40: W_dist=0.35, ...  # 持续下降
Epoch 60: W_dist=0.28, ...  # 接近目标
Epoch 100: W_dist=0.20, ... # ✅ 成功！
```

#### ⚠️ 需要继续优化：
```
Epoch 20: W_dist=0.55, ...  # 下降缓慢
Epoch 40: W_dist=0.50, ...
Epoch 60: W_dist=0.48, ...
Epoch 100: W_dist=0.45, ... # 有改善但不够
```
→ 继续看完整的优化方案

#### ❌ 可能有问题：
```
Epoch 20: W_dist=0.80, ...  # 没有下降
Epoch 40: W_dist=0.85, ...  # 甚至上升
Epoch 100: W_dist=0.90, ... # 没有改善
```
→ 检查代码是否正确修改

---

## 📊 如何判断效果？

### 关键指标对比表

| 指标 | 优化前（估计） | 优化后（目标） | 如何查看 |
|-----|--------------|--------------|----------|
| W距离 | > 0.5 | < 0.2 | 每20轮打印 |
| Summer MAPE | 400-500% | 200-300% | 训练结束后的表格 |
| 整体MAPE | 200-300% | 150-200% | 训练结束后的表格 |

### 快速判断法

**看第100轮的W距离**：
- `W_dist < 0.2`：✅ 优化成功！继续训练到500轮
- `0.2 < W_dist < 0.3`：⚠️ 有改善但不够，继续观察200轮
- `W_dist > 0.3`：❌ 需要深度优化，查看完整方案

---

## 💡 常见问题速查

### Q1: 改完后报错 "ModuleNotFoundError"
```bash
# 确保quick_fix文件和你的代码在同一目录
ls -la
# 应该看到：
# your_script.py
# quick_fix_v1.py
# quick_fix_v2.py
```

### Q2: 改完后报错 "NameError: name 'balanced_loss_function' is not defined"
```python
# 检查文件开头是否有导入
from quick_fix_v1 import balanced_loss_function
from quick_fix_v2 import unified_season_noise
```

### Q3: W距离没有下降
**可能原因**：
1. 代码没有正确替换 → 检查改动位置
2. 学习率过高 → 降低到1e-4
3. 需要更多优化 → 查看完整方案

### Q4: GPU内存不足
```python
# 进一步降低批次大小
batch_size = 16  # 32 → 16
```

---

## 🎯 预期时间线

| 阶段 | 时间 | 做什么 |
|-----|------|--------|
| 0-1分钟 | 下载文件 | 下载quick_fix_v1.py和v2.py |
| 1-10分钟 | 修改代码 | 替换2处代码 |
| 10-30分钟 | 运行测试 | 观察前100轮 |
| **总计** | **30分钟** | **完成快速优化** |

---

## ✅ 成功清单

- [ ] 下载了quick_fix_v1.py和v2.py
- [ ] 修改了损失函数（改动1）
- [ ] 修改了噪声策略（改动2）
- [ ] 调整了训练参数（改动3）
- [ ] 运行了100轮测试
- [ ] W距离降到了<0.3
- [ ] 准备继续完整训练500轮

---

## 📖 如果需要更多优化

完成上述快速优化后，如果还想进一步提升：

1. **查看** `IMPLEMENTATION_GUIDE.md` 了解路径2的完整优化
2. **使用** `quick_fix_v3.py` 改进超参数优化
3. **使用** `quick_fix_v4.py` 简化归一化方案

预期可以将Summer MAPE进一步降低到150-250%。

---

## 🚀 现在开始！

1. 下载文件 → 
2. 修改代码 → 
3. 运行测试 → 
4. 🎉 30分钟后看到改善！

**祝优化顺利！** 🎊
