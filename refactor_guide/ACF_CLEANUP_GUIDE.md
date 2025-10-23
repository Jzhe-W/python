# 🗑️ ACF 相关代码清理指南

## 📋 需要删除的所有注释掉的ACF代码

---

## 位置1：训练循环中的ACF损失（约第3060行）

### **搜索关键词：**
```python
# 时序相关性损失
temporal_loss = temporal_correlation_loss(gen_curves, season_vec)
```

### **找到这一行（紧接着）：**
```python
# autocorr_loss = autocorrelation_loss(gen_curves, season_vec)  # 完全移除ACF损失
```

### **删除内容：**
```python
# ❌ 删除这一行
# autocorr_loss = autocorrelation_loss(gen_curves, season_vec)  # 完全移除ACF损失
```

---

## 位置2：ACF权重设置（约第3100行）

### **搜索关键词：**
```python
# 使用季节化权重
div_loss_weight = seasonal_weights['div_loss_weight']
```

### **找到这部分：**
```python
# autocorr_weight = seasonal_weights['autocorr_weight']  # 完全移除ACF权重
quality_weight = seasonal_weights['quality_weight']
```

### **删除内容：**
```python
# ❌ 删除这一行
# autocorr_weight = seasonal_weights['autocorr_weight']  # 完全移除ACF权重
```

---

## 位置3：ACF精确权重调度（约第3130行）

### **搜索关键词：**
```python
# ACF精确权重调度
progress = min(1.0, max(0.0, epoch / float(epochs)))
```

### **找到这部分：**
```python
# ACF精确权重调度：完全移除ACF相关权重
progress = min(1.0, max(0.0, epoch / float(epochs)))
# enh_autocorr_w = 0.8 + 0.4 * progress  # 完全移除ACF权重
enh_freq_w = 0.4 + 0.2 * progress  # 0.4→0.6 降低范围
sep_w = 0.2 + 0.1 * progress  # 适中季节分离权重 (0.2->0.3)
```

### **删除内容：**
```python
# ❌ 删除这一行
# enh_autocorr_w = 0.8 + 0.4 * progress  # 完全移除ACF权重
```

---

## 位置4：季节特殊优化ACF权重（约第3160-3180行）

### **搜索关键词：**
```python
# A2: 季节特殊优化
spring_mask = (season_vec.argmax(dim=1) == 0)
```

### **找到这部分：**
```python
# 完全移除ACF权重计算
# spring_acf_weight = 1.5 if spring_mask.any() else 1.0  # 完全移除
# summer_acf_weight = 4.0 if summer_mask.any() else 1.0  # 完全移除
# autumn_acf_weight = 1.8 if autumn_mask.any() else 1.0  # 完全移除
# winter_acf_weight = 2.0 if winter_mask.any() else 1.0  # 完全移除
```

### **删除内容：**
```python
# ❌ 删除这4行
# spring_acf_weight = 1.5 if spring_mask.any() else 1.0  # 完全移除
# summer_acf_weight = 4.0 if summer_mask.any() else 1.0  # 完全移除
# autumn_acf_weight = 1.8 if autumn_mask.any() else 1.0  # 完全移除
# winter_acf_weight = 2.0 if winter_mask.any() else 1.0  # 完全移除
```

---

## 位置5：ACF正则化损失（约第3190行）

### **搜索关键词：**
```python
# 完全移除ACF正则化项和相关计算
```

### **找到这部分：**
```python
# 完全移除ACF正则化项和相关计算
# acf_reg_loss = torch.tensor(0.0, device=gen_curves.device)  # 完全移除ACF正则化
acf_reg_loss = torch.tensor(0.0, device=gen_curves.device)  # 设为0
```

### **删除内容：**
```python
# ❌ 删除这一行
# acf_reg_loss = torch.tensor(0.0, device=gen_curves.device)  # 完全移除ACF正则化
```

**注意：** 保留下面这行（因为代码中使用了这个变量）
```python
acf_reg_loss = torch.tensor(0.0, device=gen_curves.device)  # 设为0
```

---

## 位置6：多lag ACF损失（约第3200行）

### **搜索关键词：**
```python
# 完全移除多lag ACF损失函数和相关计算
```

### **找到这部分：**
```python
# 完全移除多lag ACF损失函数和相关计算
# multi_acf_loss = torch.tensor(0.0, device=gen_curves.device)  # 完全移除多lag ACF
```

### **删除内容：**
```python
# ❌ 删除这2行注释
# 完全移除多lag ACF损失函数和相关计算
# multi_acf_loss = torch.tensor(0.0, device=gen_curves.device)  # 完全移除多lag ACF
```

---

## 位置7：精确ACF匹配损失（约第3280行）

### **搜索关键词：**
```python
# 计算精确ACF匹配损失 - 删除: 风电不需要精确ACF匹配
```

### **找到这部分：**
```python
        # 计算精确ACF匹配损失 - 删除: 风电不需要精确ACF匹配
        # precise_acf_loss = precise_acf_matching_loss(gen_curves, real_curves, max_lag=15)
        precise_acf_loss = torch.tensor(0.0, device=gen_curves.device)  # 设为0
```

### **删除内容：**
```python
# ❌ 删除这一行
# precise_acf_loss = precise_acf_matching_loss(gen_curves, real_curves, max_lag=15)
```

**注意：** 保留下面这行
```python
precise_acf_loss = torch.tensor(0.0, device=gen_curves.device)  # 设为0
```

---

## 位置8：Winter季节ACF损失（约第3390行）

### **搜索关键词：**
```python
# Winter季节专用优化 - 针对性解决Winter多样性暴跌问题
```

### **找到这部分：**
```python
        # 修复: 完全移除Winter的ACF损失，专注多样性
        # winter_acf = autocorrelation_loss(gen_curves[winter_mask], season_vec[winter_mask])

        # 添加到总损失 (Winter专用优化权重)
        total_loss_G += 1.5 * winter_div_loss + 0.8 * winter_shape_loss  # 移除 winter_acf
```

### **删除内容：**
```python
# ❌ 删除这2行注释
        # 修复: 完全移除Winter的ACF损失，专注多样性
        # winter_acf = autocorrelation_loss(gen_curves[winter_mask], season_vec[winter_mask])
```

---

## 位置9：Spring季节ACF损失（约第3410行）

### **搜索关键词：**
```python
# Spring季节保护机制 - 防止过度优化多样性导致质量下降
```

### **找到这部分：**
```python
            # 修复: 完全移除Spring的ACF损失，专注质量保护
            # spring_acf = autocorrelation_loss(spring_curves, season_vec[spring_mask])

            # 添加到总损失 (Spring保护权重)
            total_loss_G += 1.2 * spring_quality_loss + 0.3 * spring_div_loss  # 移除 spring_acf
```

### **删除内容：**
```python
# ❌ 删除这2行注释
            # 修复: 完全移除Spring的ACF损失，专注质量保护
            # spring_acf = autocorrelation_loss(spring_curves, season_vec[spring_mask])
```

---

## 位置10：训练说明中的ACF相关注释（约第2850行）

### **搜索关键词：**
```python
print("🎯 风电优化策略:")
```

### **找到这部分：**
```python
print("   - ✅ 完全移除ACF约束：风电自相关性极弱，已彻底删除所有ACF损失计算")
```

### **处理方式：**
这是说明文字，**可以保留**（告诉用户已经移除了ACF）

---

## 位置11：无用的ACF数据准备（程序末尾，约第3800行）

### **搜索关键词：**
```python
# 准备数据用于ACF相关性误差图
```

### **删除整个代码块（约30行）：**
```python
# ❌ 删除从这里开始
# 准备数据用于ACF相关性误差图
print("📊 准备数据用于ACF相关性误差图...")

# 收集真实数据和生成数据
real_data_for_acf = []
generated_data_for_acf = []
all_target_indices = []

# 向量化计算所有季节的目标索引
all_target_indices = []
for i, season in enumerate(season_names):
    # 获取该季节的原始数据
    pool_original = season_dict_original[i]
    val_indices = season_indices_dict[i]

    # 使用改进的曲线选择函数
    target_idx, _ = select_representative_curve(
        pool_original,
        selection_strategy='random_from_median'
    )

    all_target_indices.append(target_idx)

    # 获取真实数据（原始量级）
    real_curve_original = pool_original[target_idx]
    real_data_for_acf.append(real_curve_original)

    # 获取生成数据（原始量级）
    generated_curve_original = fake_denorm[i]
    generated_data_for_acf.append(generated_curve_original)

print("🎨 ACF相关性误差图分析完成！")
print("📋 提示：您可以在GUI窗口的'ACF相关性误差图'标签页中查看结果")
print("💡 该图表展示了真实数据与生成数据之间的ACF相关性误差分布")
# ❌ 删除到这里结束
```

---

## 清理汇总表

| 位置 | 行数范围 | 删除内容 | 是否影响运行 |
|------|---------|---------|-------------|
| 位置1 | ~3060 | 1行注释 | ❌ 不影响 |
| 位置2 | ~3100 | 1行注释 | ❌ 不影响 |
| 位置3 | ~3130 | 1行注释 | ❌ 不影响 |
| 位置4 | ~3170 | 4行注释 | ❌ 不影响 |
| 位置5 | ~3190 | 1行注释 | ❌ 不影响 |
| 位置6 | ~3200 | 2行注释 | ❌ 不影响 |
| 位置7 | ~3280 | 1行注释 | ❌ 不影响 |
| 位置8 | ~3390 | 2行注释 | ❌ 不影响 |
| 位置9 | ~3410 | 2行注释 | ❌ 不影响 |
| 位置10 | ~2850 | 保留 | - |
| 位置11 | ~3800 | 30行代码 | ❌ 不影响 |

**总计：** 约15行注释 + 30行无用代码 = **45行可删除**

---

## 快速批量删除方法

### **方法1：使用正则表达式（VSCode/Sublime等）**

1. 打开查找替换（Ctrl+H）
2. 启用正则表达式模式
3. 搜索模式：
   ```regex
   ^\s*#.*ACF.*完全移除.*\n
   ```
4. 替换为：（空）
5. 全部替换

### **方法2：手动逐个删除**

按照上面的位置列表，使用 Ctrl+F 搜索关键词，逐个删除。

### **方法3：使用脚本自动清理**

```python
# cleanup_acf_comments.py
import re

def clean_acf_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 删除包含"完全移除ACF"的注释行
    content = re.sub(r'^\s*#.*完全移除.*ACF.*\n', '', content, flags=re.MULTILINE)
    
    # 删除包含"完全移除多lag ACF"的注释行
    content = re.sub(r'^\s*#.*完全移除多lag.*\n', '', content, flags=re.MULTILINE)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ ACF注释清理完成")

# 使用
clean_acf_comments('your_program.py')
```

---

## 清理后的验证

删除后运行程序，检查：

```bash
# 1. 搜索"完全移除"，应该找不到或很少
grep -n "完全移除" your_program.py

# 2. 搜索"ACF"，应该只剩下：
#    - ACF曲线绘制代码（保留）
#    - ACF对比打印（保留）
#    - 说明文字（保留）
grep -n "ACF" your_program.py

# 3. 运行程序，确保无错误
python your_program.py
```

---

## 为什么可以安全删除？

### **这些注释掉的代码的特点：**

1. ✅ 已经被注释掉，不会执行
2. ✅ 有替代代码（如 `acf_reg_loss = torch.tensor(0.0, ...)`)
3. ✅ 删除后不影响任何功能
4. ✅ 只是历史遗留，用于说明修改过程

### **保留的ACF相关代码：**

1. ✅ `acf()` 函数调用（用于计算和对比）
2. ✅ ACF打印输出（用于数值对比）
3. ✅ ACF折线图绘制（用于可视化）
4. ✅ 说明文字（告诉用户做了什么修改）

---

## 清理顺序建议

### **推荐顺序：**

1. **先清理简单的单行注释**（位置1-9）
   - 风险：低
   - 时间：5分钟

2. **再清理无用的数据准备代码**（位置11）
   - 风险：低
   - 时间：2分钟

3. **验证程序运行**
   - 运行一次，确保无错误
   - 时间：3分钟

**总时间：** 约10分钟

---

## 完成标志

清理完成后，您应该看到：

```
✅ 代码更简洁
✅ 没有多余的注释
✅ 程序正常运行
✅ 功能完全不受影响
✅ ACF对比功能仍然可用（折线图）
```

---

## 需要帮助吗？

如果删除后出现问题，可以：

1. 使用Git恢复：`git checkout your_file.py`
2. 从备份恢复
3. 重新参考本指南逐个删除
