# 📖 完整重构示例：手把手教程

## 示例：重构生成器模块

### **原文件（2500行）**

```python
# your_program.py（部分代码）

# ... 前面400行 ...

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
            nn.Linear(out_features, out_features)
        )
        # ... 更多代码 ...

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=100, season_dim=4, forecast_dim=96, hidden=512):
        super().__init__()
        # ... 更多代码 ...

# ... 后面2000行 ...
```

---

### **步骤1：创建新文件**

```bash
# 在项目根目录执行
mkdir -p models
touch models/__init__.py
touch models/generator.py
```

---

### **步骤2：复制代码到新文件**

**models/generator.py：**

```python
"""
生成器模块
包含 ResidualBlock 和两个生成器类
"""

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
        # ... （复制所有代码）
    
    def forward(self, z, c, forecast, training_step=0):
        # ... （复制所有代码）
        return torch.clamp(adjusted_outputs, -1, 1)


class Generator(nn.Module):
    """基础生成器（不使用预测数据）"""
    
    def __init__(self, z_dim=100, season_dim=4, hidden=512):
        # ... （复制所有代码）
    
    def forward(self, z, c, training_step=0):
        # ... （复制所有代码）
        return torch.clamp(adjusted_outputs, -1, 1)
```

---

### **步骤3：创建 __init__.py**

**models/__init__.py：**

```python
"""
模型模块
导出所有模型类
"""

from .generator import ConditionalGenerator, Generator, ResidualBlock
from .discriminator import ConditionalDiscriminator, Discriminator, ResidualBlockCond

__all__ = [
    'ConditionalGenerator',
    'Generator',
    'ResidualBlock',
    'ConditionalDiscriminator',
    'Discriminator',
    'ResidualBlockCond'
]
```

---

### **步骤4：测试新模块**

**test_generator.py：**

```python
"""测试生成器模块是否正常工作"""

import torch
from models.generator import ConditionalGenerator

def test_conditional_generator():
    print("🧪 测试 ConditionalGenerator...")
    
    # 创建模型
    G = ConditionalGenerator(z_dim=64, hidden=256)
    
    # 准备输入
    batch_size = 4
    z = torch.randn(batch_size, 64)
    c = torch.eye(4)
    forecast = torch.randn(batch_size, 96)
    
    # 前向传播
    output = G(z, c, forecast)
    
    # 验证输出
    assert output.shape == (batch_size, 96), f"输出形状错误: {output.shape}"
    assert output.min() >= -1 and output.max() <= 1, "输出超出[-1, 1]范围"
    
    print(f"✅ ConditionalGenerator测试通过")
    print(f"   输出形状: {output.shape}")
    print(f"   输出范围: [{output.min():.4f}, {output.max():.4f}]")

if __name__ == '__main__':
    test_conditional_generator()
```

**运行测试：**
```bash
python test_generator.py
```

**预期输出：**
```
🧪 测试 ConditionalGenerator...
✅ ConditionalGenerator测试通过
   输出形状: torch.Size([4, 96])
   输出范围: [-0.9234, 0.8756]
```

---

### **步骤5：在主程序中使用**

**修改原文件或创建新的main.py：**

```python
# 原代码（修改前）
# G = ConditionalGenerator(z_dim=best_params['z_dim'], hidden=best_params['hidden_dim']).to(device)

# 新代码（修改后）
from models import ConditionalGenerator
G = ConditionalGenerator(
    z_dim=best_params['z_dim'], 
    hidden=best_params['hidden_dim']
).to(device)
```

---

## 🎯 完整迁移示例：损失函数模块

### **原文件中的代码（分散）**

```python
# 原文件第730行
def pointwise_mse_loss(gen_curves, real_curves, weight=3.0):
    return weight * F.mse_loss(gen_curves, real_curves)

# 原文件第740行
def segment_matching_loss(gen_curves, real_curves, weight=2.0):
    # ... 代码 ...

# 原文件第3110行
def enhanced_similarity_loss(fake_curves, real_curves, weight=1.5):
    # ... 代码 ...
```

### **新文件（集中）**

**losses/similarity_losses.py：**

```python
"""
相似性损失函数
包含所有用于提升生成数据与真实数据相似性的损失
"""

import torch
import torch.nn.functional as F


def pointwise_mse_loss(gen_curves, real_curves, weight=3.0):
    """逐点MSE损失：确保每个时间点都接近真实数据"""
    return weight * F.mse_loss(gen_curves, real_curves)


def segment_matching_loss(gen_curves, real_curves, weight=2.0):
    """分段匹配损失：将一天分成多个时段，分别匹配"""
    # ... （复制原代码）


def enhanced_similarity_loss(fake_curves, real_curves, weight=1.5):
    """增强相似性损失：多维度提升生成质量"""
    # ... （复制原代码）


def magnitude_matching_loss(fake_curves, real_curves):
    """量级匹配损失：确保生成数据量级与真实数据匹配"""
    # ... （复制原代码）
```

### **使用新模块**

```python
# 原代码
pointwise_loss = pointwise_mse_loss(gen_curves, real_curves, weight=3.0)
segment_loss = segment_matching_loss(gen_curves, real_curves, weight=2.0)

# 新代码（需要先导入）
from losses.similarity_losses import pointwise_mse_loss, segment_matching_loss

pointwise_loss = pointwise_mse_loss(gen_curves, real_curves, weight=3.0)
segment_loss = segment_matching_loss(gen_curves, real_curves, weight=2.0)
# ⬆️ 使用方式完全一样，只是需要先导入
```

---

## 🎯 最小可行重构（1小时版）

如果时间有限，只做这3个最重要的：

### **第1步：提取配置（15分钟）**
```
✅ 创建 config.py
✅ 迁移所有常量和超参数
```

### **第2步：提取模型（30分钟）**
```
✅ 创建 models/generator.py
✅ 创建 models/discriminator.py
✅ 创建 models/__init__.py
```

### **第3步：提取主要损失（15分钟）**
```
✅ 创建 losses/similarity_losses.py
✅ 创建 losses/__init__.py
```

**完成后的导入示例：**
```python
# main.py（简化版）
from config import Config
from models import ConditionalGenerator, ConditionalDiscriminator
from losses.similarity_losses import pointwise_mse_loss, segment_matching_loss

# 其他代码保持不变，只是添加了这些import
```

---

## 📋 重构对照检查表

```
迁移前检查：
□ 原代码能正常运行
□ 已创建备份
□ Git已提交（可选）

迁移中检查：
□ 每个新模块都能独立导入
□ 没有循环依赖
□ 必要的import都已添加

迁移后检查：
□ 主程序能正常运行
□ 所有功能都保留
□ 结果与重构前一致
□ 代码更易读易维护

清理检查：
□ 删除了原文件中的重复定义（可选）
□ 删除了注释掉的ACF代码
□ 代码格式统一
```

---

## 🆘 常见问题解决

### **问题1：ImportError: No module named 'models'**

**原因：** Python找不到模块

**解决：** 
```python
# 方法1: 添加项目路径
import sys
sys.path.append('/path/to/cwgan_project')

# 方法2: 使用相对导入
from .models import ConditionalGenerator

# 方法3: 在项目根目录运行
cd cwgan_project
python main.py
```

### **问题2：循环依赖错误**

**原因：** A导入B，B又导入A

**解决：** 
```python
# 不好的设计
# models/generator.py
from losses.similarity_losses import pointwise_mse_loss  # ❌

# 好的设计
# models/generator.py 只定义模型，不导入损失
# main.py 中同时导入模型和损失
from models import Generator
from losses.similarity_losses import pointwise_mse_loss  # ✅
```

### **问题3：代码运行结果不一致**

**原因：** 随机种子设置问题

**解决：**
```python
# 在main.py最开始设置
from config import Config
Config.set_seed()
```

---

## 🎉 重构完成标志

当你看到这样的项目结构时，就成功了：

```
cwgan_project/
├── config.py                  ✅ 100行
├── models/
│   ├── __init__.py           ✅ 10行
│   ├── generator.py          ✅ 200行
│   └── discriminator.py      ✅ 100行
├── losses/
│   ├── __init__.py           ✅ 20行
│   ├── similarity_losses.py  ✅ 150行
│   ├── diversity_losses.py   ✅ 180行
│   └── physical_losses.py    ✅ 120行
├── data/
│   ├── __init__.py           ✅ 10行
│   ├── dataset.py            ✅ 50行
│   └── preprocessing.py      ✅ 250行
├── training/
│   ├── __init__.py           ✅ 10行
│   ├── trainer.py            ✅ 300行
│   └── hyperparameter_opt.py ✅ 200行
├── evaluation/
│   ├── __init__.py           ✅ 10行
│   ├── metrics.py            ✅ 80行
│   └── visualization.py      ✅ 350行
└── main.py                    ✅ 150行

总行数：约2200行（比原来更清晰！）
```

每个文件都**短小精悍**，易于理解和维护！🎯
