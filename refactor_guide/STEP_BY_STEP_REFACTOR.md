# 🔧 模块化重构：分步指南

## 🎯 目标

将2500行的单文件代码拆分为10+个模块，提高可维护性。

---

## 📝 准备工作

### **1. 备份原代码**
```bash
cp your_program.py your_program_backup.py
```

### **2. 创建项目文件夹**
```bash
mkdir cwgan_project
cd cwgan_project
```

### **3. 创建目录结构**
```bash
mkdir -p models losses data training evaluation utils
touch models/__init__.py losses/__init__.py data/__init__.py
touch training/__init__.py evaluation/__init__.py utils/__init__.py
```

---

## 🔨 步骤1：提取配置（最简单）

### **1.1 创建 config.py**

```python
# config.py
import torch

class Config:
    # 数据路径
    REAL_DATA_PATH = r'E:\python\data\data_2024.07.01-2025.07.01.xlsm'
    FORECAST_DATA_PATH = r'E:\python\data\data_2024.07.01-2025.07.01_forecast.xlsm'
    
    # 随机种子
    SEED = 42
    COMPARISON_SEED = None
    CANDIDATE_RANGE = 0.15
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数
    Z_DIM = 64
    HIDDEN_DIM = 256
    SEASON_DIM = 4
    
    # 训练参数
    EPOCHS = 1500
    BATCH_SIZE = 64
    LR_G = 2e-4
    LR_D = 1e-4
    
    # 季节名称
    SEASON_NAMES = ['Spring', 'Summer', 'Autumn', 'Winter']
```

### **1.2 在原文件中查找并迁移**

**原文件中搜索：**
```python
epochs = 1500          → Config.EPOCHS
batch_size = 64        → Config.BATCH_SIZE
lr_G = 2e-4           → Config.LR_G
z_dim = 64            → Config.Z_DIM
```

### **1.3 测试配置模块**

```python
# test_config.py
from config import Config

print(f"训练轮数: {Config.EPOCHS}")
print(f"设备: {Config.DEVICE}")
# 应该正常输出
```

---

## 🔨 步骤2：提取模型定义

### **2.1 从原文件复制生成器代码**

**原文件位置：** 第405-640行

**操作步骤：**
1. 选中 `class ResidualBlock` 到 `class Generator` 结束
2. 复制到 `models/generator.py`
3. 添加必要的import

```python
# models/generator.py
import torch
import torch.nn as nn

# 粘贴 ResidualBlock 类
# 粘贴 ConditionalGenerator 类
# 粘贴 Generator 类
```

### **2.2 从原文件复制判别器代码**

**原文件位置：** 第640-720行

**操作步骤：**
1. 选中 `class ResidualBlockCond` 到 `class Discriminator` 结束
2. 复制到 `models/discriminator.py`

### **2.3 创建 models/__init__.py**

```python
# models/__init__.py
from .generator import ConditionalGenerator, Generator
from .discriminator import ConditionalDiscriminator, Discriminator

__all__ = [
    'ConditionalGenerator',
    'Generator', 
    'ConditionalDiscriminator',
    'Discriminator'
]
```

### **2.4 测试模型模块**

```python
# test_models.py
import torch
from models import ConditionalGenerator, ConditionalDiscriminator

G = ConditionalGenerator(z_dim=64, hidden=256)
D = ConditionalDiscriminator(hidden=256)

z = torch.randn(4, 64)
c = torch.eye(4)
forecast = torch.randn(4, 96)

output = G(z, c, forecast)
print(f"✅ 生成器输出形状: {output.shape}")

d_out, aux_out = D(output, c, forecast)
print(f"✅ 判别器输出形状: {d_out.shape}, {aux_out.shape}")
```

---

## 🔨 步骤3：提取损失函数

### **3.1 分类损失函数**

#### **相似性损失 → losses/similarity_losses.py**

**原文件位置：** 
- `pointwise_mse_loss`（第730行）
- `segment_matching_loss`（第740行）
- `peak_valley_matching_loss`（第770行）
- `distribution_matching_loss`（第800行）
- `enhanced_similarity_loss`（第3110行）
- `magnitude_matching_loss`（第3180行）

**操作：** 复制这6个函数到 `losses/similarity_losses.py`

#### **多样性损失 → losses/diversity_losses.py**

**原文件位置：**
- `simplified_intra_season_diversity_loss`（第1830行）
- `seasonal_diversity_loss`（第1900行）
- `mode_collapse_loss`（第1940行）
- `seasonal_consistency_loss`（第1970行）

**操作：** 复制这4个函数到 `losses/diversity_losses.py`

#### **物理约束损失 → losses/physical_losses.py**

**原文件位置：**
- `wind_power_fluctuation_loss`（第2530行）
- `wind_power_physical_constraints`（第2560行）
- `wind_power_physical_loss`（第2600行）
- `wind_summer_constraint`（第2650行）

**操作：** 复制这4个函数到 `losses/physical_losses.py`

### **3.2 创建 losses/__init__.py**

```python
# losses/__init__.py
from .similarity_losses import (
    pointwise_mse_loss,
    segment_matching_loss,
    peak_valley_matching_loss,
    distribution_matching_loss,
    enhanced_similarity_loss,
    magnitude_matching_loss
)

from .diversity_losses import (
    simplified_intra_season_diversity_loss,
    seasonal_diversity_loss,
    mode_collapse_loss,
    seasonal_consistency_loss
)

from .physical_losses import (
    wind_power_fluctuation_loss,
    wind_power_physical_constraints,
    wind_power_physical_loss,
    wind_summer_constraint
)
```

---

## 🔨 步骤4：提取数据处理

### **4.1 数据集类 → data/dataset.py**

**原文件位置：** 第370-390行

```python
# data/dataset.py
import torch
from torch.utils.data import Dataset

class ForecastIntegratedDataset(Dataset):
    # ... 复制原代码 ...
```

### **4.2 数据预处理 → data/preprocessing.py**

**原文件位置：** 第70-350行

```python
# data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(real_path, forecast_path):
    """加载真实数据和预测数据"""
    data = pd.read_excel(real_path)
    forecast_data = pd.read_excel(forecast_path)
    return data, forecast_data

def create_season_labels(n_days):
    """创建季节标签"""
    # 复制季节标签生成代码
    pass

def normalize_data(data, forecast_data):
    """归一化数据"""
    # 复制归一化代码
    pass

def split_train_val(data, season_labels, val_ratio=0.2):
    """划分训练集和验证集"""
    # 复制数据分割代码
    pass
```

---

## 🔨 步骤5：提取训练逻辑

### **5.1 创建训练器类 → training/trainer.py**

```python
# training/trainer.py
import torch
import torch.nn.functional as F
from torch.autograd import grad

class Trainer:
    def __init__(self, G, D, opt_G, opt_D, config):
        self.G = G
        self.D = D
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.config = config
        
        # 训练历史
        self.loss_G_list = []
        self.loss_D_list = []
        self.wasserstein_list = []
    
    def gradient_penalty(self, D, real, fake, c, forecast=None):
        """梯度惩罚"""
        # 复制 gradient_penalty 函数代码
        pass
    
    def train_discriminator(self, real_curves, forecast_curves, season_vec):
        """训练判别器一步"""
        # 复制判别器训练代码
        pass
    
    def train_generator(self, real_curves, forecast_curves, season_vec):
        """训练生成器一步"""
        # 复制生成器训练代码
        pass
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        for batch_idx, (real, forecast, season) in enumerate(dataloader):
            # 训练D
            loss_D = self.train_discriminator(real, forecast, season)
            
            # 训练G
            loss_G = self.train_generator(real, forecast, season)
            
            epoch_loss_G += loss_G
            epoch_loss_D += loss_D
        
        return epoch_loss_G, epoch_loss_D
    
    def train(self, train_loader, val_loader, epochs):
        """完整训练循环"""
        for epoch in range(1, epochs + 1):
            loss_G, loss_D = self.train_epoch(train_loader, epoch)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: G={loss_G:.4f}, D={loss_D:.4f}")
```

---

## 🔨 步骤6：提取评估和可视化

### **6.1 评估指标 → evaluation/metrics.py**

```python
# evaluation/metrics.py
import numpy as np

def calc_metrics(y_true, y_pred):
    """计算RMSE, MAE, MAPE, Correlation"""
    # 复制 calc_metrics 函数
    pass

def calc_picp_pinaw(real, generated_samples, confidence_level=0.95):
    """计算PICP和PINAW"""
    # 复制 calc_picp_pinaw 函数
    pass
```

### **6.2 可视化 → evaluation/visualization.py**

```python
# evaluation/visualization.py
import matplotlib.pyplot as plt

def plot_training_losses(loss_G_list, loss_D_list, wasserstein_list):
    """绘制训练损失曲线"""
    # 复制绘图代码
    pass

def plot_season_diversity(diversity_samples, season_stats):
    """绘制季节多样性图"""
    # 复制绘图代码
    pass

def plot_acf_comparison(real_acf, fake_acf, season_name):
    """绘制ACF对比图"""
    # 复制ACF绘图代码
    pass
```

---

## 🔨 步骤7：创建主入口

### **7.1 创建 main.py**

```python
# main.py
import torch
from config import Config
from models import ConditionalGenerator, ConditionalDiscriminator
from data.preprocessing import load_data, normalize_data, split_train_val
from data.dataset import ForecastIntegratedDataset
from torch.utils.data import DataLoader
from training.trainer import Trainer

def main():
    # 1. 设置
    Config.set_seed()
    Config.print_config()
    
    # 2. 加载数据
    print("📊 加载数据...")
    data, forecast_data = load_data(Config.REAL_DATA_PATH, Config.FORECAST_DATA_PATH)
    
    # 3. 预处理
    print("🔄 预处理数据...")
    data_norm, forecast_norm, params = normalize_data(data, forecast_data)
    
    # 4. 创建数据集
    print("📦 创建数据集...")
    train_dataset = ForecastIntegratedDataset(...)
    val_dataset = ForecastIntegratedDataset(...)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # 5. 创建模型
    print("🏗️ 创建模型...")
    G = ConditionalGenerator(Config.Z_DIM, Config.HIDDEN_DIM).to(Config.DEVICE)
    D = ConditionalDiscriminator(Config.HIDDEN_DIM).to(Config.DEVICE)
    
    # 6. 创建优化器
    opt_G = torch.optim.Adam(G.parameters(), lr=Config.LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=Config.LR_D)
    
    # 7. 训练
    print("🚀 开始训练...")
    trainer = Trainer(G, D, opt_G, opt_D, Config)
    trainer.train(train_loader, val_loader, Config.EPOCHS)
    
    # 8. 评估
    print("📊 评估模型...")
    # 调用评估函数
    
    print("✅ 完成！")

if __name__ == '__main__':
    main()
```

---

## 📊 迁移对照速查表

### **快速查找：原代码在哪里？**

| 你想找 | 原文件行数 | 新文件位置 | 搜索关键词 |
|--------|-----------|-----------|-----------|
| 配置参数 | 分散各处 | config.py | epochs, batch_size |
| 生成器 | 405-540 | models/generator.py | class Generator |
| 判别器 | 640-720 | models/discriminator.py | class Discriminator |
| 相似性损失 | 730-830 | losses/similarity_losses.py | pointwise_mse |
| 多样性损失 | 1830-2000 | losses/diversity_losses.py | diversity_loss |
| 物理约束 | 2530-2700 | losses/physical_losses.py | wind_power |
| 数据集 | 370-390 | data/dataset.py | class Dataset |
| 数据预处理 | 70-350 | data/preprocessing.py | read_excel |
| 训练循环 | 2900-3400 | training/trainer.py | for epoch in |
| 评估指标 | 3500-3600 | evaluation/metrics.py | calc_metrics |
| 可视化 | 3600-3800 | evaluation/visualization.py | plt.plot |

---

## 🧪 验证重构是否成功

### **检查清单**

```
□ 1. 所有模块都能独立导入
     python -c "from models import ConditionalGenerator"
     
□ 2. 没有循环依赖
     python -m pydeps main.py --show-cycles
     
□ 3. 主程序能正常运行
     python main.py
     
□ 4. 结果与重构前一致
     对比loss值、生成结果等
     
□ 5. 所有功能都保留
     检查GUI、保存模型、ACF对比等
```

---

## 💡 重构技巧

### **技巧1：逐个模块迁移**

```
❌ 不要：一次性拆分所有代码
✅ 推荐：先拆分config.py，测试通过后再拆分models
```

### **技巧2：保持原文件可运行**

```
❌ 不要：边拆边删原文件
✅ 推荐：复制出新模块，原文件保持完整，最后再删除
```

### **技巧3：使用Git跟踪**

```bash
git init
git add .
git commit -m "初始版本（未重构）"

# 每完成一个模块
git add .
git commit -m "重构：添加config.py"
```

---

## 🎯 优先级建议

### **最小可行重构（1小时）**

只拆分这3个：
1. ✅ config.py（配置）
2. ✅ models/（模型定义）
3. ✅ losses/similarity_losses.py（主要损失）

### **标准重构（3小时）**

再添加：
4. ✅ data/（数据处理）
5. ✅ losses/（所有损失）

### **完整重构（5小时）**

全部拆分：
6. ✅ training/（训练逻辑）
7. ✅ evaluation/（评估可视化）

---

## 📝 实际操作示例

### **示例：如何迁移 ConditionalGenerator**

#### **Step 1: 在原文件中找到代码**
```bash
# 使用Ctrl+F搜索
class ConditionalGenerator
```

#### **Step 2: 复制代码范围**
从 `class ConditionalGenerator` 开始  
到 `return torch.clamp(adjusted_outputs, -1, 1)` 结束

#### **Step 3: 创建新文件并粘贴**
```python
# models/generator.py
import torch
import torch.nn as nn

# 粘贴复制的代码
class ConditionalGenerator(nn.Module):
    # ...
```

#### **Step 4: 测试新模块**
```python
from models.generator import ConditionalGenerator
G = ConditionalGenerator(64, 256)
print("✅ 导入成功")
```

#### **Step 5: 在main.py中使用**
```python
# main.py
from models.generator import ConditionalGenerator

G = ConditionalGenerator(config.Z_DIM, config.HIDDEN_DIM)
```

---

## 🚀 开始重构

### **推荐流程：**

```
第1天（1-2小时）：
  ├─ 创建项目结构
  ├─ 提取config.py
  └─ 测试

第2天（2-3小时）：
  ├─ 提取models/
  ├─ 提取losses/
  └─ 测试

第3天（1-2小时）：
  ├─ 提取data/
  ├─ 提取training/
  └─ 测试

完成！🎉
```

---

## 需要帮助？

1. **遇到导入错误？** 检查 `__init__.py` 文件
2. **代码找不到？** 使用对照表查找原位置
3. **不确定怎么拆？** 先看示例代码

我可以帮您实现任何步骤！
