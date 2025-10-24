# 快速修复版本4：简化归一化方案
# 替换原代码中的双重归一化（Z-score + MinMax）为单一MinMax

# ========== 简化的归一化方案 ==========

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def simplified_normalization(data, forecast_data):
    """
    简化的归一化方案：只使用MinMax归一化
    
    原问题：
    1. Z-score + MinMax双重归一化导致信息损失
    2. 季节特定的Z-score参数复杂且容易出错
    3. 反归一化时索引错误
    
    改进方案：
    - 只使用MinMax归一化到[0, 1]
    - 简化反归一化流程
    - 避免季节索引错误
    """
    print("🔄 简化归一化方案（只使用MinMax）...")
    
    # 1. 直接使用MinMax归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 归一化真实数据
    data_norm = pd.DataFrame(
        scaler.fit_transform(data.values),
        index=data.index,
        columns=data.columns
    )
    
    # 归一化预测数据（使用相同的scaler）
    forecast_data_norm = pd.DataFrame(
        scaler.transform(forecast_data.values),
        index=forecast_data.index,
        columns=forecast_data.columns
    )
    
    print(f"✅ 归一化完成")
    print(f"  真实数据范围: [{data_norm.min().min():.4f}, {data_norm.max().max():.4f}]")
    print(f"  预测数据范围: [{forecast_data_norm.min().min():.4f}, {forecast_data_norm.max().max():.4f}]")
    
    return data_norm, forecast_data_norm, scaler


def simplified_denormalization(fake_curves, scaler):
    """
    简化的反归一化
    
    输入：fake_curves [4, 96] - 生成的4个季节样本
    输出：fake_denorm [4, 96] - 反归一化后的样本
    """
    # 转置为 (96, 4) 进行批量处理
    fake_reshaped = fake_curves.T  # (96, 4)
    
    # 反归一化
    fake_denorm = scaler.inverse_transform(fake_reshaped)
    
    # 转置回 (4, 96)
    fake_denorm = fake_denorm.T
    
    # 确保风电出力非负
    fake_denorm = np.clip(fake_denorm, 0, None)
    
    print(f"✅ 反归一化完成，范围: [{fake_denorm.min():.2f}, {fake_denorm.max():.2f}]")
    
    return fake_denorm


# ========== 使用方法 ==========
"""
# 替换原代码中的归一化部分（约第100-200行）：

# 原代码：
# 季节特定的归一化策略
data_zscore = data.copy()
for i, season in enumerate(season_names):
    season_indices = season_indices_dict[i]
    season_mean = season_stats[season]['mean']
    season_std = season_stats[season]['std']
    data_zscore.iloc[:, season_indices] = (data.iloc[:, season_indices] - season_mean) / (season_std + 1e-8)

scaler = MinMaxScaler(feature_range=(0, 1))
data_norm = pd.DataFrame(scaler.fit_transform(data_zscore.values), ...)

# 替换为：
data_norm, forecast_data_norm, scaler = simplified_normalization(data, forecast_data)


# 替换原代码中的反归一化部分（约第1600行）：

# 原代码：
fake_reshaped = fake_all.T
for i, season in enumerate(season_names):
    season_fake = fake_reshaped[:, i]
    season_indices = season_indices_dict[i]
    season_zscore_min = zscore_min[season_indices].mean()
    season_zscore_max = zscore_max[season_indices].mean()
    ...

# 替换为：
fake_denorm = simplified_denormalization(fake_all, scaler)
"""


# ========== 完整的简化流程示例 ==========
def complete_simplified_pipeline():
    """完整的简化数据处理流程"""
    
    # 1. 读取数据
    data = pd.read_excel('data_2024.07.01-2025.07.01.xlsm')
    forecast_data = pd.read_excel('data_2024.07.01-2025.07.01_forecast.xlsm')
    
    # 2. 简化归一化
    data_norm, forecast_data_norm, scaler = simplified_normalization(data, forecast_data)
    
    # 3. 构造季节标签（保持不变）
    days = data_norm.columns.tolist()
    season_labels = np.zeros((len(days), 4), dtype=np.float32)
    indices = np.arange(1, len(days) + 1)
    
    summer_mask = (indices >= 1) & (indices <= 62) | (indices >= 336) & (indices <= 365)
    autumn_mask = (indices >= 63) & (indices <= 153)
    winter_mask = (indices >= 154) & (indices <= 243)
    spring_mask = (indices >= 244) & (indices <= 335)
    
    season_labels[summer_mask, 1] = 1
    season_labels[autumn_mask, 2] = 1
    season_labels[winter_mask, 3] = 1
    season_labels[spring_mask, 0] = 1
    
    # 4. 数据分割（保持不变）
    season_indices = {s: np.where(season_labels[:, s] == 1)[0] for s in range(4)}
    
    real_train_idx = []
    real_val_idx = []
    for s in range(4):
        indices = season_indices[s]
        perm = np.random.permutation(indices)
        val_count = int(len(indices) * 0.2)
        real_val_idx.extend(perm[:val_count])
        real_train_idx.extend(perm[val_count:])
    
    real_train_idx = np.array(real_train_idx)
    real_val_idx = np.array(real_val_idx)
    
    # 5. 构造数据集
    real_train_data = data_norm.iloc[:, real_train_idx]
    real_train_labels = season_labels[real_train_idx]
    
    forecast_train_data = forecast_data_norm.iloc[:, real_train_idx]
    forecast_train_labels = season_labels[real_train_idx]
    
    # 6. 创建Dataset（保持不变）
    train_dataset = ForecastIntegratedDataset(
        real_train_data, 
        forecast_train_data, 
        real_train_labels
    )
    
    # 7. 训练模型（保持不变）
    # ... 训练代码 ...
    
    # 8. 生成样本
    G.eval()
    with torch.no_grad():
        z_all = torch.randn(4, z_dim, device=device) * noise_sigma
        season_eye = torch.eye(4, device=device)
        
        if USE_CONDITIONAL_GAN:
            forecast_samples = []
            for season in range(4):
                random_idx = torch.randint(0, len(val_dataset), (1,))
                forecast_samples.append(val_dataset.forecast_curves[random_idx])
            forecast_all = torch.cat(forecast_samples, dim=0).to(device)
            fake_all = G(z_all, season_eye, forecast_all, 0).detach().cpu().numpy()
        else:
            fake_all = G(z_all, season_eye, 0).detach().cpu().numpy()
    
    # 9. 简化反归一化
    fake_denorm = simplified_denormalization(fake_all, scaler)
    
    # 10. 对比分析（保持不变）
    # ... 对比代码 ...
    
    return fake_denorm


# ========== 额外的数据质量检查 ==========
def validate_normalization(data_norm, data_original, scaler):
    """验证归一化的正确性"""
    print("\n🔍 归一化验证:")
    
    # 1. 检查归一化范围
    assert data_norm.min().min() >= 0, "归一化后最小值 < 0"
    assert data_norm.max().max() <= 1, "归一化后最大值 > 1"
    print("✅ 归一化范围正确: [0, 1]")
    
    # 2. 检查反归一化
    data_denorm = scaler.inverse_transform(data_norm.values)
    reconstruction_error = np.abs(data_denorm - data_original.values).mean()
    print(f"✅ 反归一化重构误差: {reconstruction_error:.6f}")
    
    if reconstruction_error > 1e-6:
        print("⚠️  警告：重构误差较大，可能存在数值精度问题")
    
    # 3. 检查数据分布
    print(f"  原始数据: 均值={data_original.values.mean():.2f}, "
          f"标准差={data_original.values.std():.2f}")
    print(f"  归一化数据: 均值={data_norm.values.mean():.4f}, "
          f"标准差={data_norm.values.std():.4f}")
    print(f"  重构数据: 均值={data_denorm.mean():.2f}, "
          f"标准差={data_denorm.std():.2f}")
    
    return reconstruction_error < 1e-6
