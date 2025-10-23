# config.py - 集中管理所有配置参数

import torch

class Config:
    """CWGAN-GP 配置类"""
    
    # ========== 数据路径 ==========
    REAL_DATA_PATH = r'E:\python\data\data_2024.07.01-2025.07.01.xlsm'
    FORECAST_DATA_PATH = r'E:\python\data\data_2024.07.01-2025.07.01_forecast.xlsm'
    
    # ========== 随机种子 ==========
    SEED = 42
    COMPARISON_SEED = None  # None表示随机，整数表示固定
    CANDIDATE_RANGE = 0.15  # 曲线选择候选范围
    
    # ========== 设备配置 ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== 模型超参数 ==========
    Z_DIM = 64              # 噪声维度
    HIDDEN_DIM = 256        # 隐藏层维度
    SEASON_DIM = 4          # 季节维度
    FORECAST_DIM = 96       # 预测数据维度
    
    # ========== 训练超参数 ==========
    EPOCHS = 1500
    BATCH_SIZE = 64
    LR_G = 2e-4            # 生成器学习率
    LR_D = 1e-4            # 判别器学习率
    BETA1 = 0.5
    BETA2 = 0.9
    N_CRITIC = 5           # 判别器更新次数
    
    # ========== 损失权重 ==========
    LAMBDA_GP = 10          # 梯度惩罚权重
    NOISE_SIGMA = 0.2       # 噪声强度
    
    # 相似性损失权重
    SIMILARITY_WEIGHT = 5.0
    MAGNITUDE_WEIGHT = 5.0
    POINTWISE_WEIGHT = 3.0
    SEGMENT_WEIGHT = 2.0
    PEAK_VALLEY_WEIGHT = 2.0
    DIST_WEIGHT = 1.5
    
    # 多样性损失权重
    DIV_LOSS_WEIGHT = 0.5
    INTRA_DIV_WEIGHT = 0.8
    
    # 时序和频域权重
    TEMPORAL_WEIGHT = 0.02
    FREQ_WEIGHT = 0.02
    
    # 物理约束权重
    WIND_PHYSICAL_WEIGHT = 2.0
    WIND_PHYSICAL_V2_WEIGHT = 5.0
    WIND_CONSTRAINT_WEIGHT = 10.0
    
    # ========== 早停参数 ==========
    EARLY_STOPPING_PATIENCE = 50
    
    # ========== 超参数优化 ==========
    USE_OPTUNA = True
    N_TRIALS = 2
    ULTRA_FAST_MODE = True
    
    # ========== 季节名称 ==========
    SEASON_NAMES = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    # ========== 季节权重配置 ==========
    SEASON_CONFIGS = {
        0: {  # Spring
            'div_loss_weight': 0.8,
            'intra_div_weight': 1.0,
            'temporal_weight': 0.1,
            'quality_weight': 1.0,
        },
        1: {  # Summer
            'div_loss_weight': 0.3,
            'intra_div_weight': 0.4,
            'temporal_weight': 0.2,
            'quality_weight': 1.5,
        },
        2: {  # Autumn
            'div_loss_weight': 0.8,
            'intra_div_weight': 1.0,
            'temporal_weight': 0.1,
            'quality_weight': 1.0,
        },
        3: {  # Winter
            'div_loss_weight': 0.6,
            'intra_div_weight': 0.7,
            'temporal_weight': 0.15,
            'quality_weight': 1.2,
        }
    }
    
    @classmethod
    def set_seed(cls):
        """设置所有随机种子"""
        import random
        import numpy as np
        
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("🔧 CWGAN-GP 配置")
        print("=" * 60)
        print(f"设备: {cls.DEVICE}")
        print(f"训练轮数: {cls.EPOCHS}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: G={cls.LR_G:.6f}, D={cls.LR_D:.6f}")
        print(f"模型维度: Z={cls.Z_DIM}, Hidden={cls.HIDDEN_DIM}")
        print("=" * 60)
