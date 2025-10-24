# 快速修复版本3：改进超参数优化
# 替换原代码中的超参数优化部分

# ========== 改进的超参数优化配置 ==========

# 方案1：适中的优化（推荐用于快速实验）
if OPTUNA_AVAILABLE:
    print("⚡ 使用改进的超参数寻优...")
    optimizer = SimpleHyperparameterOptimizer(
        train_dataset, val_dataset, 
        n_trials=10,              # 2 → 10
        ultra_fast_mode=False     # 关闭极速模式
    )
    optimizer.ultra_fast_epochs = 30  # 5 → 30
    optimizer.ultra_fast_eval_samples = 10  # 3 → 10
    best_params = optimizer.optimize()
else:
    # 改进的默认参数
    best_params = {
        'hidden_dim': 256,
        'z_dim': 64,
        'lr_G': 2e-4,
        'lr_D': 1e-4,
        'batch_size': 32,        # 64 → 32（提高稳定性）
        'n_critic': 3,           # 5 → 3（加快G更新）
        'lambda_gp': 10,
        'noise_sigma': 0.2,
        # 平衡的损失权重
        'div_loss_weight': 1.5,      # 0.3 → 1.5
        'intra_div_weight': 2.0,     # 0.4 → 2.0
        'temporal_weight': 0.3,      # 0.5 → 0.3
        'autocorr_weight': 0.5,      # 0.5 → 0.5
        'freq_weight': 0.2,          # 0.4 → 0.2
    }


# ========== 改进的训练参数 ==========
epochs = 1000              # 2000 → 1000（先看效果）
batch_size = 32            # 64 → 32
n_critic = 3               # 5 → 3
early_stopping_patience = 100  # 50 → 100


# ========== 动态学习率调整策略 ==========
def dynamic_lr_adjustment(opt_G, opt_D, avg_wasserstein, epoch, 
                         w_threshold=0.3, adjustment_factor=0.9):
    """
    动态学习率调整
    
    当W距离过大时，降低学习率以提高稳定性
    """
    if epoch > 100 and avg_wasserstein > w_threshold:
        for param_group in opt_G.param_groups:
            param_group['lr'] *= adjustment_factor
        for param_group in opt_D.param_groups:
            param_group['lr'] *= adjustment_factor
        
        print(f"  📉 动态调整学习率: G_lr={opt_G.param_groups[0]['lr']:.6f}, "
              f"D_lr={opt_D.param_groups[0]['lr']:.6f}")
        
        return True
    return False


# ========== 改进的早停策略 ==========
class ImprovedEarlyStopping:
    """改进的早停策略：同时监控W距离和MAPE"""
    
    def __init__(self, patience=100, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_wasserstein = float('inf')
        self.best_mape = float('inf')
        self.should_stop = False
    
    def __call__(self, avg_wasserstein, avg_mape=None):
        # 综合评分：W距离为主，MAPE为辅
        if avg_mape is not None:
            score = avg_wasserstein + 0.01 * avg_mape  # MAPE权重较小
        else:
            score = avg_wasserstein
        
        # 检查是否改善
        if score < self.best_wasserstein - self.min_delta:
            self.best_wasserstein = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  🛑 早停触发! 连续{self.patience}轮无改善")
                print(f"  📊 最佳W距离: {self.best_wasserstein:.4f}")
                return True
        
        return False


# ========== 使用方法 ==========
"""
# 1. 在训练前创建早停对象：
early_stopping = ImprovedEarlyStopping(patience=100, min_delta=0.001)

# 2. 在训练循环中使用：
for epoch in range(1, epochs + 1):
    # ... 训练代码 ...
    
    if epoch % 20 == 0:
        avg_loss_G = epoch_loss_G / batch_count
        avg_loss_D = epoch_loss_D / batch_count
        avg_wasserstein = epoch_wasserstein / batch_count
        
        # 动态学习率调整
        dynamic_lr_adjustment(opt_G, opt_D, avg_wasserstein, epoch)
        
        # 早停检查
        if early_stopping(avg_wasserstein):
            print("训练提前结束（早停机制）")
            break
"""


# ========== 改进的超参数优化目标函数 ==========
def improved_objective_function(trial, train_dataset, val_dataset):
    """
    改进的超参数优化目标函数
    
    关键改进：
    1. 增加训练轮数（5 → 30）
    2. 增加评估样本数（3 → 10）
    3. 使用综合评分（W距离 + 多样性）
    """
    # 超参数空间
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
        'z_dim': trial.suggest_categorical('z_dim', [32, 64, 100]),
        'lr_G': trial.suggest_float('lr_G', 1e-5, 5e-4, log=True),
        'lr_D': trial.suggest_float('lr_D', 1e-5, 5e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'n_critic': trial.suggest_categorical('n_critic', [1, 3, 5]),
        'lambda_gp': trial.suggest_float('lambda_gp', 5, 15),
        'noise_sigma': trial.suggest_float('noise_sigma', 0.1, 0.3),
    }
    
    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(z_dim=params['z_dim'], hidden=params['hidden_dim']).to(device)
    D = Discriminator(hidden=params['hidden_dim']).to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=params['lr_G'], betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=params['lr_D'], betas=(0.5, 0.9))
    
    dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    # 训练30轮（而不是5轮）
    for epoch in range(30):
        for batch_idx, (real_curves, forecast_curves, season_vec) in enumerate(dataloader):
            if batch_idx >= 10:  # 每轮最多10个batch
                break
            
            real_curves = real_curves.to(device)
            forecast_curves = forecast_curves.to(device)
            season_vec = season_vec.to(device)
            batch_size = real_curves.size(0)
            
            # 更新判别器
            for _ in range(params['n_critic']):
                z = torch.randn(batch_size, params['z_dim'], device=device) * params['noise_sigma']
                fake_curves = G(z, season_vec, forecast_curves, 0).detach()
                d_real, aux_real = D(real_curves, season_vec, forecast_curves)
                d_fake, aux_fake = D(fake_curves, season_vec, forecast_curves)
                
                loss_D = d_fake.mean() - d_real.mean()
                gp = gradient_penalty(D, real_curves, fake_curves, season_vec, forecast_curves)
                aux_loss = F.cross_entropy(aux_real, season_vec.argmax(dim=1))
                
                opt_D.zero_grad()
                (loss_D + params['lambda_gp'] * gp + aux_loss).backward()
                opt_D.step()
            
            # 更新生成器
            z = torch.randn(batch_size, params['z_dim'], device=device) * params['noise_sigma']
            gen_curves = G(z, season_vec, forecast_curves, 0)
            d_gen, aux_gen = D(gen_curves, season_vec, forecast_curves)
            
            loss_G = -d_gen.mean()
            aux_loss_G = F.cross_entropy(aux_gen, season_vec.argmax(dim=1))
            
            # 简化的损失（快速优化）
            mse_loss = F.mse_loss(gen_curves, real_curves)
            total_loss_G = loss_G + aux_loss_G + 2.0 * mse_loss
            
            opt_G.zero_grad()
            total_loss_G.backward()
            opt_G.step()
    
    # 评估（生成10个样本而不是3个）
    G.eval()
    with torch.no_grad():
        z = torch.randn(40, params['z_dim'], device=device) * params['noise_sigma']
        season_vec = torch.eye(4, device=device).repeat(10, 1)
        
        # 使用验证集的预测数据
        forecast_samples = []
        for _ in range(10):
            random_idx = torch.randint(0, len(val_dataset), (4,))
            forecast_samples.append(val_dataset.forecast_curves[random_idx])
        forecast_all = torch.cat(forecast_samples, dim=0).to(device)
        
        fake_curves = G(z, season_vec, forecast_all, 0).detach().cpu().numpy()
        
        # 计算多样性评分
        diversity_score = 0
        for season in range(4):
            season_curves = fake_curves[season*10:(season+1)*10]
            if len(season_curves) > 1:
                diff_matrix = np.abs(season_curves[:, np.newaxis, :] - season_curves[np.newaxis, :, :])
                upper_tri_mask = np.triu(np.ones((len(season_curves), len(season_curves))), k=1).astype(bool)
                diversity_score += np.mean(diff_matrix[upper_tri_mask])
        
        diversity_score /= 4
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return diversity_score  # 目标：最大化多样性


# ========== 完整的超参数优化示例 ==========
"""
if OPTUNA_AVAILABLE:
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: improved_objective_function(trial, train_dataset, val_dataset),
        n_trials=10,
        show_progress_bar=True
    )
    
    print(f"✅ 超参数寻优完成！最佳分数: {study.best_value:.4f}")
    best_params = study.best_params
    
    # 添加固定参数
    best_params.update({
        'lambda_gp': 10,
        'noise_sigma': 0.2,
        'div_loss_weight': 1.5,
        'intra_div_weight': 2.0,
        'temporal_weight': 0.3,
        'autocorr_weight': 0.5,
        'freq_weight': 0.2,
    })
"""
