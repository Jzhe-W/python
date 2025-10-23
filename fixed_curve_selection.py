# 改进的曲线选择策略 - 解决每次选择同一条曲线的问题

import numpy as np
import time

# ========== 配置选项 ==========
COMPARISON_SEED = None  # 设为None表示随机，设为整数则固定
CANDIDATE_RANGE = 0.15  # 中位数±15%范围内都是候选曲线

# ========== 在结果对比阶段之前添加 ==========
print("\n" + "=" * 60)
print("🎯 真实曲线选择策略配置")
print("=" * 60)

if COMPARISON_SEED is not None:
    np.random.seed(COMPARISON_SEED)
    print(f"✅ 使用固定对比种子: {COMPARISON_SEED}")
    print("   每次运行将选择相同的真实曲线（可复现）")
else:
    comparison_seed = int(time.time() * 1000) % 100000
    np.random.seed(comparison_seed)
    print(f"🎲 使用随机对比种子: {comparison_seed}")
    print("   每次运行将选择不同的真实曲线（探索性）")
    print(f"   如需复现此次结果，设置 COMPARISON_SEED = {comparison_seed}")

print(f"📊 候选范围: 中位数±{CANDIDATE_RANGE*100:.0f}%")
print("=" * 60 + "\n")

# ========== 改进的曲线选择函数 ==========
def select_representative_curve(pool_data, selection_strategy='random_from_median'):
    """
    选择代表性曲线
    
    Parameters:
    -----------
    pool_data : np.ndarray
        候选曲线池 [n_samples, 96]
    selection_strategy : str
        'random_from_median': 从接近中位数的曲线中随机选择（推荐）
        'median': 选择最接近中位数的（原策略）
        'random': 完全随机选择
        'max': 选择变化最剧烈的
        'min': 选择变化最平缓的
    
    Returns:
    --------
    target_idx : int
        选中的曲线索引
    selection_info : dict
        选择过程的详细信息
    """
    # 计算综合评分
    variances = np.var(pool_data, axis=1)
    ranges = np.ptp(pool_data, axis=1)
    stds = np.std(pool_data, axis=1)
    composite_score = 0.3 * variances + 0.5 * ranges + 0.2 * stds
    
    median_score = np.median(composite_score)
    
    # 根据策略选择
    if selection_strategy == 'random_from_median':
        # 找到得分在中位数±range范围内的所有曲线
        score_range = CANDIDATE_RANGE * median_score
        candidates_mask = np.abs(composite_score - median_score) <= score_range
        candidates_indices = np.where(candidates_mask)[0]
        
        if len(candidates_indices) == 0:
            # 如果没有候选，回退到最接近的
            target_idx = np.argmin(np.abs(composite_score - median_score))
            n_candidates = 1
            print(f"    ⚠️  没有找到候选曲线，使用最接近中位数的")
        else:
            # 从候选中随机选择
            target_idx = np.random.choice(candidates_indices)
            n_candidates = len(candidates_indices)
        
        selection_info = {
            'strategy': 'random_from_median',
            'n_candidates': n_candidates,
            'median_score': median_score,
            'selected_score': composite_score[target_idx],
            'score_diff': abs(composite_score[target_idx] - median_score)
        }
        
    elif selection_strategy == 'median':
        # 原策略：选择最接近中位数的
        target_idx = np.argmin(np.abs(composite_score - median_score))
        selection_info = {
            'strategy': 'median',
            'median_score': median_score,
            'selected_score': composite_score[target_idx]
        }
        
    elif selection_strategy == 'random':
        # 完全随机
        target_idx = np.random.choice(len(pool_data))
        selection_info = {
            'strategy': 'random',
            'selected_score': composite_score[target_idx]
        }
        
    elif selection_strategy == 'max':
        # 变化最剧烈
        target_idx = np.argmax(composite_score)
        selection_info = {
            'strategy': 'max',
            'selected_score': composite_score[target_idx]
        }
        
    elif selection_strategy == 'min':
        # 变化最平缓
        target_idx = np.argmin(composite_score)
        selection_info = {
            'strategy': 'min',
            'selected_score': composite_score[target_idx]
        }
    
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    # 添加通用信息
    selection_info.update({
        'variance': variances[target_idx],
        'range': ranges[target_idx],
        'std': stds[target_idx]
    })
    
    return target_idx, selection_info


# ========== 在原代码的结果对比循环中替换 ==========
"""
原代码：
    target_idx = closest_to_median
    target_real_curve = pool_original[target_idx]

替换为：
    target_idx, selection_info = select_representative_curve(
        pool_original, 
        selection_strategy='random_from_median'  # 可改为其他策略
    )
    target_real_curve = pool_original[target_idx]
    
    # 打印选择信息
    print(f"    📌 选择策略: {selection_info['strategy']}")
    if 'n_candidates' in selection_info:
        print(f"    📊 候选曲线数: {selection_info['n_candidates']}")
    print(f"    📈 综合评分: {selection_info['selected_score']:.4f}")
    print(f"       方差={selection_info['variance']:.4f}, "
          f"极差={selection_info['range']:.4f}, "
          f"标准差={selection_info['std']:.4f}")
"""

# ========== 多策略对比示例 ==========
def compare_multiple_strategies(pool_data, season_name):
    """对比不同选择策略的效果"""
    strategies = ['random_from_median', 'median', 'max', 'min', 'random']
    
    print(f"\n{'='*60}")
    print(f"🔍 {season_name} - 多策略对比")
    print(f"{'='*60}")
    
    selected_curves = {}
    for strategy in strategies:
        idx, info = select_representative_curve(pool_data, strategy)
        selected_curves[strategy] = (idx, info)
        
        print(f"\n策略: {strategy}")
        print(f"  索引: {idx}")
        print(f"  评分: {info['selected_score']:.4f}")
        if 'n_candidates' in info:
            print(f"  候选数: {info['n_candidates']}")
    
    return selected_curves


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    mock_data = np.random.randn(20, 96) * 100 + 500
    
    print("\n🧪 测试曲线选择功能\n")
    
    # 测试1：默认策略
    idx1, info1 = select_representative_curve(mock_data)
    print(f"✅ 测试1 - 默认策略")
    print(f"   选中索引: {idx1}, 候选数: {info1['n_candidates']}")
    
    # 测试2：再次运行（如果是随机种子，应该不同）
    idx2, info2 = select_representative_curve(mock_data)
    print(f"\n✅ 测试2 - 再次运行")
    print(f"   选中索引: {idx2}, 候选数: {info2['n_candidates']}")
    print(f"   是否相同: {'是' if idx1 == idx2 else '否'}")
    
    # 测试3：多策略对比
    compare_multiple_strategies(mock_data, "Spring")
