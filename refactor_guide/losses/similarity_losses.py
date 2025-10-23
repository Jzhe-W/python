# losses/similarity_losses.py - 相似性相关损失函数

import torch
import torch.nn.functional as F

def pointwise_mse_loss(gen_curves, real_curves, weight=3.0):
    """逐点MSE损失：确保每个时间点都接近真实数据"""
    return weight * F.mse_loss(gen_curves, real_curves)


def segment_matching_loss(gen_curves, real_curves, weight=2.0):
    """分段匹配损失：将一天分成多个时段，分别匹配"""
    n_segments = 8
    segment_size = gen_curves.size(1) // n_segments

    total_loss = 0
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size

        gen_segment = gen_curves[:, start:end]
        real_segment = real_curves[:, start:end]

        gen_mean = gen_segment.mean(dim=1)
        real_mean = real_segment.mean(dim=1)
        gen_std = gen_segment.std(dim=1)
        real_std = real_segment.std(dim=1)

        mean_loss = F.mse_loss(gen_mean, real_mean)
        std_loss = F.mse_loss(gen_std, real_std)

        total_loss += (mean_loss + std_loss)

    return weight * total_loss / n_segments


def peak_valley_matching_loss(gen_curves, real_curves, weight=2.0):
    """峰谷匹配损失：确保峰值和谷值位置及数值接近"""
    # 找峰值
    gen_peaks = torch.max(gen_curves, dim=1)[0]
    real_peaks = torch.max(real_curves, dim=1)[0]
    peak_loss = F.mse_loss(gen_peaks, real_peaks)

    # 找谷值
    gen_valleys = torch.min(gen_curves, dim=1)[0]
    real_valleys = torch.min(real_curves, dim=1)[0]
    valley_loss = F.mse_loss(gen_valleys, real_valleys)

    # 峰谷差（幅度）
    gen_range = gen_peaks - gen_valleys
    real_range = real_peaks - real_valleys
    range_loss = F.mse_loss(gen_range, real_range)

    return weight * (peak_loss + valley_loss + range_loss)


def distribution_matching_loss(gen_curves, real_curves, weight=1.5):
    """分布匹配损失：使用直方图匹配确保分布一致"""
    n_bins = 20

    gen_hist = torch.histc(gen_curves.flatten(), bins=n_bins, min=0.0, max=1.0)
    gen_hist = gen_hist / (gen_hist.sum() + 1e-8)

    real_hist = torch.histc(real_curves.flatten(), bins=n_bins, min=0.0, max=1.0)
    real_hist = real_hist / (real_hist.sum() + 1e-8)

    kl_loss = F.kl_div((gen_hist + 1e-8).log(), real_hist, reduction='batchmean')

    return weight * kl_loss


def enhanced_similarity_loss(fake_curves, real_curves, weight=1.5):
    """增强相似性损失：多维度提升生成质量"""
    # 1. 形状相似性
    shape_loss = torch.mean((fake_curves - real_curves) ** 2)

    # 2. 趋势相似性
    fake_diff = fake_curves[:, 1:] - fake_curves[:, :-1]
    real_diff = real_curves[:, 1:] - real_curves[:, :-1]
    trend_loss = torch.mean((fake_diff - real_diff) ** 2)

    # 3. 变化率相似性
    fake_diff2 = fake_diff[:, 1:] - fake_diff[:, :-1]
    real_diff2 = real_diff[:, 1:] - real_diff[:, :-1]
    change_loss = torch.mean((fake_diff2 - real_diff2) ** 2)

    # 4. 统计特性相似性
    fake_mean = torch.mean(fake_curves, dim=1, keepdim=True)
    real_mean = torch.mean(real_curves, dim=1, keepdim=True)
    fake_std = torch.std(fake_curves, dim=1, keepdim=True)
    real_std = torch.std(real_curves, dim=1, keepdim=True)

    stat_loss = torch.mean((fake_mean - real_mean) ** 2) + torch.mean((fake_std - real_std) ** 2)

    total_similarity = (0.5 * shape_loss + 0.3 * trend_loss +
                       0.15 * change_loss + 0.05 * stat_loss)

    return weight * total_similarity


def magnitude_matching_loss(fake_curves, real_curves):
    """量级匹配损失：确保生成数据量级与真实数据匹配"""
    fake_mean = torch.mean(fake_curves)
    fake_std = torch.std(fake_curves)
    real_mean = torch.mean(real_curves)
    real_std = torch.std(real_curves)

    mean_loss = torch.abs(fake_mean - real_mean) / (real_mean + 1e-8)
    std_loss = torch.abs(fake_std - real_std) / (real_std + 1e-8)

    return mean_loss + std_loss
