# ACF图表修改示例 - 删除柱状图，保留折线图

"""
这个文件展示了如何修改ACF图表部分
只需要找到原代码中对应的部分，替换即可
"""

# ============================================
# 修改前的代码（原代码）
# ============================================

def BEFORE_original_code():
    """这是原来的代码 - 有2个子图（折线图+柱状图）"""
    
    # === ACF 数据对比 ===
    from statsmodels.tsa.stattools import acf

    lags = 20  # 减少lag数量，便于观察

    # 计算ACF - 使用生成样本的均值
    real_acf, real_acf_conf = acf(real_curve_original, nlags=lags, alpha=0.05)
    fake_acf, fake_acf_conf = acf(generated_mean, nlags=lags, alpha=0.05)

    # 向量化打印数据对比
    print(f"\n{season_names[i]} ACF 对比 (lag 0-{lags}):")
    print("Lag\tReal ACF\t\tGenerated ACF\t\tDifference")
    print("-" * 60)

    # 向量化计算所有差异
    lag_range = np.arange(lags + 1)
    acf_diffs = np.abs(real_acf - fake_acf)

    # 批量格式化和打印
    for lag, real_val, fake_val, diff_val in zip(lag_range, real_acf, fake_acf, acf_diffs):
        print(f"{lag}\t{real_val:.4f}\t\t{fake_val:.4f}\t\t{diff_val:.4f}")

    # 计算相关性指标
    acf_corr = np.corrcoef(real_acf, fake_acf)[0, 1]
    acf_rmse = np.sqrt(np.mean((real_acf - fake_acf) ** 2))

    print(f"\n{season_names[i]} 相关性统计:")
    print(f"ACF 相关系数: {acf_corr:.4f}")
    print(f"ACF RMSE: {acf_rmse:.4f}")

    # ⬇️⬇️⬇️ 从这里开始修改 ⬇️⬇️⬇️
    # 可视化对比
    # 创建Figure 8+i的框架
    frame_acf = ttk.Frame(notebook)
    notebook.add(frame_acf, text=f"Figure {8 + i}: {season_names[i]}ACF")

    fig = plt.figure(8 + i, figsize=(12, 6))  # ⬅️ 宽度12（2个子图）
    axes = fig.subplots(1, 2)  # ⬅️ 创建2个子图

    # ACF对比图（折线图）- 第1个子图
    axes[0].plot(range(lags + 1), real_acf, 'b-o', label='Real', alpha=0.7)
    axes[0].plot(range(lags + 1), fake_acf, 'r-s', label='Generated', alpha=0.7)
    axes[0].set_title(f'Figure {8 + i}: {season_names[i]} ACF Comparison')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ⬇️⬇️⬇️ 删除下面这个柱状图部分 ⬇️⬇️⬇️
    # ACF差异图（柱状图）- 第2个子图
    axes[1].bar(range(lags + 1), real_acf - fake_acf, alpha=0.7, color='orange')
    axes[1].set_title(f'{season_names[i]} ACF Difference (Real - Generated)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Difference')
    axes[1].grid(True, alpha=0.3)
    # ⬆️⬆️⬆️ 删除到这里 ⬆️⬆️⬆️

    plt.tight_layout()

    # 将Figure 8+i嵌入到框架中
    canvas_acf = FigureCanvasTkAgg(fig, frame_acf)
    canvas_acf.draw()
    canvas_acf.get_tk_widget().pack(fill='both', expand=True)

    # 确保图形正确显示
    plt.close(fig)  # 关闭matplotlib的默认显示

    # 强制刷新显示
    root.update()


# ============================================
# 修改后的代码（新代码）
# ============================================

def AFTER_modified_code():
    """这是修改后的代码 - 只有1个子图（折线图）"""
    
    # === ACF 数据对比 ===
    from statsmodels.tsa.stattools import acf

    lags = 20  # 减少lag数量，便于观察

    # 计算ACF - 使用生成样本的均值
    real_acf, real_acf_conf = acf(real_curve_original, nlags=lags, alpha=0.05)
    fake_acf, fake_acf_conf = acf(generated_mean, nlags=lags, alpha=0.05)

    # 向量化打印数据对比
    print(f"\n{season_names[i]} ACF 对比 (lag 0-{lags}):")
    print("Lag\tReal ACF\t\tGenerated ACF\t\tDifference")
    print("-" * 60)

    # 向量化计算所有差异
    lag_range = np.arange(lags + 1)
    acf_diffs = np.abs(real_acf - fake_acf)

    # 批量格式化和打印
    for lag, real_val, fake_val, diff_val in zip(lag_range, real_acf, fake_acf, acf_diffs):
        print(f"{lag}\t{real_val:.4f}\t\t{fake_val:.4f}\t\t{diff_val:.4f}")

    # 计算相关性指标
    acf_corr = np.corrcoef(real_acf, fake_acf)[0, 1]
    acf_rmse = np.sqrt(np.mean((real_acf - fake_acf) ** 2))

    print(f"\n{season_names[i]} 相关性统计:")
    print(f"ACF 相关系数: {acf_corr:.4f}")
    print(f"ACF RMSE: {acf_rmse:.4f}")

    # ✅✅✅ 修改后的代码 ✅✅✅
    # 可视化对比
    # 创建Figure 8+i的框架
    frame_acf = ttk.Frame(notebook)
    notebook.add(frame_acf, text=f"Figure {8 + i}: {season_names[i]}ACF")

    fig = plt.figure(8 + i, figsize=(10, 6))  # ✅ 改为10（1个子图）
    ax = fig.add_subplot(111)  # ✅ 改为单个子图

    # ACF对比图（折线图）
    ax.plot(range(lags + 1), real_acf, 'b-o', label='Real', alpha=0.7, linewidth=2)  # ✅ axes[0] → ax
    ax.plot(range(lags + 1), fake_acf, 'r-s', label='Generated', alpha=0.7, linewidth=2)
    ax.set_title(f'Figure {8 + i}: {season_names[i]} ACF Comparison', 
                 fontsize=14, fontweight='bold')  # ✅ 可选：增加字体大小
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ❌ 删除了axes[1]的柱状图部分

    plt.tight_layout()

    # 将Figure 8+i嵌入到框架中
    canvas_acf = FigureCanvasTkAgg(fig, frame_acf)
    canvas_acf.draw()
    canvas_acf.get_tk_widget().pack(fill='both', expand=True)

    # 确保图形正确显示
    plt.close(fig)  # 关闭matplotlib的默认显示

    # 强制刷新显示
    root.update()


# ============================================
# 关键修改点总结
# ============================================

"""
修改点1: figsize
  原: figsize=(12, 6)
  新: figsize=(10, 6)

修改点2: 子图创建
  原: axes = fig.subplots(1, 2)
  新: ax = fig.add_subplot(111)

修改点3: 子图引用
  原: axes[0].plot(...)
      axes[0].set_title(...)
      axes[0].set_xlabel(...)
      axes[0].set_ylabel(...)
      axes[0].legend()
      axes[0].grid(...)
  新: ax.plot(...)
      ax.set_title(...)
      ax.set_xlabel(...)
      ax.set_ylabel(...)
      ax.legend()
      ax.grid(...)

修改点4: 删除柱状图
  原: # ACF差异图
      axes[1].bar(...)
      axes[1].set_title(...)
      axes[1].set_xlabel(...)
      axes[1].set_ylabel(...)
      axes[1].grid(...)
  新: （删除这5行）

可选美化:
  - linewidth=2 (加粗线条)
  - fontsize=14, fontweight='bold' (标题加大加粗)
  - fontsize=12 (坐标轴标签)
  - fontsize=11 (图例)
"""


# ============================================
# 快速替换脚本（仅供参考）
# ============================================

def quick_replace_guide():
    """
    使用文本编辑器的批量替换功能（Ctrl+H）:
    
    替换1: 
      查找: axes = fig.subplots(1, 2)
      替换: ax = fig.add_subplot(111)
    
    替换2: 
      查找: axes[0].
      替换: ax.
      
    替换3:
      查找: figsize=(12, 6)
      替换: figsize=(10, 6)
    
    然后手动删除:
      - axes[1].bar(...) 及后续4行
    """
    pass


if __name__ == "__main__":
    print("这个文件提供了ACF图表修改的完整示例")
    print("请参考BEFORE和AFTER函数中的代码")
    print("\n关键修改：")
    print("1. 删除axes[1]的柱状图（5行代码）")
    print("2. 将axes改为ax")
    print("3. 将subplots(1,2)改为add_subplot(111)")
    print("4. 调整figsize宽度")
