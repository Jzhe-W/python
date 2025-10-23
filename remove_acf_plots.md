# 需要删除的ACF柱状图相关代码

## 位置1：ACF对比和柱状图（第2300-2400行左右）

### 删除范围：从这里开始
```python
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

        # 可视化对比
        # 创建Figure 8+i的框架
        frame_acf = ttk.Frame(notebook)
        notebook.add(frame_acf, text=f"Figure {8 + i}: {season_names[i]}ACF")

        fig = plt.figure(8 + i, figsize=(12, 6))
        axes = fig.subplots(1, 2)

        # ACF对比图
        axes[0].plot(range(lags + 1), real_acf, 'b-o', label='Real', alpha=0.7)
        axes[0].plot(range(lags + 1), fake_acf, 'r-s', label='Generated', alpha=0.7)
        axes[0].set_title(f'Figure {8 + i}: {season_names[i]} ACF Comparison')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ACF差异图
        axes[1].bar(range(lags + 1), real_acf - fake_acf, alpha=0.7, color='orange')
        axes[1].set_title(f'{season_names[i]} ACF Difference (Real - Generated)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('Difference')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 将Figure 8+i嵌入到框架中
        canvas_acf = FigureCanvasTkAgg(fig, frame_acf)
        canvas_acf.draw()
        canvas_acf.get_tk_widget().pack(fill='both', expand=True)

        # 确保图形正确显示
        plt.close(fig)  # 关闭matplotlib的默认显示

        # 强制刷新显示
        root.update()
```

### 删除到这里结束（在这行之前）
```python
        # 记录结果
        results_data.append({
```

---

## 位置2：results_data中的ACF指标（第2400行左右）

### 修改这部分：
```python
        # 记录结果
        results_data.append({
            'Season': season_names[i],
            'Gen_vs_Real_RMSE': rmse,
            'Gen_vs_Real_MAE': mae,
            'Gen_vs_Real_MAPE (%)': mape,
            'Gen_vs_Real_Correlation': correlation,
            'ACF Correlation': acf_corr,      # ⬅️ 删除这行
            'ACF RMSE': acf_rmse,             # ⬅️ 删除这行
            'PICP': picp,
            'PINAW': pinaw
        })
```

### 改为：
```python
        # 记录结果
        results_data.append({
            'Season': season_names[i],
            'Gen_vs_Real_RMSE': rmse,
            'Gen_vs_Real_MAE': mae,
            'Gen_vs_Real_MAPE (%)': mape,
            'Gen_vs_Real_Correlation': correlation,
            'PICP': picp,
            'PINAW': pinaw
        })
```

---

## 位置3：平均值计算部分（第2450行左右）

### 修改这部分：
```python
    # 计算平均值
    print("\n各指标平均值:")
    avg_metrics = {
        'Gen_vs_Real_RMSE': np.mean([float(row['Gen_vs_Real_RMSE']) for row in results_data]),
        'Gen_vs_Real_MAE': np.mean([float(row['Gen_vs_Real_MAE']) for row in results_data]),
        'Gen_vs_Real_MAPE (%)': np.mean([float(row['Gen_vs_Real_MAPE (%)']) for row in results_data]),
        'Gen_vs_Real_Correlation': np.mean([float(row['Gen_vs_Real_Correlation']) for row in results_data]),
        'ACF Correlation': np.mean([float(row['ACF Correlation']) for row in results_data]),     # ⬅️ 删除这行
        'ACF RMSE': np.mean([float(row['ACF RMSE']) for row in results_data]),                   # ⬅️ 删除这行
        'PICP': np.mean([float(row['PICP']) for row in results_data]),
        'PINAW': np.mean([float(row['PINAW']) for row in results_data])
    }
```

### 改为：
```python
    # 计算平均值
    print("\n各指标平均值:")
    avg_metrics = {
        'Gen_vs_Real_RMSE': np.mean([float(row['Gen_vs_Real_RMSE']) for row in results_data]),
        'Gen_vs_Real_MAE': np.mean([float(row['Gen_vs_Real_MAE']) for row in results_data]),
        'Gen_vs_Real_MAPE (%)': np.mean([float(row['Gen_vs_Real_MAPE (%)']) for row in results_data]),
        'Gen_vs_Real_Correlation': np.mean([float(row['Gen_vs_Real_Correlation']) for row in results_data]),
        'PICP': np.mean([float(row['PICP']) for row in results_data]),
        'PINAW': np.mean([float(row['PINAW']) for row in results_data])
    }
```

---

## 位置4：3D ACF相关性误差图（第2600-2800行左右）

### 删除整个函数和调用：

```python
def create_acf_correlation_error_plot(real_data, generated_data, season_names, notebook):
    """创建ACF相关性误差图"""
    # ... 整个函数体 ...
    # 约200行代码
```

### 以及删除函数调用：
```python
# 创建ACF相关性误差图（添加异常处理）
try:
    fig_acf = create_acf_correlation_error_plot(real_data_for_3d, generated_data_for_3d, season_names, notebook)
    print("✅ ACF相关性误差图创建成功")
except Exception as e:
    print(f"⚠️ ACF相关性误差图创建失败: {e}")
    fig_acf = None
```

---

## 快速删除脚本

创建一个Python脚本自动删除这些部分：
