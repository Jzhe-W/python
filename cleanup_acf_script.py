# cleanup_acf_script.py
# 自动清理注释掉的ACF代码

import re
import os

def clean_acf_comments(input_file, output_file=None, dry_run=True):
    """
    清理文件中注释掉的ACF相关代码
    
    Parameters:
    -----------
    input_file : str
        输入文件路径
    output_file : str, optional
        输出文件路径，默认为None（覆盖原文件）
    dry_run : bool
        是否只是预览，不实际修改文件
    """
    
    if output_file is None:
        output_file = input_file
    
    # 读取文件
    print(f"📖 读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_line_count = len(lines)
    print(f"   原文件行数: {original_line_count}")
    
    # 定义需要删除的模式
    patterns_to_remove = [
        # 单行注释模式
        r'^\s*#\s*autocorr_loss\s*=\s*autocorrelation_loss.*完全移除',
        r'^\s*#\s*autocorr_weight\s*=.*完全移除',
        r'^\s*#\s*enh_autocorr_w\s*=.*完全移除',
        r'^\s*#\s*spring_acf_weight\s*=.*完全移除',
        r'^\s*#\s*summer_acf_weight\s*=.*完全移除',
        r'^\s*#\s*autumn_acf_weight\s*=.*完全移除',
        r'^\s*#\s*winter_acf_weight\s*=.*完全移除',
        r'^\s*#\s*acf_reg_loss\s*=.*完全移除ACF正则化',
        r'^\s*#\s*multi_acf_loss\s*=.*完全移除',
        r'^\s*#\s*precise_acf_loss\s*=.*删除.*风电不需要',
        r'^\s*#\s*winter_acf\s*=.*autocorrelation_loss',
        r'^\s*#\s*spring_acf\s*=.*autocorrelation_loss',
        
        # 说明性注释
        r'^\s*#\s*完全移除ACF权重计算\s*$',
        r'^\s*#\s*完全移除多lag ACF损失函数和相关计算\s*$',
        r'^\s*#\s*修复:\s*完全移除.*ACF损失.*$',
    ]
    
    # 筛选要保留的行
    cleaned_lines = []
    deleted_lines = []
    
    for i, line in enumerate(lines, 1):
        should_delete = False
        
        # 检查是否匹配删除模式
        for pattern in patterns_to_remove:
            if re.search(pattern, line):
                should_delete = True
                deleted_lines.append((i, line.rstrip()))
                break
        
        if not should_delete:
            cleaned_lines.append(line)
    
    deleted_count = len(deleted_lines)
    
    # 打印删除的行
    if deleted_count > 0:
        print(f"\n🗑️  将删除 {deleted_count} 行注释：")
        print("=" * 80)
        for line_num, line_content in deleted_lines[:10]:  # 只显示前10行
            print(f"  行{line_num}: {line_content}")
        if deleted_count > 10:
            print(f"  ... 还有 {deleted_count - 10} 行")
        print("=" * 80)
    else:
        print("ℹ️  没有找到需要删除的注释")
    
    # 写入文件或显示预览
    if dry_run:
        print(f"\n🔍 预览模式（未实际修改文件）")
        print(f"   如需实际删除，设置 dry_run=False")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"\n✅ 文件已更新: {output_file}")
        print(f"   原行数: {original_line_count}")
        print(f"   新行数: {len(cleaned_lines)}")
        print(f"   删除数: {deleted_count}")
    
    return deleted_count


def clean_acf_data_preparation(input_file, output_file=None, dry_run=True):
    """
    删除无用的ACF数据准备代码块（程序末尾）
    
    这是一个独立的代码块，需要特殊处理
    """
    
    if output_file is None:
        output_file = input_file
    
    print(f"\n📖 清理无用的ACF数据准备代码...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义删除范围的标记
    start_marker = "# 准备数据用于ACF相关性误差图"
    end_marker = "print(\"✅ 程序执行完成！\")"
    
    # 查找标记位置
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos == -1 or end_pos == -1:
        print("⚠️  未找到ACF数据准备代码块的标记")
        return 0
    
    # 提取要删除的部分
    deleted_content = content[start_pos:end_pos]
    deleted_lines_count = deleted_content.count('\n')
    
    print(f"🗑️  找到ACF数据准备代码块：")
    print(f"   起始位置: {start_pos}")
    print(f"   结束位置: {end_pos}")
    print(f"   将删除约 {deleted_lines_count} 行")
    
    # 显示预览
    preview_lines = deleted_content.split('\n')[:5]
    print(f"\n   预览（前5行）：")
    for line in preview_lines:
        print(f"     {line}")
    print(f"   ...")
    
    if not dry_run:
        # 删除该代码块
        new_content = content[:start_pos] + content[end_pos:]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"\n✅ 已删除ACF数据准备代码块")
    else:
        print(f"\n🔍 预览模式（未实际修改）")
    
    return deleted_lines_count


def main():
    """主函数：清理ACF相关代码"""
    
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         ACF 代码自动清理工具                              ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # 配置
    input_file = input("请输入文件路径（或直接回车使用默认值）: ").strip()
    if not input_file:
        input_file = "your_program.py"
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    # 询问是否预览
    mode = input("选择模式 (1=预览, 2=实际删除): ").strip()
    dry_run = (mode != '2')
    
    if dry_run:
        print("\n🔍 预览模式：只显示将要删除的内容，不会修改文件")
    else:
        print("\n⚠️  实际删除模式：将会修改文件！")
        confirm = input("确认继续？(yes/no): ").strip().lower()
        if confirm != 'yes':
            print("❌ 已取消操作")
            return
    
    # 创建备份（如果是实际删除模式）
    if not dry_run:
        backup_file = input_file + '.backup'
        print(f"📦 创建备份: {backup_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 执行清理
    print("\n" + "=" * 80)
    print("开始清理...")
    print("=" * 80)
    
    # 第1步：清理注释行
    count1 = clean_acf_comments(input_file, dry_run=dry_run)
    
    # 第2步：清理无用代码块
    count2 = clean_acf_data_preparation(input_file, dry_run=dry_run)
    
    # 总结
    print("\n" + "=" * 80)
    print("清理总结")
    print("=" * 80)
    print(f"删除注释行: {count1} 行")
    print(f"删除代码块: {count2} 行")
    print(f"总计删除: {count1 + count2} 行")
    
    if dry_run:
        print("\n💡 提示：这是预览模式，文件未被修改")
        print("   如需实际删除，请重新运行并选择模式2")
    else:
        print(f"\n✅ 清理完成！")
        print(f"   备份文件: {backup_file}")
        print(f"   修改文件: {input_file}")
        print(f"\n📋 建议：运行程序测试是否正常")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
