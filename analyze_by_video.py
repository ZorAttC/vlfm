#!/usr/bin/env python3
# filepath: /home/zoratt/DataDisk/3D_ws/vlfm/analyze_videos.py

import os
import re
from pathlib import Path
from collections import defaultdict
import argparse

def parse_filename(filename):
    """解析文件名中的参数"""
    params = {}
    
    # 使用正则表达式提取各种参数
    patterns = {
        'episode': r'episode=([^-]+)',
        'ckpt': r'ckpt=([^-]+)', 
        'distance_to_goal': r'distance_to_goal=([^-]+)',
        'success': r'success=([^-]+)',
        'spl': r'spl=([^-]+)',
        'soft_spl': r'soft_spl=([^-]+)',
        'distance_to_goal_reward': r'distance_to_goal_reward=([^-]+)',
        'traveled_stairs': r'traveled_stairs=([^-]+)',
        'yaw': r'yaw=([^-]+)',
        'target_detected': r'target_detected=([^-]+)',
        'stop_called': r'stop_called=([^-]+)',
        'start_yaw': r'start_yaw=([^-]+)'
    }
    
    for param_name, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            try:
                # 尝试转换为float，如果失败则保持字符串
                value = float(match.group(1))
                params[param_name] = value
            except ValueError:
                params[param_name] = match.group(1)
    
    return params

def analyze_video_directory(video_dir):
    """分析video目录中的文件"""
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        print(f"错误: 目录 {video_dir} 不存在")
        return
    
    # 获取所有视频文件 (常见的视频格式)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    total_files = len(video_files)
    print(f"总文件数量: {total_files}")
    
    if total_files == 0:
        print("未找到视频文件")
        return
    
    # 统计success相关信息
    success_count = 0
    success_files = []
    failed_files = []
    parse_errors = []
    
    stats = defaultdict(list)
    
    for video_file in video_files:
        filename = video_file.name
        try:
            params = parse_filename(filename)
            
            # 收集所有参数的统计信息
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    stats[key].append(value)
            
            # 检查success状态
            if 'success' in params:
                if params['success'] == 1.0 or params['success'] == '1.00':
                    success_count += 1
                    success_files.append((filename, params))
                else:
                    failed_files.append((filename, params))
            else:
                parse_errors.append(filename)
                
        except Exception as e:
            parse_errors.append(f"{filename}: {str(e)}")
    
    # 打印统计结果
    print(f"\n=== 成功率统计 ===")
    print(f"成功文件数量: {success_count}")
    print(f"失败文件数量: {total_files - success_count - len(parse_errors)}")
    print(f"解析错误文件数量: {len(parse_errors)}")
    
    if total_files > 0:
        success_rate = (success_count / total_files) * 100
        print(f"成功率: {success_rate:.2f}% ({success_count}/{total_files})")
    
    # 打印详细统计信息
    print(f"\n=== 详细统计 ===")
    for param_name, values in stats.items():
        if values:
            avg_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            print(f"{param_name}: 平均={avg_val:.4f}, 最小={min_val:.4f}, 最大={max_val:.4f}")
    
    # 显示一些成功案例
    # if success_files:
    #     print(f"\n=== 成功案例示例 (前5个) ===")
    #     for i, (filename, params) in enumerate(success_files[:5]):
    #         print(f"{i+1}. {filename}")
    #         print(f"   参数: {params}")
    
    # 显示解析错误
    if parse_errors:
        print(f"\n=== 解析错误文件 ===")
        for error in parse_errors[:10]:  # 只显示前10个错误
            print(f"  {error}")
        if len(parse_errors) > 10:
            print(f"  ... 还有 {len(parse_errors) - 10} 个错误")
    
    return {
        'total_files': total_files,
        'success_count': success_count,
        'success_rate': success_count / total_files * 100 if total_files > 0 else 0,
        'stats': dict(stats),
        'success_files': success_files,
        'failed_files': failed_files,
        'parse_errors': parse_errors
    }

def main():
    parser = argparse.ArgumentParser(description='分析video目录中的视频文件统计信息')
    parser.add_argument('video_dir', nargs='?', default='./video_dir', 
                       help='视频目录路径 (默认: ./video_dir)')
    parser.add_argument('--export-csv', action='store_true',
                       help='导出统计结果到CSV文件')
    
    args = parser.parse_args()
    
    result = analyze_video_directory(args.video_dir)
    
    if args.export_csv and result:
        import csv
        csv_file = Path(args.video_dir) / 'video_analysis.csv'
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入汇总信息
            writer.writerow(['统计项', '数值'])
            writer.writerow(['总文件数', result['total_files']])
            writer.writerow(['成功文件数', result['success_count']])
            writer.writerow(['成功率(%)', f"{result['success_rate']:.2f}"])
            writer.writerow([])
            
            # 写入详细统计
            writer.writerow(['参数名', '平均值', '最小值', '最大值'])
            for param, values in result['stats'].items():
                if values:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    writer.writerow([param, f"{avg_val:.4f}", f"{min_val:.4f}", f"{max_val:.4f}"])
        
        print(f"\n统计结果已导出到: {csv_file}")

if __name__ == "__main__":
    main()