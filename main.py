"""
静态多无人机隐蔽通感系统仿真平台主程序

使用方法:
    python main.py                              # 运行基准仿真
    python main.py --config config.json        # 使用指定配置文件
    python main.py --sensitivity warden_position  # 运行敏感性分析
    python main.py --quick                      # 快速仿真（减少权重点）
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import time

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.simulation import SimulationController
from src.core.config import create_default_config_file


def setup_logging(level: str = 'INFO'):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log', encoding='utf-8')
        ]
    )


def progress_callback(current: int, total: int, weight: float, objectives: tuple, feasible: bool):
    """进度回调函数"""
    status = "✓" if feasible else "✗"
    print(f"\r进度: {current}/{total} | 权重: {weight:.1f} | 目标: ({objectives[0]:.3f}, {objectives[1]:.3f}) {status}", 
          end='', flush=True)
    if current == total:
        print()  # 换行


def sensitivity_progress_callback(param_idx: int, total_params: int, 
                                current: int, total: int, weight: float, 
                                objectives: tuple, feasible: bool):
    """敏感性分析进度回调"""
    status = "✓" if feasible else "✗"
    print(f"\r参数 {param_idx+1}/{total_params} | 权重 {current}/{total} | w={weight:.1f} | {status}", 
          end='', flush=True)


def run_baseline_simulation(controller: SimulationController, args):
    """运行基准仿真"""
    print("=" * 60)
    print("运行基准仿真")
    print("=" * 60)
    
    # 显示系统信息
    print(f"系统配置:")
    print(f"  区域大小: {controller.parameters.area_size}")
    print(f"  无人机: {len(controller.uavs)} 架")
    print(f"  用户: {len(controller.users)} 个")
    print(f"  监控者: {len(controller.wardens)} 个")
    print(f"  权重点数: {len(controller.parameters.weight_range)}")
    print(f"  隐蔽阈值: {controller.parameters.covert_threshold}")
    print()
    
    # 运行仿真
    result = controller.run_baseline_simulation(progress_callback)
    
    # 显示结果摘要
    print("\n" + "=" * 60)
    print("仿真结果摘要")
    print("=" * 60)
    print(controller.get_experiment_summary())
    
    # 生成报告
    if not args.no_report:
        print("\n生成报告中...")
        report_path = controller.generate_report(output_format='full' if args.full_report else 'summary')
        print(f"报告已生成: {report_path}")
    
    return result


def run_sensitivity_analysis(controller: SimulationController, args):
    """运行敏感性分析"""
    analysis_type = args.sensitivity
    
    print("=" * 60)
    print(f"运行敏感性分析: {analysis_type}")
    print("=" * 60)
    
    # 定义参数值
    if analysis_type == 'warden_position':
        # 监控者位置变化
        area_x, area_y = controller.parameters.area_size
        center_x, center_y = area_x / 2, area_y / 2
        positions = [
            (center_x, center_y),  # 中心
            (center_x - 300, center_y),  # 左
            (center_x + 300, center_y),  # 右
            (center_x, center_y - 300),  # 下
            (center_x, center_y + 300),  # 上
        ]
        parameter_values = positions
        labels = ['中心', '左移', '右移', '下移', '上移']
    
    elif analysis_type == 'covert_threshold':
        # 隐蔽阈值变化
        parameter_values = [0.7, 0.8, 0.9, 0.95, 0.99]
        labels = [f'阈值{v}' for v in parameter_values]
    
    elif analysis_type == 'power_limit':
        # 功率限制变化
        parameter_values = [15, 18, 20, 25, 30]
        labels = [f'{v}dBm' for v in parameter_values]
    
    else:
        raise ValueError(f"不支持的敏感性分析类型: {analysis_type}")
    
    print(f"参数值: {parameter_values}")
    print()
    
    # 运行敏感性分析
    results = controller.run_sensitivity_analysis(
        analysis_type, parameter_values, sensitivity_progress_callback
    )
    
    print(f"\n敏感性分析完成，成功运行 {len(results)} 个场景")
    
    # 生成报告
    if not args.no_report:
        print("\n生成比较报告中...")
        report_path = controller.generate_report(f'sensitivity_{analysis_type}', 'full')
        print(f"报告已生成: {report_path}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='静态多无人机隐蔽通感系统仿真平台')
    
    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    # 仿真模式
    parser.add_argument('--sensitivity', type=str, 
                       choices=['warden_position', 'covert_threshold', 'power_limit'],
                       help='运行敏感性分析')
    parser.add_argument('--quick', action='store_true', help='快速仿真（减少权重点）')
    
    # 输出选项
    parser.add_argument('--no-report', action='store_true', help='不生成报告')
    parser.add_argument('--full-report', action='store_true', help='生成完整报告')
    parser.add_argument('--output-dir', type=str, default='results', help='输出目录')
    
    # 特殊操作
    parser.add_argument('--create-config', action='store_true', help='创建默认配置文件')
    parser.add_argument('--validate-only', action='store_true', help='仅验证系统配置')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 创建默认配置文件
    if args.create_config:
        config_path = Path('config/default_config.json')
        create_default_config_file(config_path)
        print(f"默认配置文件已创建: {config_path}")
        return
    
    try:
        # 初始化仿真控制器
        print("初始化仿真系统...")
        config_path = Path(args.config) if args.config else None
        controller = SimulationController(config_path)
        
        # 快速模式：减少权重点
        if args.quick:
            controller.parameters.weight_range = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            print("启用快速模式，权重点数: 6")
        
        # 设置输出目录
        controller.report_generator.output_dir = Path(args.output_dir)
        
        # 仅验证配置
        if args.validate_only:
            if controller.validate_system():
                print("✓ 系统配置验证通过")
                return
            else:
                print("✗ 系统配置验证失败")
                sys.exit(1)
        
        # 运行仿真
        start_time = time.time()
        
        if args.sensitivity:
            results = run_sensitivity_analysis(controller, args)
        else:
            result = run_baseline_simulation(controller, args)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n总用时: {total_time:.2f} 秒")
        print("仿真完成！")
        
    except KeyboardInterrupt:
        print("\n\n用户中断仿真")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"仿真失败: {e}", exc_info=True)
        print(f"\n仿真失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()