"""
简单示例脚本

演示如何使用仿真平台的基本功能。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.simulation import SimulationController


def run_simple_example():
    """运行简单示例"""
    print("=" * 60)
    print("静态多无人机隐蔽通感系统仿真平台 - 简单示例")
    print("=" * 60)
    
    # 1. 初始化仿真控制器
    print("1. 初始化仿真系统...")
    controller = SimulationController()
    
    # 使用更少的权重点来加快演示
    controller.parameters.weight_range = np.array([0.0, 0.3, 0.6, 1.0])
    
    # 2. 验证系统配置
    print("2. 验证系统配置...")
    if not controller.validate_system():
        print("系统配置验证失败！")
        return
    print("✓ 系统配置验证通过")
    
    # 3. 显示系统信息
    print("\n3. 系统信息:")
    print(f"   区域大小: {controller.parameters.area_size}")
    print(f"   无人机数量: {len(controller.uavs)}")
    print(f"   用户数量: {len(controller.users)}")
    print(f"   监控者数量: {len(controller.wardens)}")
    print(f"   权重点数: {len(controller.parameters.weight_range)}")
    
    # 4. 运行基准仿真
    print("\n4. 运行基准仿真...")
    
    def simple_progress(current, total, weight, objectives, feasible):
        status = "✓" if feasible else "✗"
        print(f"   进度: {current}/{total} | 权重: {weight:.1f} | {status}")
    
    result = controller.run_baseline_simulation(simple_progress)
    
    # 5. 显示结果
    print("\n5. 仿真结果:")
    stats = result.get_statistics()
    print(f"   总解数: {stats['num_solutions']}")
    print(f"   可行解数: {stats['num_feasible']}")
    print(f"   可行率: {stats['feasible_ratio']:.1%}")
    
    # 6. 生成可视化
    print("\n6. 生成可视化图表...")
    
    # 系统布局图
    fig1 = controller.system_viz.plot_system_layout()
    plt.show()
    
    # 帕累托前沿图
    fig2 = controller.results_viz.plot_pareto_front(result)
    plt.show()
    
    # 7. 生成简要报告
    print("\n7. 生成报告...")
    report_path = controller.generate_report(output_format='summary')
    print(f"   报告已生成: {report_path}")
    
    print("\n示例完成！")


def run_quick_sensitivity():
    """运行快速敏感性分析示例"""
    print("=" * 60)
    print("敏感性分析示例 - 隐蔽阈值变化")
    print("=" * 60)
    
    # 初始化
    controller = SimulationController()
    controller.parameters.weight_range = np.array([0.0, 0.5, 1.0])  # 只用3个权重点
    
    # 运行敏感性分析
    thresholds = [0.8, 0.9, 0.95]
    print(f"分析隐蔽阈值: {thresholds}")
    
    def sens_progress(param_idx, total_params, current, total, weight, obj, feasible):
        print(f"阈值 {param_idx+1}/{total_params} | 权重 {current}/{total}")
    
    results = controller.run_sensitivity_analysis(
        'covert_threshold', thresholds, sens_progress
    )
    
    # 比较结果
    print(f"\n敏感性分析完成，共 {len(results)} 个场景")
    
    # 生成比较图
    result_list = list(results.values())
    labels = [f'阈值{t}' for t in thresholds]
    
    controller.results_viz.plot_pareto_front(result_list, labels)
    plt.title('不同隐蔽阈值下的帕累托前沿比较')
    plt.show()
    
    print("敏感性分析示例完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='仿真平台示例')
    parser.add_argument('--sensitivity', action='store_true', help='运行敏感性分析示例')
    args = parser.parse_args()
    
    try:
        if args.sensitivity:
            run_quick_sensitivity()
        else:
            run_simple_example()
    
    except KeyboardInterrupt:
        print("\n用户中断示例")
    
    except Exception as e:
        print(f"\n示例运行失败: {e}")
        import traceback
        traceback.print_exc()