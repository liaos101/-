"""
报告生成器

生成实验报告和图表汇总。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

from ..optimization.optimizer import OptimizationResult
from ..optimization.analysis import ParetoAnalyzer, ResultExporter
from ..environment.environment import Environment
from .system_viz import SystemVisualizer
from .results_viz import ResultsVisualizer


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: Union[str, Path] = "results"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.analyzer = ParetoAnalyzer()
        self.exporter = ResultExporter()
        self.system_viz = None
        self.results_viz = ResultsVisualizer()
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def set_environment(self, environment: Environment):
        """设置环境对象"""
        self.system_viz = SystemVisualizer(environment)
    
    def generate_full_report(self, results: Union[OptimizationResult, List[OptimizationResult]],
                           labels: Optional[List[str]] = None,
                           experiment_name: str = "simulation") -> str:
        """
        生成完整的实验报告
        
        Args:
            results: 优化结果或结果列表
            labels: 标签列表
            experiment_name: 实验名称
            
        Returns:
            报告文件路径
        """
        # 统一处理为列表格式
        if isinstance(results, OptimizationResult):
            results = [results]
        
        if labels is None:
            labels = [f'方案{i+1}' for i in range(len(results))]
        
        # 创建实验目录
        exp_dir = self.output_dir / f"{experiment_name}_{self.timestamp}"
        exp_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"开始生成实验报告: {experiment_name}")
        
        # 1. 生成系统布局图
        if self.system_viz:
            system_layout_path = exp_dir / "01_system_layout.png"
            self.system_viz.plot_system_layout(save_path=str(system_layout_path))
            
            # 生成信道增益热力图
            channel_heatmap_path = exp_dir / "02_channel_heatmap.png"
            self.system_viz.plot_channel_heatmap(save_path=str(channel_heatmap_path))
            
            # 生成覆盖地图
            coverage_map_path = exp_dir / "03_coverage_map.png"
            self.system_viz.plot_coverage_map(save_path=str(coverage_map_path))
        
        # 2. 生成帕累托前沿图
        pareto_front_path = exp_dir / "04_pareto_front.png"
        self.results_viz.plot_pareto_front(results, labels, save_path=str(pareto_front_path))
        
        # 3. 为每个结果生成详细图表
        for i, (result, label) in enumerate(zip(results, labels)):
            result_dir = exp_dir / f"result_{i+1}_{label.replace(' ', '_')}"
            result_dir.mkdir(exist_ok=True)
            
            # 收敛历史
            conv_path = result_dir / "convergence_history.png"
            self.results_viz.plot_convergence_history(result, save_path=str(conv_path))
            
            # 目标函数演化
            obj_evolution_path = result_dir / "objective_evolution.png"
            self.results_viz.plot_objective_evolution(result, save_path=str(obj_evolution_path))
            
            # 功率分配
            power_alloc_path = result_dir / "power_allocation.png"
            self.results_viz.plot_power_allocation(result, save_path=str(power_alloc_path))
            
            # 约束分析
            constraint_path = result_dir / "constraint_analysis.png"
            self.results_viz.plot_constraint_analysis(result, save_path=str(constraint_path))
            
            # 导出数据
            data_path = result_dir / "result_data.csv"
            self.exporter.export_to_csv(result, str(data_path))
        
        # 4. 生成文本报告
        report_path = exp_dir / "experiment_report.txt"
        self._generate_text_report(results, labels, experiment_name, str(report_path))
        
        # 5. 生成汇总图表
        if len(results) > 1:
            comparison_path = exp_dir / "05_comparison_analysis.png"
            self._generate_comparison_plot(results, labels, str(comparison_path))
        
        self.logger.info(f"实验报告已生成: {exp_dir}")
        
        return str(exp_dir)
    
    def _generate_text_report(self, results: List[OptimizationResult], 
                            labels: List[str], experiment_name: str, 
                            report_path: str):
        """生成文本报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""
静态多无人机隐蔽通感系统仿真报告
================================

实验名称: {experiment_name}
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
方案数量: {len(results)}

""")
            
            # 逐个分析结果
            for i, (result, label) in enumerate(zip(results, labels)):
                f.write(f"\n方案 {i+1}: {label}\n")
                f.write("=" * 50 + "\n")
                
                # 基本统计
                stats = result.get_statistics()
                f.write(f"解的数量: {stats['num_solutions']}\n")
                f.write(f"可行解数量: {stats['num_feasible']}\n")
                f.write(f"可行率: {stats['feasible_ratio']:.2%}\n")
                f.write(f"收敛成功率: {stats['convergence_success_rate']:.2%}\n")
                
                # 目标函数范围
                obj_ranges = stats['objective_ranges']
                f.write(f"\n目标函数范围:\n")
                f.write(f"  -F_comm: [{obj_ranges['neg_f_comm']['min']:.4f}, {obj_ranges['neg_f_comm']['max']:.4f}]\n")
                f.write(f"  F_sens:  [{obj_ranges['f_sens']['min']:.4f}, {obj_ranges['f_sens']['max']:.4f}]\n")
                
                # 帕累托前沿分析
                analysis = self.analyzer.analyze_pareto_front(result)
                if 'error' not in analysis:
                    f.write(f"\n帕累托前沿分析:\n")
                    f.write(f"  前沿点数: {analysis['num_points']}\n")
                    f.write(f"  超体积: {analysis['spread_metrics']['hypervolume']:.4f}\n")
                    f.write(f"  间距度量: {analysis['spread_metrics']['spacing']:.4f}\n")
                    
                    if analysis['knee_point']:
                        knee = analysis['knee_point']
                        f.write(f"  膝点: ({knee['point'][0]:.4f}, {knee['point'][1]:.4f})\n")
                
                f.write("\n" + "=" * 50 + "\n")
            
            # 比较分析
            if len(results) > 1:
                f.write(f"\n\n比较分析\n")
                f.write("=" * 20 + "\n")
                
                comparison = self.analyzer.compare_pareto_fronts(results, labels)
                hypervolumes = comparison['hypervolume_comparison']
                
                f.write("超体积比较:\n")
                for i, (label, hv) in enumerate(zip(labels, hypervolumes)):
                    f.write(f"  {label}: {hv:.4f}\n")
                
                best_hv_idx = np.argmax(hypervolumes)
                f.write(f"\n最佳方案 (超体积): {labels[best_hv_idx]}\n")
        
        self.logger.info(f"文本报告已保存: {report_path}")
    
    def _generate_comparison_plot(self, results: List[OptimizationResult], 
                                labels: List[str], save_path: str):
        """生成比较分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 超体积比较
        comparison = self.analyzer.compare_pareto_fronts(results, labels)
        hypervolumes = comparison['hypervolume_comparison']
        
        ax1.bar(labels, hypervolumes, alpha=0.7, color='skyblue')
        ax1.set_ylabel('超体积')
        ax1.set_title('超体积比较')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. 可行解数量比较
        feasible_counts = [result.get_statistics()['num_feasible'] for result in results]
        ax2.bar(labels, feasible_counts, alpha=0.7, color='lightgreen')
        ax2.set_ylabel('可行解数量')
        ax2.set_title('可行解数量比较')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 收敛成功率比较
        convergence_rates = [result.get_statistics()['convergence_success_rate'] for result in results]
        ax3.bar(labels, convergence_rates, alpha=0.7, color='orange')
        ax3.set_ylabel('收敛成功率')
        ax3.set_title('收敛成功率比较')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 目标函数范围比较
        f_comm_ranges = []
        f_sens_ranges = []
        
        for result in results:
            obj_ranges = result.get_statistics()['objective_ranges']
            f_comm_ranges.append(obj_ranges['neg_f_comm']['max'] - obj_ranges['neg_f_comm']['min'])
            f_sens_ranges.append(obj_ranges['f_sens']['max'] - obj_ranges['f_sens']['min'])
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax4.bar(x - width/2, f_comm_ranges, width, label='-F_comm 范围', alpha=0.7)
        ax4.bar(x + width/2, f_sens_ranges, width, label='F_sens 范围', alpha=0.7)
        ax4.set_ylabel('目标函数范围')
        ax4.set_title('目标函数范围比较')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"比较分析图已保存: {save_path}")
    
    def generate_quick_summary(self, result: OptimizationResult, 
                             experiment_name: str = "quick_sim") -> str:
        """
        生成快速摘要报告
        
        Args:
            result: 优化结果
            experiment_name: 实验名称
            
        Returns:
            摘要报告路径
        """
        # 创建输出目录
        summary_dir = self.output_dir / f"{experiment_name}_summary_{self.timestamp}"
        summary_dir.mkdir(exist_ok=True)
        
        # 生成帕累托前沿图
        pareto_path = summary_dir / "pareto_front.png"
        self.results_viz.plot_pareto_front(result, save_path=str(pareto_path))
        
        # 生成文本摘要
        summary_path = summary_dir / "summary.txt"
        summary_text = self.exporter.export_summary(result)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"实验: {experiment_name}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(summary_text)
        
        self.logger.info(f"快速摘要已生成: {summary_dir}")
        
        return str(summary_dir)
    
    def save_all_figures(self, figures: Dict[str, plt.Figure], 
                        experiment_name: str = "figures"):
        """
        保存所有图表
        
        Args:
            figures: {文件名: 图形对象} 字典
            experiment_name: 实验名称
        """
        fig_dir = self.output_dir / f"{experiment_name}_figures_{self.timestamp}"
        fig_dir.mkdir(exist_ok=True)
        
        for filename, fig in figures.items():
            if not filename.endswith(('.png', '.pdf', '.jpg')):
                filename += '.png'
            
            fig_path = fig_dir / filename
            fig.savefig(str(fig_path), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        self.logger.info(f"所有图表已保存到: {fig_dir}")
        
        return str(fig_dir)