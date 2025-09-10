"""
结果可视化模块

提供帕累托前沿、敏感性分析等结果的可视化功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging

from ..optimization.optimizer import OptimizationResult
from ..optimization.analysis import ParetoAnalyzer


class ResultsVisualizer:
    """结果可视化"""
    
    def __init__(self):
        """初始化结果可视化器"""
        self.logger = logging.getLogger(__name__)
        self.analyzer = ParetoAnalyzer()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_pareto_front(self, results: Union[OptimizationResult, List[OptimizationResult]], 
                         labels: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制帕累托前沿
        
        Args:
            results: 优化结果或结果列表
            labels: 标签列表
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 统一处理为列表格式
        if isinstance(results, OptimizationResult):
            results = [results]
        
        if labels is None:
            labels = [f'方案{i+1}' for i in range(len(results))]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 定义颜色和标记
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        for i, (result, label) in enumerate(zip(results, labels)):
            # 获取可行解
            feasible_result = result.get_feasible_solutions()
            pareto_front = feasible_result.get_pareto_front()
            
            if len(pareto_front) == 0:
                self.logger.warning(f"{label}: 没有可行解")
                continue
            
            # 绘制帕累托前沿
            color = colors[i]
            marker = markers[i % len(markers)]
            
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                      c=[color], marker=marker, s=80, alpha=0.7, 
                      label=f'{label} (可行解: {len(pareto_front)})')
            
            # 连接线
            if len(pareto_front) > 1:
                # 按第一个目标排序
                sorted_indices = np.argsort(pareto_front[:, 0])
                sorted_front = pareto_front[sorted_indices]
                ax.plot(sorted_front[:, 0], sorted_front[:, 1], 
                       color=color, alpha=0.5, linewidth=1.5)
            
            # 标记膝点
            knee_info = self.analyzer._find_knee_point_advanced(pareto_front)
            if knee_info:
                knee_point = knee_info['point']
                ax.scatter(knee_point[0], knee_point[1], 
                          c='red', marker='*', s=200, 
                          label=f'{label} 膝点' if i == 0 else "")
        
        # 设置坐标轴
        ax.set_xlabel('-F_comm (通信效用负值)')
        ax.set_ylabel('F_sens (感知性能)')
        ax.set_title('帕累托前沿')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加理想点标注
        if len(results) > 0 and len(results[0].get_feasible_solutions().get_pareto_front()) > 0:
            all_points = np.vstack([result.get_feasible_solutions().get_pareto_front() 
                                   for result in results if len(result.get_feasible_solutions().get_pareto_front()) > 0])
            ideal_point = np.min(all_points, axis=0)
            ax.scatter(ideal_point[0], ideal_point[1], 
                      c='gold', marker='*', s=300, 
                      label='理想点', edgecolors='black', linewidth=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"帕累托前沿图已保存到: {save_path}")
        
        return fig
    
    def plot_convergence_history(self, result: OptimizationResult,
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制收敛历史
        
        Args:
            result: 优化结果
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        weights = result.weights
        objective_values = [info.get('objective_value', np.inf) for info in result.convergence_info]
        iterations = [info.get('iterations', 0) for info in result.convergence_info]
        success_flags = [info.get('success', False) for info in result.convergence_info]
        
        # 绘制目标函数值vs权重
        colors = ['green' if success else 'red' for success in success_flags]
        ax1.scatter(weights, objective_values, c=colors, alpha=0.7)
        ax1.set_xlabel('权重 w')
        ax1.set_ylabel('目标函数值')
        ax1.set_title('目标函数值 vs 权重')
        ax1.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='收敛'),
                          Patch(facecolor='red', label='未收敛')]
        ax1.legend(handles=legend_elements)
        
        # 绘制迭代次数vs权重
        ax2.scatter(weights, iterations, c=colors, alpha=0.7)
        ax2.set_xlabel('权重 w')
        ax2.set_ylabel('迭代次数')
        ax2.set_title('迭代次数 vs 权重')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"收敛历史图已保存到: {save_path}")
        
        return fig
    
    def plot_sensitivity_analysis(self, baseline_result: OptimizationResult,
                                comparison_results: List[OptimizationResult],
                                comparison_labels: List[str],
                                baseline_label: str = "基准",
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制敏感性分析图
        
        Args:
            baseline_result: 基准结果
            comparison_results: 比较结果列表
            comparison_labels: 比较标签列表
            baseline_label: 基准标签
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        all_results = [baseline_result] + comparison_results
        all_labels = [baseline_label] + comparison_labels
        
        fig = self.plot_pareto_front(all_results, all_labels, figsize)
        
        # 修改标题
        fig.axes[0].set_title('敏感性分析 - 帕累托前沿比较')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"敏感性分析图已保存到: {save_path}")
        
        return fig
    
    def plot_objective_evolution(self, result: OptimizationResult,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制目标函数演化图
        
        Args:
            result: 优化结果
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        feasible_result = result.get_feasible_solutions()
        
        if len(feasible_result.pareto_points) == 0:
            self.logger.warning("没有可行解，无法绘制目标函数演化图")
            return None
        
        weights = np.array(feasible_result.weights)
        pareto_front = feasible_result.get_pareto_front()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 绘制-F_comm vs 权重
        ax1.plot(weights, pareto_front[:, 0], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('权重 w')
        ax1.set_ylabel('-F_comm')
        ax1.set_title('通信性能 vs 权重')
        ax1.grid(True, alpha=0.3)
        
        # 绘制F_sens vs 权重
        ax2.plot(weights, pareto_front[:, 1], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('权重 w')
        ax2.set_ylabel('F_sens')
        ax2.set_title('感知性能 vs 权重')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"目标函数演化图已保存到: {save_path}")
        
        return fig
    
    def plot_power_allocation(self, result: OptimizationResult, 
                            weight_index: int = None,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制功率分配图
        
        Args:
            result: 优化结果
            weight_index: 权重索引，如果为None则显示所有
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        if weight_index is not None:
            # 显示单个权重的功率分配
            allocation = result.power_allocations[weight_index]
            weight = result.weights[weight_index]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            uav_ids = list(allocation.uav_powers.keys())
            powers = list(allocation.uav_powers.values())
            ratios = list(allocation.comm_ratios.values())
            
            # 绘制总功率
            ax1.bar(uav_ids, powers, alpha=0.7, color='skyblue')
            ax1.set_ylabel('总功率 (dBm)')
            ax1.set_title(f'总功率分配 (w={weight:.1f})')
            ax1.grid(True, alpha=0.3)
            
            # 绘制功率分配比例
            comm_powers = [allocation.parameters.dbm_to_watts(p) * r for p, r in zip(powers, ratios)]
            sensing_powers = [allocation.parameters.dbm_to_watts(p) * (1-r) for p, r in zip(powers, ratios)]
            
            x = range(len(uav_ids))
            ax2.bar(x, comm_powers, label='通信功率', alpha=0.7, color='blue')
            ax2.bar(x, sensing_powers, bottom=comm_powers, label='感知功率', alpha=0.7, color='red')
            ax2.set_xticks(x)
            ax2.set_xticklabels(uav_ids)
            ax2.set_ylabel('功率 (瓦特)')
            ax2.set_title(f'通信/感知功率分配 (w={weight:.1f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        else:
            # 显示所有权重的功率分配演化
            weights = result.weights
            uav_ids = list(result.power_allocations[0].uav_powers.keys())
            
            fig, axes = plt.subplots(len(uav_ids), 2, figsize=figsize)
            if len(uav_ids) == 1:
                axes = axes.reshape(1, -1)
            
            for i, uav_id in enumerate(uav_ids):
                powers = [allocation.uav_powers[uav_id] for allocation in result.power_allocations]
                ratios = [allocation.comm_ratios[uav_id] for allocation in result.power_allocations]
                
                # 绘制总功率演化
                axes[i, 0].plot(weights, powers, 'o-', linewidth=2, markersize=6)
                axes[i, 0].set_ylabel('总功率 (dBm)')
                axes[i, 0].set_title(f'{uav_id} 总功率 vs 权重')
                axes[i, 0].grid(True, alpha=0.3)
                
                # 绘制功率比例演化
                axes[i, 1].plot(weights, ratios, 's-', linewidth=2, markersize=6, color='orange')
                axes[i, 1].set_ylabel('通信功率比例')
                axes[i, 1].set_title(f'{uav_id} 功率比例 vs 权重')
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_ylim(0, 1)
            
            # 设置最后一行的x轴标签
            for j in range(2):
                axes[-1, j].set_xlabel('权重 w')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"功率分配图已保存到: {save_path}")
        
        return fig
    
    def plot_constraint_analysis(self, result: OptimizationResult,
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制约束分析图
        
        Args:
            result: 优化结果
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        weights = result.weights
        feasible_flags = result.feasible_flags
        
        # 绘制可行性vs权重
        colors = ['green' if feasible else 'red' for feasible in feasible_flags]
        ax1.scatter(weights, [1 if f else 0 for f in feasible_flags], 
                   c=colors, alpha=0.7, s=80)
        ax1.set_xlabel('权重 w')
        ax1.set_ylabel('可行性')
        ax1.set_title('约束满足情况 vs 权重')
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['不可行', '可行'])
        ax1.grid(True, alpha=0.3)
        
        # 计算可行性统计
        feasible_ratio = sum(feasible_flags) / len(feasible_flags)
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax1.text(0.02, 0.9, f'可行解比例: {feasible_ratio:.1%}', 
                transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 绘制收敛性vs权重
        success_flags = [info.get('success', False) for info in result.convergence_info]
        colors2 = ['blue' if success else 'orange' for success in success_flags]
        ax2.scatter(weights, [1 if s else 0 for s in success_flags], 
                   c=colors2, alpha=0.7, s=80)
        ax2.set_xlabel('权重 w')
        ax2.set_ylabel('收敛性')
        ax2.set_title('优化收敛情况 vs 权重')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['未收敛', '收敛'])
        ax2.grid(True, alpha=0.3)
        
        # 计算收敛性统计
        convergence_ratio = sum(success_flags) / len(success_flags)
        ax2.text(0.02, 0.9, f'收敛成功率: {convergence_ratio:.1%}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"约束分析图已保存到: {save_path}")
        
        return fig