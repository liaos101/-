"""
优化结果分析工具

提供对优化结果的分析和处理功能。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .optimizer import OptimizationResult
from ..core.power_allocation import PowerAllocation


class ParetoAnalyzer:
    """帕累托前沿分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_pareto_front(self, result: OptimizationResult) -> Dict[str, any]:
        """
        分析帕累托前沿
        
        Args:
            result: 优化结果
            
        Returns:
            分析结果字典
        """
        feasible_result = result.get_feasible_solutions()
        pareto_front = feasible_result.get_pareto_front()
        
        if len(pareto_front) == 0:
            return {'error': '没有可行解'}
        
        analysis = {
            'num_points': len(pareto_front),
            'objective_statistics': self._calculate_objective_statistics(pareto_front),
            'spread_metrics': self._calculate_spread_metrics(pareto_front),
            'knee_point': self._find_knee_point_advanced(pareto_front),
            'extreme_points': self._find_extreme_points(pareto_front),
            'convexity': self._analyze_convexity(pareto_front)
        }
        
        return analysis
    
    def _calculate_objective_statistics(self, pareto_front: np.ndarray) -> Dict[str, Dict[str, float]]:
        """计算目标函数统计信息"""
        stats = {}
        
        for i, obj_name in enumerate(['neg_f_comm', 'f_sens']):
            values = pareto_front[:, i]
            stats[obj_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'range': float(np.max(values) - np.min(values))
            }
        
        return stats
    
    def _calculate_spread_metrics(self, pareto_front: np.ndarray) -> Dict[str, float]:
        """计算分布度量"""
        if len(pareto_front) < 2:
            return {'hypervolume': 0.0, 'spacing': 0.0}
        
        # 计算超体积（简化版本）
        # 使用原点作为参考点
        ref_point = np.max(pareto_front, axis=0) + 1.0
        hypervolume = self._calculate_hypervolume_2d(pareto_front, ref_point)
        
        # 计算间距度量
        spacing = self._calculate_spacing(pareto_front)
        
        return {
            'hypervolume': float(hypervolume),
            'spacing': float(spacing)
        }
    
    def _calculate_hypervolume_2d(self, points: np.ndarray, ref_point: np.ndarray) -> float:
        """计算二维超体积"""
        # 对点进行排序
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_points:
            width = prev_x - point[0]
            height = ref_point[1] - point[1]
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_x = point[0]
        
        return hypervolume
    
    def _calculate_spacing(self, pareto_front: np.ndarray) -> float:
        """计算间距度量"""
        if len(pareto_front) < 2:
            return 0.0
        
        distances = []
        for i in range(len(pareto_front)):
            min_dist = float('inf')
            for j in range(len(pareto_front)):
                if i != j:
                    dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        mean_dist = np.mean(distances)
        spacing = np.sqrt(np.mean([(d - mean_dist)**2 for d in distances]))
        
        return spacing
    
    def _find_knee_point_advanced(self, pareto_front: np.ndarray) -> Optional[Dict[str, any]]:
        """寻找膝点（高级版本）"""
        if len(pareto_front) < 3:
            return None
        
        # 归一化
        normalized_front = self._normalize_front(pareto_front)
        
        # 计算到理想点的距离
        ideal_point = np.min(normalized_front, axis=0)
        distances = np.linalg.norm(normalized_front - ideal_point, axis=1)
        
        knee_index = np.argmin(distances)
        
        return {
            'index': int(knee_index),
            'point': pareto_front[knee_index].tolist(),
            'distance_to_ideal': float(distances[knee_index])
        }
    
    def _find_extreme_points(self, pareto_front: np.ndarray) -> Dict[str, Dict[str, any]]:
        """寻找极值点"""
        extreme_points = {}
        
        # 每个目标的最优点
        for i, obj_name in enumerate(['neg_f_comm', 'f_sens']):
            best_index = np.argmin(pareto_front[:, i])
            extreme_points[f'best_{obj_name}'] = {
                'index': int(best_index),
                'point': pareto_front[best_index].tolist(),
                'value': float(pareto_front[best_index, i])
            }
        
        return extreme_points
    
    def _analyze_convexity(self, pareto_front: np.ndarray) -> Dict[str, any]:
        """分析前沿的凸性"""
        if len(pareto_front) < 3:
            return {'is_convex': None, 'convexity_measure': 0.0}
        
        # 按第一个目标排序
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        # 检查凸性
        is_convex = True
        convexity_violations = 0
        
        for i in range(1, len(sorted_front) - 1):
            # 检查中间点是否在线段下方
            p1, p2, p3 = sorted_front[i-1], sorted_front[i], sorted_front[i+1]
            
            # 线性插值
            t = (p2[0] - p1[0]) / (p3[0] - p1[0]) if p3[0] != p1[0] else 0.5
            interpolated_y = p1[1] + t * (p3[1] - p1[1])
            
            if p2[1] > interpolated_y:
                is_convex = False
                convexity_violations += 1
        
        convexity_measure = 1.0 - (convexity_violations / max(1, len(sorted_front) - 2))
        
        return {
            'is_convex': is_convex,
            'convexity_measure': float(convexity_measure),
            'violations': convexity_violations
        }
    
    def _normalize_front(self, pareto_front: np.ndarray) -> np.ndarray:
        """归一化帕累托前沿"""
        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        ranges = max_vals - min_vals
        
        # 避免除零
        ranges[ranges == 0] = 1.0
        
        normalized = (pareto_front - min_vals) / ranges
        return normalized
    
    def compare_pareto_fronts(self, results: List[OptimizationResult], 
                            labels: List[str]) -> Dict[str, any]:
        """
        比较多个帕累托前沿
        
        Args:
            results: 优化结果列表
            labels: 标签列表
            
        Returns:
            比较结果
        """
        comparison = {
            'labels': labels,
            'analyses': [],
            'dominance_matrix': [],
            'hypervolume_comparison': []
        }
        
        # 分析每个前沿
        for result in results:
            analysis = self.analyze_pareto_front(result)
            comparison['analyses'].append(analysis)
        
        # 计算支配关系
        dominance_matrix = self._calculate_dominance_matrix(results)
        comparison['dominance_matrix'] = dominance_matrix.tolist()
        
        # 比较超体积
        hypervolumes = [
            analysis.get('spread_metrics', {}).get('hypervolume', 0.0)
            for analysis in comparison['analyses']
        ]
        comparison['hypervolume_comparison'] = hypervolumes
        
        return comparison
    
    def _calculate_dominance_matrix(self, results: List[OptimizationResult]) -> np.ndarray:
        """计算支配关系矩阵"""
        n = len(results)
        dominance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    front_i = results[i].get_feasible_solutions().get_pareto_front()
                    front_j = results[j].get_feasible_solutions().get_pareto_front()
                    
                    if len(front_i) > 0 and len(front_j) > 0:
                        dominance_score = self._calculate_dominance_score(front_i, front_j)
                        dominance_matrix[i, j] = dominance_score
        
        return dominance_matrix
    
    def _calculate_dominance_score(self, front1: np.ndarray, front2: np.ndarray) -> float:
        """计算支配分数"""
        dominated_count = 0
        total_comparisons = 0
        
        for point1 in front1:
            for point2 in front2:
                total_comparisons += 1
                # 检查point1是否支配point2（所有目标都不差，至少一个更好）
                if np.all(point1 <= point2) and np.any(point1 < point2):
                    dominated_count += 1
        
        return dominated_count / max(1, total_comparisons)


class ResultExporter:
    """结果导出器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_to_dict(self, result: OptimizationResult) -> Dict[str, any]:
        """导出结果到字典"""
        export_data = {
            'pareto_front': result.get_pareto_front().tolist(),
            'weights': result.weights,
            'feasible_flags': result.feasible_flags,
            'convergence_info': result.convergence_info,
            'statistics': result.get_statistics()
        }
        
        # 导出功率分配
        power_allocations = []
        for allocation in result.power_allocations:
            allocation_dict = {
                'uav_powers': allocation.uav_powers,
                'comm_ratios': allocation.comm_ratios,
                'communication_powers': allocation.get_communication_powers(),
                'sensing_powers': allocation.get_sensing_powers()
            }
            power_allocations.append(allocation_dict)
        
        export_data['power_allocations'] = power_allocations
        
        return export_data
    
    def export_to_csv(self, result: OptimizationResult, filename: str):
        """导出结果到CSV文件"""
        import pandas as pd
        
        # 准备数据
        data = []
        for i, (weight, point, allocation, convergence, feasible) in enumerate(
            zip(result.weights, result.pareto_points, result.power_allocations, 
                result.convergence_info, result.feasible_flags)
        ):
            row = {
                'weight': weight,
                'neg_f_comm': point[0],
                'f_sens': point[1],
                'feasible': feasible,
                'success': convergence.get('success', False),
                'iterations': convergence.get('iterations', 0),
                'objective_value': convergence.get('objective_value', None)
            }
            
            # 添加功率分配信息
            for uav_id, power in allocation.uav_powers.items():
                row[f'power_{uav_id}'] = power
            
            for uav_id, ratio in allocation.comm_ratios.items():
                row[f'comm_ratio_{uav_id}'] = ratio
            
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        self.logger.info(f"结果已导出到: {filename}")
    
    def export_summary(self, result: OptimizationResult) -> str:
        """生成结果摘要"""
        stats = result.get_statistics()
        
        summary = f"""
优化结果摘要
============
总解数: {stats['num_solutions']}
可行解数: {stats['num_feasible']}
可行率: {stats['feasible_ratio']:.2%}
收敛成功率: {stats['convergence_success_rate']:.2%}

目标函数范围:
  -F_comm: [{stats['objective_ranges']['neg_f_comm']['min']:.4f}, {stats['objective_ranges']['neg_f_comm']['max']:.4f}]
  F_sens:  [{stats['objective_ranges']['f_sens']['min']:.4f}, {stats['objective_ranges']['f_sens']['max']:.4f}]
"""
        
        return summary