"""
多目标优化模块

实现基于加权和方法的多目标优化算法，生成帕累托前沿。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, OptimizeResult

from ..core.entities import UAV
from ..core.parameters import SystemParameters
from ..core.power_allocation import PowerAllocation
from ..metrics.calculator import MetricsCalculator


@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    
    def __init__(self):
        # 帕累托前沿点
        self.pareto_points: List[Tuple[float, float]] = []  # [(-F_comm, F_sens), ...]
        
        # 对应的功率分配
        self.power_allocations: List[PowerAllocation] = []
        
        # 权重值
        self.weights: List[float] = []
        
        # 优化收敛信息
        self.convergence_info: List[Dict] = []
        
        # 可行解标记
        self.feasible_flags: List[bool] = []
    
    def add_solution(self, weight: float, point: Tuple[float, float], 
                    allocation: PowerAllocation, convergence: Dict, feasible: bool):
        """添加一个解"""
        self.weights.append(weight)
        self.pareto_points.append(point)
        self.power_allocations.append(allocation)
        self.convergence_info.append(convergence)
        self.feasible_flags.append(feasible)
    
    def get_pareto_front(self) -> np.ndarray:
        """获取帕累托前沿点数组"""
        return np.array(self.pareto_points)
    
    def get_feasible_solutions(self) -> 'OptimizationResult':
        """获取仅包含可行解的结果"""
        feasible_result = OptimizationResult()
        
        for i, feasible in enumerate(self.feasible_flags):
            if feasible:
                feasible_result.add_solution(
                    self.weights[i],
                    self.pareto_points[i],
                    self.power_allocations[i],
                    self.convergence_info[i],
                    True
                )
        
        return feasible_result
    
    def get_statistics(self) -> Dict[str, any]:
        """获取优化结果统计"""
        pareto_array = self.get_pareto_front()
        
        if len(pareto_array) == 0:
            return {'num_solutions': 0, 'num_feasible': 0}
        
        stats = {
            'num_solutions': len(self.pareto_points),
            'num_feasible': sum(self.feasible_flags),
            'feasible_ratio': sum(self.feasible_flags) / len(self.feasible_flags),
            'objective_ranges': {
                'neg_f_comm': {
                    'min': np.min(pareto_array[:, 0]),
                    'max': np.max(pareto_array[:, 0]),
                    'mean': np.mean(pareto_array[:, 0])
                },
                'f_sens': {
                    'min': np.min(pareto_array[:, 1]),
                    'max': np.max(pareto_array[:, 1]),
                    'mean': np.mean(pareto_array[:, 1])
                }
            },
            'convergence_success_rate': sum(
                1 for info in self.convergence_info if info.get('success', False)
            ) / len(self.convergence_info)
        }
        
        return stats


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, metrics_calculator: MetricsCalculator, uavs: List[UAV], 
                 parameters: SystemParameters):
        """
        初始化多目标优化器
        
        Args:
            metrics_calculator: 性能指标计算器
            uavs: 无人机列表
            parameters: 系统参数
        """
        self.metrics_calculator = metrics_calculator
        self.uavs = uavs
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        
        # 创建功率分配模板
        self.power_allocation_template = PowerAllocation(uavs, parameters)
    
    def weighted_sum_method(self, weight: float, 
                          initial_guess: Optional[np.ndarray] = None) -> Tuple[PowerAllocation, Dict]:
        """
        使用加权和方法求解单目标优化问题
        
        min: w * (-F_comm) + (1-w) * F_sens
        
        Args:
            weight: 权重 w ∈ [0,1]
            initial_guess: 初始猜测值
            
        Returns:
            (最优功率分配, 收敛信息)
        """
        # 设置初始猜测
        if initial_guess is None:
            initial_guess = self._get_initial_guess()
        
        # 设置边界约束
        bounds = self.power_allocation_template.get_bounds()
        
        # 定义目标函数
        def objective_function(x: np.ndarray) -> float:
            try:
                allocation = self.power_allocation_template.clone()
                allocation.set_from_vector(x)
                return self.metrics_calculator.calculate_weighted_objective(allocation, weight)
            except Exception as e:
                self.logger.warning(f"目标函数计算失败: {e}")
                return 1e6  # 返回大的惩罚值
        
        # 定义约束函数
        def constraint_function(x: np.ndarray) -> np.ndarray:
            try:
                allocation = self.power_allocation_template.clone()
                allocation.set_from_vector(x)
                violations = self.metrics_calculator.evaluate_constraint_violations(allocation)
                return np.array(violations)
            except Exception as e:
                self.logger.warning(f"约束函数计算失败: {e}")
                return np.array([1e6])  # 返回约束违反
        
        # 设置约束
        constraints = []
        if len(self.metrics_calculator.environment.wardens) > 0:
            # 隐蔽约束
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: -constraint_function(x)  # ≤0 转换为 ≥0
            })
        
        try:
            # 执行优化
            result = minimize(
                fun=objective_function,
                x0=initial_guess,
                method=self.parameters.solver_method,
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': self.parameters.optimization_tolerance,
                    'maxiter': self.parameters.max_iterations,
                    'disp': False
                }
            )
            
            # 创建最优功率分配
            optimal_allocation = self.power_allocation_template.clone()
            optimal_allocation.set_from_vector(result.x)
            
            # 收敛信息
            convergence_info = {
                'success': result.success,
                'message': result.message,
                'iterations': result.get('nit', 0),
                'function_evaluations': result.get('nfev', 0),
                'objective_value': result.fun,
                'weight': weight
            }
            
            return optimal_allocation, convergence_info
            
        except Exception as e:
            self.logger.error(f"优化失败 (w={weight}): {e}")
            
            # 返回初始猜测作为备选方案
            fallback_allocation = self.power_allocation_template.clone()
            fallback_allocation.set_from_vector(initial_guess)
            
            convergence_info = {
                'success': False,
                'message': f"优化异常: {str(e)}",
                'iterations': 0,
                'function_evaluations': 0,
                'objective_value': float('inf'),
                'weight': weight
            }
            
            return fallback_allocation, convergence_info
    
    def generate_pareto_front(self, progress_callback: Optional[Callable] = None) -> OptimizationResult:
        """
        生成帕累托前沿
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            优化结果对象
        """
        result = OptimizationResult()
        weights = self.parameters.weight_range
        
        self.logger.info(f"开始生成帕累托前沿，权重范围: {weights}")
        
        for i, weight in enumerate(weights):
            try:
                # 执行单目标优化
                allocation, convergence_info = self.weighted_sum_method(weight)
                
                # 计算目标函数值
                neg_f_comm, f_sens = self.metrics_calculator.calculate_objective_values(allocation)
                
                # 检查可行性
                feasible = self.metrics_calculator._check_all_constraints()
                
                # 添加解到结果中
                result.add_solution(weight, (neg_f_comm, f_sens), allocation, convergence_info, feasible)
                
                # 日志记录
                status = "可行" if feasible else "不可行"
                self.logger.info(f"权重 {weight:.1f}: (-F_comm={neg_f_comm:.4f}, F_sens={f_sens:.4f}) [{status}]")
                
                # 进度回调
                if progress_callback:
                    progress_callback(i + 1, len(weights), weight, (neg_f_comm, f_sens), feasible)
                
            except Exception as e:
                self.logger.error(f"权重 {weight} 的优化失败: {e}")
                continue
        
        self.logger.info(f"帕累托前沿生成完成，共 {result.get_statistics()['num_solutions']} 个解")
        
        return result
    
    def _get_initial_guess(self) -> np.ndarray:
        """
        生成初始猜测值
        
        Returns:
            初始猜测向量
        """
        initial_guess = []
        
        for uav in self.uavs:
            # 总功率：使用中等功率
            total_power = (self.parameters.power_min_dbm + self.parameters.power_max_dbm) / 2
            initial_guess.append(total_power)
            
            # 通信功率比例：使用平衡分配
            comm_ratio = 0.5
            initial_guess.append(comm_ratio)
        
        return np.array(initial_guess)
    
    def optimize_single_objective(self, objective_type: str, 
                                maximize: bool = False) -> Tuple[PowerAllocation, Dict]:
        """
        单目标优化
        
        Args:
            objective_type: 'communication' 或 'sensing'
            maximize: 是否最大化（默认最小化）
            
        Returns:
            (最优功率分配, 收敛信息)
        """
        if objective_type == 'communication':
            weight = 1.0 if not maximize else 0.0
        elif objective_type == 'sensing':
            weight = 0.0 if not maximize else 1.0
        else:
            raise ValueError("objective_type 必须是 'communication' 或 'sensing'")
        
        return self.weighted_sum_method(weight)
    
    def find_knee_point(self, result: OptimizationResult) -> Optional[int]:
        """
        找到帕累托前沿的膝点
        
        Args:
            result: 优化结果
            
        Returns:
            膝点的索引，如果找不到返回None
        """
        feasible_result = result.get_feasible_solutions()
        pareto_front = feasible_result.get_pareto_front()
        
        if len(pareto_front) < 3:
            return None
        
        # 归一化目标函数值
        f1_norm = (pareto_front[:, 0] - np.min(pareto_front[:, 0])) / (np.max(pareto_front[:, 0]) - np.min(pareto_front[:, 0]))
        f2_norm = (pareto_front[:, 1] - np.min(pareto_front[:, 1])) / (np.max(pareto_front[:, 1]) - np.min(pareto_front[:, 1]))
        
        # 计算到理想点(0,0)的距离
        distances = np.sqrt(f1_norm**2 + f2_norm**2)
        
        # 找到距离最小的点作为膝点
        knee_index = np.argmin(distances)
        
        return knee_index
    
    def validate_optimization_setup(self) -> Tuple[bool, List[str]]:
        """
        验证优化设置
        
        Returns:
            (是否有效, 问题列表)
        """
        issues = []
        
        try:
            # 检查无人机数量
            if len(self.uavs) == 0:
                issues.append("没有无人机")
            
            # 检查用户数量
            if len(self.metrics_calculator.environment.users) == 0:
                issues.append("没有用户")
            
            # 检查参数合理性
            try:
                self.parameters.validate()
            except ValueError as e:
                issues.append(f"参数无效: {e}")
            
            # 测试目标函数计算
            test_allocation = self.power_allocation_template.clone()
            test_allocation.set_from_vector(self._get_initial_guess())
            
            try:
                self.metrics_calculator.calculate_objective_values(test_allocation)
            except Exception as e:
                issues.append(f"目标函数计算失败: {e}")
            
            # 测试约束函数计算
            try:
                self.metrics_calculator.evaluate_constraint_violations(test_allocation)
            except Exception as e:
                issues.append(f"约束函数计算失败: {e}")
            
        except Exception as e:
            issues.append(f"优化设置验证失败: {e}")
        
        return len(issues) == 0, issues
    
    def __repr__(self) -> str:
        return f"MultiObjectiveOptimizer({len(self.uavs)}UAVs, 权重范围: {self.parameters.weight_range})"