"""
性能指标计算器

统一管理通信、感知、隐蔽三个性能指标的计算。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from ..core.entities import UAV, User, Warden
from ..core.parameters import SystemParameters, PerformanceMetrics
from ..core.power_allocation import PowerAllocation
from ..environment.environment import Environment
from .communication import CommunicationMetrics
from .sensing import SensingMetrics
from .covert import CovertMetrics


class MetricsCalculator:
    """性能指标计算器"""
    
    def __init__(self, environment: Environment, parameters: SystemParameters):
        """
        初始化性能指标计算器
        
        Args:
            environment: 环境对象
            parameters: 系统参数
        """
        self.environment = environment
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        
        # 创建各性能计算模块
        self.comm_metrics = CommunicationMetrics(environment, parameters)
        self.sensing_metrics = SensingMetrics(environment, parameters)
        self.covert_metrics = CovertMetrics(environment, parameters)
    
    def calculate_all_metrics(self, power_allocation: PowerAllocation) -> PerformanceMetrics:
        """
        计算所有性能指标
        
        Args:
            power_allocation: 功率分配
            
        Returns:
            性能指标对象
        """
        # 应用功率分配
        power_allocation.apply_to_uavs()
        
        # 计算通信性能
        comm_utility = self.comm_metrics.calculate_total_utility()
        
        # 计算感知性能
        sensing_performance = self.sensing_metrics.calculate_sensing_performance()
        
        # 计算隐蔽性能
        covert_probability = self.covert_metrics.calculate_covert_performance()
        
        # 检查约束满足情况
        constraint_satisfied = self._check_all_constraints()
        
        return PerformanceMetrics(
            communication_utility=comm_utility,
            sensing_performance=sensing_performance,
            covert_probability=covert_probability,
            constraint_satisfied=constraint_satisfied
        )
    
    def calculate_objective_values(self, power_allocation: PowerAllocation) -> Tuple[float, float]:
        """
        计算多目标优化的目标函数值
        
        Args:
            power_allocation: 功率分配
            
        Returns:
            (-F_comm, F_sens) 元组
        """
        # 应用功率分配
        power_allocation.apply_to_uavs()
        
        # 计算目标函数值
        f_comm = self.comm_metrics.calculate_total_utility()
        f_sens = self.sensing_metrics.calculate_sensing_performance()
        
        # 返回优化目标（最小化）
        return -f_comm, f_sens
    
    def _check_all_constraints(self) -> bool:
        """
        检查所有约束条件
        
        Returns:
            是否满足所有约束
        """
        # 检查隐蔽约束
        all_covert_satisfied, _ = self.covert_metrics.check_all_covert_constraints()
        
        # 检查功率约束（在PowerAllocation中已检查）
        # 这里可以添加其他约束检查
        
        return all_covert_satisfied
    
    def evaluate_constraint_violations(self, power_allocation: PowerAllocation) -> List[float]:
        """
        评估约束违反情况（用于优化器）
        
        Args:
            power_allocation: 功率分配
            
        Returns:
            约束违反值列表（≤0表示满足约束）
        """
        # 应用功率分配
        power_allocation.apply_to_uavs()
        
        violations = []
        
        # 1. 功率非负性约束（已在PowerAllocation中处理）
        
        # 2. 功率范围约束（已在PowerAllocation中处理）
        
        # 3. 隐蔽约束
        for warden in self.environment.wardens:
            dep = self.covert_metrics.calculate_detection_error_probability(warden)
            # 约束: ξ_l^* ≥ ξ_min  =>  ξ_min - ξ_l^* ≤ 0
            violation = self.parameters.covert_threshold - dep
            violations.append(violation)
        
        return violations
    
    def get_all_statistics(self) -> Dict[str, any]:
        """
        获取所有性能指标的统计信息
        
        Returns:
            完整的统计信息字典
        """
        stats = {
            'communication': self.comm_metrics.get_communication_statistics(),
            'sensing': self.sensing_metrics.get_sensing_statistics(),
            'covert': self.covert_metrics.get_covert_statistics(),
            'environment': self.environment.get_statistics()
        }
        
        return stats
    
    def validate_all_metrics(self, check_power_allocation: bool = True) -> Tuple[bool, Dict[str, List[str]]]:
        """
        验证所有性能指标的合理性
        
        Args:
            check_power_allocation: 是否检查功率分配
            
        Returns:
            (是否全部合理, {模块名: 问题列表})
        """
        all_valid = True
        issues_dict = {}
        
        # 验证通信性能
        comm_valid, comm_issues = self.comm_metrics.validate_communication_performance(check_power_allocation)
        issues_dict['communication'] = comm_issues
        if not comm_valid:
            all_valid = False
        
        # 验证感知性能
        sensing_valid, sensing_issues = self.sensing_metrics.validate_sensing_performance(check_power_allocation)
        issues_dict['sensing'] = sensing_issues
        if not sensing_valid:
            all_valid = False
        
        # 验证隐蔽性能
        covert_valid, covert_issues = self.covert_metrics.validate_covert_performance(check_power_allocation)
        issues_dict['covert'] = covert_issues
        if not covert_valid:
            all_valid = False
        
        # 验证环境
        env_valid, env_issues = self.environment.validate_positions()
        issues_dict['environment'] = env_issues
        if not env_valid:
            all_valid = False
        
        return all_valid, issues_dict
    
    def calculate_weighted_objective(self, power_allocation: PowerAllocation, weight: float) -> float:
        """
        计算加权目标函数
        
        objective = w * (-F_comm) + (1-w) * F_sens
        
        Args:
            power_allocation: 功率分配
            weight: 权重 w ∈ [0,1]
            
        Returns:
            加权目标函数值
        """
        neg_f_comm, f_sens = self.calculate_objective_values(power_allocation)
        
        weighted_obj = weight * neg_f_comm + (1 - weight) * f_sens
        
        return weighted_obj
    
    def get_performance_summary(self, power_allocation: PowerAllocation) -> Dict[str, float]:
        """
        获取性能摘要
        
        Args:
            power_allocation: 功率分配
            
        Returns:
            性能摘要字典
        """
        metrics = self.calculate_all_metrics(power_allocation)
        
        summary = {
            'F_comm': -metrics.communication_utility,  # 注意负号
            'F_sens': metrics.sensing_performance,
            'covert_prob': metrics.covert_probability,
            'feasible': metrics.constraint_satisfied,
            'network_sum_rate': self.comm_metrics.calculate_network_sum_rate(),
            'total_sensing_trace': sum([
                self.sensing_metrics.calculate_crb_matrix(user).trace()
                for user in self.environment.users
            ]),
            'min_detection_error_prob': metrics.covert_probability
        }
        
        return summary
    
    def compare_allocations(self, allocations: List[PowerAllocation], 
                          labels: Optional[List[str]] = None) -> Dict[str, any]:
        """
        比较多个功率分配方案
        
        Args:
            allocations: 功率分配列表
            labels: 标签列表
            
        Returns:
            比较结果字典
        """
        if labels is None:
            labels = [f"方案{i+1}" for i in range(len(allocations))]
        
        comparison = {
            'labels': labels,
            'summaries': [],
            'objectives': [],
            'feasible': []
        }
        
        for allocation in allocations:
            summary = self.get_performance_summary(allocation)
            obj_values = self.calculate_objective_values(allocation)
            
            comparison['summaries'].append(summary)
            comparison['objectives'].append(obj_values)
            comparison['feasible'].append(summary['feasible'])
        
        return comparison
    
    def __repr__(self) -> str:
        return f"MetricsCalculator(环境: {self.environment})"