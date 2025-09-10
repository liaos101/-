"""
隐蔽性能计算模块

计算检测错误概率(DEP)和隐蔽约束验证。
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from ..core.entities import UAV, Warden
from ..core.parameters import SystemParameters
from ..environment.environment import Environment


class CovertMetrics:
    """隐蔽性能计算"""
    
    def __init__(self, environment: Environment, parameters: SystemParameters):
        """
        初始化隐蔽性能计算器
        
        Args:
            environment: 环境对象
            parameters: 系统参数
        """
        self.environment = environment
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    def calculate_mu_factor(self, warden: Warden) -> float:
        """
        计算μ因子
        
        μ_l = (H1接收功率 - H0接收功率) / σ^2
        其中H1表示有通信，H0表示无通信
        
        Args:
            warden: 监控者
            
        Returns:
            μ因子
        """
        # H1假设：有通信时监控者接收的总功率
        h1_power = 0.0
        for uav in self.environment.uavs:
            if not uav.is_power_allocated():
                continue
            
            comm_power = uav.communication_power_w
            if comm_power <= 0:
                continue
            
            # UAV到监控者的信道增益
            channel_gain = self.environment.calculate_channel_gain(uav, warden)
            
            # 累加监控者接收到的通信信号功率
            h1_power += comm_power * (channel_gain ** 2)
        
        # H0假设：无通信时监控者接收功率为0
        h0_power = 0.0
        
        # 计算μ因子
        power_diff = h1_power - h0_power  # = h1_power
        
        if power_diff <= 0:
            # 如果没有通信功率，返回0
            return 0.0
        
        mu_factor = power_diff / self.parameters.noise_power_w
        
        return mu_factor
    
    def calculate_detection_error_probability(self, warden: Warden) -> float:
        """
        计算检测错误概率 ξ_l^*
        
        ξ_l^* = 1 + e^{-(1+μ_l)/μ_l * ln(1+μ_l)} - e^{-1/μ_l * ln(1+μ_l)}
        
        Args:
            warden: 监控者
            
        Returns:
            检测错误概率
        """
        mu = self.calculate_mu_factor(warden)
        
        # 处理特殊情况
        if mu <= 0:
            # 没有通信信号，监控者无法检测，错误概率为1
            return 1.0
        
        if mu > 1e6:
            # μ过大时，检测错误概率趋近于0
            return 0.0
        
        try:
            # 计算 ln(1 + μ)
            ln_1_plus_mu = np.log(1 + mu)
            
            # 计算指数项
            exp1 = np.exp(-(1 + mu) / mu * ln_1_plus_mu)
            exp2 = np.exp(-1 / mu * ln_1_plus_mu)
            
            # 计算DEP
            dep = 1 + exp1 - exp2
            
            # 确保结果在[0, 1]范围内
            dep = np.clip(dep, 0.0, 1.0)
            
            return dep
            
        except (OverflowError, UnderflowError, FloatingPointError):
            self.logger.warning(f"监控者{warden.entity_id}的DEP计算发生数值异常，μ={mu}")
            # 数值异常时的处理
            if mu > 100:
                return 0.0  # 强信号，容易检测
            else:
                return 0.5  # 中等强度信号
    
    def check_covert_constraint(self, warden: Warden) -> bool:
        """
        检查隐蔽约束 ξ_l^* ≥ ξ_min
        
        Args:
            warden: 监控者
            
        Returns:
            是否满足隐蔽约束
        """
        dep = self.calculate_detection_error_probability(warden)
        return dep >= self.parameters.covert_threshold
    
    def check_all_covert_constraints(self) -> Tuple[bool, Dict[str, bool]]:
        """
        检查所有监控者的隐蔽约束
        
        Returns:
            (是否全部满足, {warden_id: 是否满足})
        """
        results = {}
        all_satisfied = True
        
        for warden in self.environment.wardens:
            satisfied = self.check_covert_constraint(warden)
            results[warden.entity_id] = satisfied
            if not satisfied:
                all_satisfied = False
        
        return all_satisfied, results
    
    def calculate_covert_performance(self) -> float:
        """
        计算隐蔽性能指标（所有监控者的最小DEP）
        
        Returns:
            最小检测错误概率
        """
        if not self.environment.wardens:
            return 1.0  # 没有监控者时认为完全隐蔽
        
        min_dep = float('inf')
        
        for warden in self.environment.wardens:
            dep = self.calculate_detection_error_probability(warden)
            min_dep = min(min_dep, dep)
        
        return min_dep
    
    def calculate_constraint_violation(self) -> float:
        """
        计算约束违反程度
        
        Returns:
            违反程度（正值表示违反，0表示满足）
        """
        all_satisfied, warden_results = self.check_all_covert_constraints()
        
        if all_satisfied:
            return 0.0
        
        # 计算最大违反程度
        max_violation = 0.0
        
        for warden in self.environment.wardens:
            dep = self.calculate_detection_error_probability(warden)
            violation = max(0.0, self.parameters.covert_threshold - dep)
            max_violation = max(max_violation, violation)
        
        return max_violation
    
    def get_covert_statistics(self) -> Dict[str, any]:
        """
        获取隐蔽性能统计信息
        
        Returns:
            统计信息字典
        """
        warden_stats = {}
        
        for warden in self.environment.wardens:
            mu = self.calculate_mu_factor(warden)
            dep = self.calculate_detection_error_probability(warden)
            satisfied = self.check_covert_constraint(warden)
            
            warden_stats[warden.entity_id] = {
                'mu_factor': mu,
                'detection_error_prob': dep,
                'constraint_satisfied': satisfied,
                'violation': max(0.0, self.parameters.covert_threshold - dep)
            }
        
        all_satisfied, _ = self.check_all_covert_constraints()
        
        stats = {
            'warden_statistics': warden_stats,
            'min_detection_error_prob': self.calculate_covert_performance(),
            'all_constraints_satisfied': all_satisfied,
            'max_constraint_violation': self.calculate_constraint_violation(),
            'covert_threshold': self.parameters.covert_threshold,
            'num_wardens': len(self.environment.wardens)
        }
        
        return stats
    
    def get_detection_probabilities(self) -> Dict[str, float]:
        """
        获取所有监控者的检测概率（1 - DEP）
        
        Returns:
            {warden_id: detection_probability}
        """
        detection_probs = {}
        
        for warden in self.environment.wardens:
            dep = self.calculate_detection_error_probability(warden)
            detection_prob = 1.0 - dep
            detection_probs[warden.entity_id] = detection_prob
        
        return detection_probs
    
    def validate_covert_performance(self, check_power_allocation: bool = True) -> Tuple[bool, List[str]]:
        """
        验证隐蔽性能的合理性
        
        Args:
            check_power_allocation: 是否检查功率分配
            
        Returns:
            (是否合理, 问题描述列表)
        """
        issues = []
        
        try:
            # 检查功率分配（可选）
            if check_power_allocation:
                for uav in self.environment.uavs:
                    if not uav.is_power_allocated():
                        issues.append(f"UAV {uav.entity_id} 未分配功率")
                    elif uav.communication_power_w < 0:
                        issues.append(f"UAV {uav.entity_id} 通信功率为负: {uav.communication_power_w}")
            
            # 检查监控者
            if not self.environment.wardens:
                issues.append("没有监控者，无法计算隐蔽性能")
            
            # 检查每个监控者的计算结果（仅在功率已分配时）
            if check_power_allocation and all(uav.is_power_allocated() for uav in self.environment.uavs):
                for warden in self.environment.wardens:
                    mu = self.calculate_mu_factor(warden)
                    dep = self.calculate_detection_error_probability(warden)
                    
                    if mu < 0:
                        issues.append(f"监控者 {warden.entity_id} 的μ因子为负: {mu}")
                    
                    if not 0 <= dep <= 1:
                        issues.append(f"监控者 {warden.entity_id} 的DEP超出[0,1]范围: {dep}")
                    
                    if np.isnan(mu) or np.isinf(mu):
                        issues.append(f"监控者 {warden.entity_id} 的μ因子异常: {mu}")
                    
                    if np.isnan(dep) or np.isinf(dep):
                        issues.append(f"监控者 {warden.entity_id} 的DEP异常: {dep}")
                
                # 检查隐蔽性能
                covert_perf = self.calculate_covert_performance()
                if np.isnan(covert_perf) or np.isinf(covert_perf):
                    issues.append(f"隐蔽性能异常: {covert_perf}")
            
        except Exception as e:
            if check_power_allocation:
                issues.append(f"隐蔽性能计算出错: {str(e)}")
        
        return len(issues) == 0, issues