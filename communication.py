"""
通信性能计算模块

计算数据速率和通信效用函数 F_comm。
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from ..core.entities import UAV, User
from ..core.parameters import SystemParameters
from ..environment.environment import Environment


class CommunicationMetrics:
    """通信性能计算"""
    
    def __init__(self, environment: Environment, parameters: SystemParameters):
        """
        初始化通信性能计算器
        
        Args:
            environment: 环境对象
            parameters: 系统参数
        """
        self.environment = environment
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    def calculate_data_rate(self, uav: UAV, user: User, interference_power: float = 0.0) -> float:
        """
        计算UAV到用户的数据速率
        
        r_{k,m} = B * log2(1 + (p_k^com * h_{k,m}^2) / (σ^2 + I_{k,m}))
        
        Args:
            uav: 无人机
            user: 用户
            interference_power: 干扰功率(瓦特)
            
        Returns:
            数据速率(bps)
        """
        # 获取通信功率
        if not uav.is_power_allocated():
            return 0.0
        
        comm_power_w = uav.communication_power_w
        if comm_power_w <= 0:
            return 0.0
        
        # 计算信道增益
        channel_gain = self.environment.calculate_channel_gain(uav, user)
        
        # 计算接收信号功率
        received_power = comm_power_w * (channel_gain ** 2)
        
        # 计算总噪声功率（噪声 + 干扰）
        noise_power = self.parameters.noise_power_w
        total_noise = noise_power + interference_power
        
        # 计算信噪比
        if total_noise <= 0:
            snr = float('inf')
        else:
            snr = received_power / total_noise
        
        # 计算数据速率
        data_rate = self.parameters.bandwidth * np.log2(1 + snr)
        
        return max(0.0, data_rate)
    
    def calculate_data_rates_matrix(self, include_interference: bool = True) -> Dict[str, Dict[str, float]]:
        """
        计算所有UAV到用户的数据速率矩阵
        
        Args:
            include_interference: 是否考虑多用户干扰
            
        Returns:
            {uav_id: {user_id: rate}}
        """
        rates_matrix = {}
        
        for uav in self.environment.uavs:
            rates_matrix[uav.entity_id] = {}
            
            for user in self.environment.users:
                if include_interference:
                    # 计算来自其他UAV的干扰
                    interference = self._calculate_interference(uav, user)
                else:
                    interference = 0.0
                
                rate = self.calculate_data_rate(uav, user, interference)
                rates_matrix[uav.entity_id][user.entity_id] = rate
        
        return rates_matrix
    
    def _calculate_interference(self, target_uav: UAV, user: User) -> float:
        """
        计算来自其他UAV的干扰功率
        
        Args:
            target_uav: 目标UAV（不计算其干扰）
            user: 用户
            
        Returns:
            干扰功率(瓦特)
        """
        interference = 0.0
        
        for uav in self.environment.uavs:
            if uav.entity_id == target_uav.entity_id:
                continue  # 跳过目标UAV
            
            if not uav.is_power_allocated():
                continue
            
            # 干扰UAV的通信功率
            interferer_power = uav.communication_power_w
            if interferer_power <= 0:
                continue
            
            # 干扰UAV到用户的信道增益
            channel_gain = self.environment.calculate_channel_gain(uav, user)
            
            # 累加干扰功率
            interference += interferer_power * (channel_gain ** 2)
        
        return interference
    
    def calculate_user_total_rates(self) -> Dict[str, float]:
        """
        计算每个用户的总数据速率
        
        Returns:
            {user_id: total_rate}
        """
        rates_matrix = self.calculate_data_rates_matrix()
        user_rates = {}
        
        for user in self.environment.users:
            total_rate = 0.0
            for uav_id, user_rates_dict in rates_matrix.items():
                total_rate += user_rates_dict.get(user.entity_id, 0.0)
            user_rates[user.entity_id] = total_rate
        
        return user_rates
    
    def calculate_total_utility(self) -> float:
        """
        计算总网络效用 F_comm = Σ log(Σ r_{k,m})
        
        注意：这里返回的是正值，在优化时需要取负号
        
        Returns:
            通信效用值
        """
        user_total_rates = self.calculate_user_total_rates()
        
        utility = 0.0
        for user_id, total_rate in user_total_rates.items():
            if total_rate > 0:
                utility += np.log(total_rate)
            else:
                # 处理速率为0的情况，使用小的负值
                utility += np.log(1e-12)
                self.logger.warning(f"用户{user_id}的总数据速率为0")
        
        return utility
    
    def calculate_network_sum_rate(self) -> float:
        """
        计算网络总和速率
        
        Returns:
            总和速率(bps)
        """
        user_total_rates = self.calculate_user_total_rates()
        return sum(user_total_rates.values())
    
    def get_communication_statistics(self) -> Dict[str, any]:
        """
        获取通信性能统计信息
        
        Returns:
            统计信息字典
        """
        rates_matrix = self.calculate_data_rates_matrix()
        user_rates = self.calculate_user_total_rates()
        
        # 收集所有数据速率
        all_rates = []
        for uav_rates in rates_matrix.values():
            all_rates.extend(uav_rates.values())
        
        all_rates = np.array(all_rates)
        non_zero_rates = all_rates[all_rates > 0]
        
        stats = {
            'total_utility': self.calculate_total_utility(),
            'network_sum_rate': self.calculate_network_sum_rate(),
            'user_rates': user_rates,
            'rates_matrix': rates_matrix,
            'num_active_links': len(non_zero_rates),
            'min_rate': np.min(non_zero_rates) if len(non_zero_rates) > 0 else 0,
            'max_rate': np.max(non_zero_rates) if len(non_zero_rates) > 0 else 0,
            'avg_rate': np.mean(non_zero_rates) if len(non_zero_rates) > 0 else 0,
            'rate_std': np.std(non_zero_rates) if len(non_zero_rates) > 0 else 0
        }
        
        return stats
    
    def validate_communication_performance(self, check_power_allocation: bool = True) -> Tuple[bool, List[str]]:
        """
        验证通信性能的合理性
        
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
            
            # 检查数据速率（仅在功率已分配时）
            if check_power_allocation and all(uav.is_power_allocated() for uav in self.environment.uavs):
                rates_matrix = self.calculate_data_rates_matrix()
                for uav_id, user_rates in rates_matrix.items():
                    for user_id, rate in user_rates.items():
                        if rate < 0:
                            issues.append(f"UAV {uav_id} 到用户 {user_id} 的数据速率为负: {rate}")
                        elif np.isnan(rate) or np.isinf(rate):
                            issues.append(f"UAV {uav_id} 到用户 {user_id} 的数据速率异常: {rate}")
                
                # 检查总效用
                utility = self.calculate_total_utility()
                if np.isnan(utility) or np.isinf(utility):
                    issues.append(f"通信效用异常: {utility}")
            
        except Exception as e:
            if check_power_allocation:
                issues.append(f"通信性能计算出错: {str(e)}")
        
        return len(issues) == 0, issues