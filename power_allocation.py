"""
功率分配数据结构

用于管理无人机的功率分配和约束验证。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .entities import UAV
from .parameters import SystemParameters


@dataclass
class PowerAllocation:
    """功率分配数据结构"""
    
    def __init__(self, uavs: List[UAV], parameters: SystemParameters):
        """
        初始化功率分配
        
        Args:
            uavs: 无人机列表
            parameters: 系统参数
        """
        self.uavs = uavs
        self.parameters = parameters
        
        # 功率分配字典
        self.uav_powers: Dict[str, float] = {}      # 总功率(dBm)
        self.comm_ratios: Dict[str, float] = {}     # 通信功率比例
    
    def set_allocation(self, uav_id: str, total_power_dbm: float, comm_ratio: float):
        """
        设置单个无人机的功率分配
        
        Args:
            uav_id: 无人机ID
            total_power_dbm: 总功率(dBm)
            comm_ratio: 通信功率比例
        """
        self.uav_powers[uav_id] = total_power_dbm
        self.comm_ratios[uav_id] = comm_ratio
    
    def set_from_vector(self, allocation_vector: np.ndarray):
        """
        从优化变量向量设置功率分配
        
        Args:
            allocation_vector: [p1, β1, p2, β2, ...] 形式的向量
        """
        if len(allocation_vector) != 2 * len(self.uavs):
            raise ValueError(f"分配向量长度应为{2 * len(self.uavs)}，实际为{len(allocation_vector)}")
        
        for i, uav in enumerate(self.uavs):
            total_power = allocation_vector[2*i]
            comm_ratio = allocation_vector[2*i + 1]
            self.set_allocation(uav.entity_id, total_power, comm_ratio)
    
    def to_vector(self) -> np.ndarray:
        """转换为优化变量向量"""
        vector = []
        for uav in self.uavs:
            vector.append(self.uav_powers.get(uav.entity_id, 0.0))
            vector.append(self.comm_ratios.get(uav.entity_id, 0.5))
        return np.array(vector)
    
    def apply_to_uavs(self):
        """将功率分配应用到无人机对象"""
        for uav in self.uavs:
            if uav.entity_id in self.uav_powers and uav.entity_id in self.comm_ratios:
                uav.set_power_allocation(
                    self.uav_powers[uav.entity_id],
                    self.comm_ratios[uav.entity_id]
                )
    
    def get_communication_powers(self) -> Dict[str, float]:
        """获取通信功率字典(瓦特)"""
        comm_powers = {}
        for uav in self.uavs:
            if uav.entity_id in self.uav_powers and uav.entity_id in self.comm_ratios:
                total_power_w = self.parameters.dbm_to_watts(self.uav_powers[uav.entity_id])
                comm_power_w = total_power_w * self.comm_ratios[uav.entity_id]
                comm_powers[uav.entity_id] = comm_power_w
            else:
                comm_powers[uav.entity_id] = 0.0
        return comm_powers
    
    def get_sensing_powers(self) -> Dict[str, float]:
        """获取感知功率字典(瓦特)"""
        sensing_powers = {}
        for uav in self.uavs:
            if uav.entity_id in self.uav_powers and uav.entity_id in self.comm_ratios:
                total_power_w = self.parameters.dbm_to_watts(self.uav_powers[uav.entity_id])
                sensing_power_w = total_power_w * (1 - self.comm_ratios[uav.entity_id])
                sensing_powers[uav.entity_id] = sensing_power_w
            else:
                sensing_powers[uav.entity_id] = 0.0
        return sensing_powers
    
    def validate_constraints(self) -> Tuple[bool, List[str]]:
        """
        验证功率约束
        
        Returns:
            (是否满足约束, 违反约束的描述列表)
        """
        violations = []
        
        for uav in self.uavs:
            uav_id = uav.entity_id
            
            if uav_id not in self.uav_powers:
                violations.append(f"{uav_id}: 未设置总功率")
                continue
            
            if uav_id not in self.comm_ratios:
                violations.append(f"{uav_id}: 未设置通信功率比例")
                continue
            
            total_power = self.uav_powers[uav_id]
            comm_ratio = self.comm_ratios[uav_id]
            
            # 总功率约束
            if total_power < self.parameters.power_min_dbm:
                violations.append(f"{uav_id}: 总功率{total_power:.2f}dBm低于最小值{self.parameters.power_min_dbm}dBm")
            
            if total_power > self.parameters.power_max_dbm:
                violations.append(f"{uav_id}: 总功率{total_power:.2f}dBm超过最大值{self.parameters.power_max_dbm}dBm")
            
            # 功率比例约束
            if comm_ratio < 0:
                violations.append(f"{uav_id}: 通信功率比例{comm_ratio:.3f}不能为负")
            
            if comm_ratio > 1:
                violations.append(f"{uav_id}: 通信功率比例{comm_ratio:.3f}不能大于1")
        
        return len(violations) == 0, violations
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """获取优化变量的边界约束"""
        bounds = []
        for uav in self.uavs:
            # 总功率边界
            bounds.append((self.parameters.power_min_dbm, self.parameters.power_max_dbm))
            # 通信功率比例边界
            bounds.append((0.0, 1.0))
        return bounds
    
    def clone(self) -> 'PowerAllocation':
        """创建副本"""
        new_allocation = PowerAllocation(self.uavs, self.parameters)
        new_allocation.uav_powers = self.uav_powers.copy()
        new_allocation.comm_ratios = self.comm_ratios.copy()
        return new_allocation
    
    def __repr__(self) -> str:
        info = []
        for uav in self.uavs:
            uav_id = uav.entity_id
            power = self.uav_powers.get(uav_id, 0.0)
            ratio = self.comm_ratios.get(uav_id, 0.0)
            info.append(f"{uav_id}: {power:.1f}dBm, β={ratio:.2f}")
        return f"PowerAllocation({', '.join(info)})"