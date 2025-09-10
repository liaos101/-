"""
基础实体类定义

包含实体基类以及无人机、用户、监控者的具体实现。
"""

import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from .parameters import EntityPosition


class Entity(ABC):
    """实体基类"""
    
    def __init__(self, x: float, y: float, z: float, entity_id: str):
        """
        初始化实体
        
        Args:
            x: x坐标(m)
            y: y坐标(m) 
            z: z坐标(m)
            entity_id: 实体ID
        """
        self.position = EntityPosition(x, y, z, self.__class__.__name__, entity_id)
        self.entity_id = entity_id
    
    @property
    def x(self) -> float:
        return self.position.x
    
    @property
    def y(self) -> float:
        return self.position.y
    
    @property
    def z(self) -> float:
        return self.position.z
    
    def distance_to(self, other: 'Entity') -> float:
        """计算到另一个实体的距离"""
        return self.position.distance_to(other.position)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.entity_id,
            'type': self.__class__.__name__,
            'position': [self.x, self.y, self.z]
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.entity_id}, pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}))"


class UAV(Entity):
    """无人机类"""
    
    def __init__(self, x: float, y: float, z: float, uav_id: str, max_power_dbm: float = 20.0):
        """
        初始化无人机
        
        Args:
            x, y, z: 位置坐标(m)
            uav_id: 无人机ID
            max_power_dbm: 最大发射功率(dBm)
        """
        super().__init__(x, y, z, uav_id)
        self.max_power_dbm = max_power_dbm
        
        # 功率分配
        self.total_power_dbm: Optional[float] = None
        self.communication_ratio: Optional[float] = None  # β_k
    
    def set_power_allocation(self, total_power_dbm: float, communication_ratio: float):
        """
        设置功率分配
        
        Args:
            total_power_dbm: 总发射功率(dBm)
            communication_ratio: 通信功率比例β_k ∈ [0,1]
        """
        if not 0 <= total_power_dbm <= self.max_power_dbm:
            raise ValueError(f"总功率必须在[0, {self.max_power_dbm}]dBm范围内")
        
        if not 0 <= communication_ratio <= 1:
            raise ValueError("通信功率比例必须在[0,1]范围内")
        
        self.total_power_dbm = total_power_dbm
        self.communication_ratio = communication_ratio
    
    @property
    def total_power_w(self) -> float:
        """总发射功率(瓦特)"""
        if self.total_power_dbm is None:
            return 0.0
        return 10 ** (self.total_power_dbm / 10.0 - 3)
    
    @property
    def communication_power_w(self) -> float:
        """通信功率(瓦特)"""
        if self.communication_ratio is None:
            return 0.0
        return self.total_power_w * self.communication_ratio
    
    @property
    def sensing_power_w(self) -> float:
        """感知功率(瓦特)"""
        if self.communication_ratio is None:
            return 0.0
        return self.total_power_w * (1 - self.communication_ratio)
    
    def is_power_allocated(self) -> bool:
        """检查是否已分配功率"""
        return self.total_power_dbm is not None and self.communication_ratio is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            'max_power_dbm': self.max_power_dbm,
            'total_power_dbm': self.total_power_dbm,
            'communication_ratio': self.communication_ratio,
            'communication_power_w': self.communication_power_w,
            'sensing_power_w': self.sensing_power_w
        })
        return base_dict


class User(Entity):
    """用户类"""
    
    def __init__(self, x: float, y: float, z: float, user_id: str):
        """
        初始化用户
        
        Args:
            x, y, z: 位置坐标(m)
            user_id: 用户ID
        """
        super().__init__(x, y, z, user_id)
        
        # 接收到的数据速率 (每个UAV到此用户)
        self.data_rates: Dict[str, float] = {}
    
    def set_data_rate(self, uav_id: str, rate: float):
        """设置来自特定UAV的数据速率"""
        self.data_rates[uav_id] = rate
    
    def get_total_data_rate(self) -> float:
        """获取总数据速率"""
        return sum(self.data_rates.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            'data_rates': self.data_rates,
            'total_rate': self.get_total_data_rate()
        })
        return base_dict


class Warden(Entity):
    """监控者类"""
    
    def __init__(self, x: float, y: float, z: float, warden_id: str):
        """
        初始化监控者
        
        Args:
            x, y, z: 位置坐标(m)
            warden_id: 监控者ID
        """
        super().__init__(x, y, z, warden_id)
        
        # 检测相关参数
        self.detection_error_probability: Optional[float] = None
        self.mu_factor: Optional[float] = None
    
    def set_detection_metrics(self, dep: float, mu: float):
        """
        设置检测指标
        
        Args:
            dep: 检测错误概率
            mu: μ因子
        """
        self.detection_error_probability = dep
        self.mu_factor = mu
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            'detection_error_prob': self.detection_error_probability,
            'mu_factor': self.mu_factor
        })
        return base_dict