"""
系统参数配置类

定义了仿真系统中所有可配置的参数。
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class SystemParameters:
    """系统参数配置"""
    
    # 仿真区域参数
    area_size: Tuple[float, float] = (2000.0, 2000.0)  # 仿真区域大小(m)
    
    # 实体高度参数
    uav_height: float = 100.0        # 无人机高度(m)
    user_height: float = 0.0         # 用户高度(m)
    warden_height: float = 105.0     # 监控者高度(m)
    
    # 信道参数
    channel_ref_power: float = -60.0    # 信道参考功率(dB)
    bandwidth: float = 51.2e6           # 带宽(Hz)
    noise_power_dbm: float = -110.0     # 噪声功率(dBm)
    
    # 功率参数
    power_min_dbm: float = 0.0          # 最小功率(dBm)
    power_max_dbm: float = 20.0         # 最大功率(dBm)
    
    # 隐蔽性参数
    covert_threshold: float = 0.9       # 隐蔽阈值ξ_min
    
    # 优化参数
    weight_range: np.ndarray = None     # 权重范围
    optimization_tolerance: float = 1e-6
    max_iterations: int = 1000
    solver_method: str = "SLSQP"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.weight_range is None:
            self.weight_range = np.arange(0.0, 1.1, 0.1)
    
    @property
    def noise_power_w(self) -> float:
        """噪声功率转换为瓦特"""
        return 10 ** (self.noise_power_dbm / 10.0 - 3)
    
    @property
    def channel_ref_power_linear(self) -> float:
        """信道参考功率转换为线性值"""
        return 10 ** (self.channel_ref_power / 10.0)
    
    def dbm_to_watts(self, power_dbm: float) -> float:
        """将dBm转换为瓦特"""
        return 10 ** (power_dbm / 10.0 - 3)
    
    def watts_to_dbm(self, power_w: float) -> float:
        """将瓦特转换为dBm"""
        return 10 * np.log10(power_w * 1000)
    
    def validate(self) -> bool:
        """验证参数合理性"""
        if self.area_size[0] <= 0 or self.area_size[1] <= 0:
            raise ValueError("仿真区域大小必须为正数")
        
        if self.bandwidth <= 0:
            raise ValueError("带宽必须为正数")
        
        if self.power_min_dbm >= self.power_max_dbm:
            raise ValueError("最小功率必须小于最大功率")
        
        if not 0 <= self.covert_threshold <= 1:
            raise ValueError("隐蔽阈值必须在[0,1]范围内")
        
        return True


@dataclass
class EntityPosition:
    """实体位置"""
    x: float
    y: float
    z: float
    entity_type: str
    entity_id: str
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'EntityPosition') -> float:
        """计算到另一个位置的距离"""
        return np.linalg.norm(self.to_array() - other.to_array())


@dataclass
class PerformanceMetrics:
    """性能指标"""
    communication_utility: float
    sensing_performance: float
    covert_probability: float
    constraint_satisfied: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'F_comm': -self.communication_utility,  # 注意负号
            'F_sens': self.sensing_performance,
            'covert_prob': self.covert_probability,
            'feasible': self.constraint_satisfied
        }