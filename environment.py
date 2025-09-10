"""
环境建模模块

管理仿真环境中的所有实体，计算距离和信道增益。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..core.entities import Entity, UAV, User, Warden
from ..core.parameters import SystemParameters


class Environment:
    """环境管理类"""
    
    def __init__(self, parameters: SystemParameters):
        """
        初始化环境
        
        Args:
            parameters: 系统参数
        """
        self.parameters = parameters
        
        # 实体列表
        self.uavs: List[UAV] = []
        self.users: List[User] = []
        self.wardens: List[Warden] = []
        
        # 缓存距离和信道增益矩阵
        self._distance_matrix_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._channel_gain_cache: Optional[Dict[str, Dict[str, float]]] = None
    
    def add_uav(self, uav: UAV):
        """添加无人机"""
        self.uavs.append(uav)
        self._clear_cache()
    
    def add_user(self, user: User):
        """添加用户"""
        self.users.append(user)
        self._clear_cache()
    
    def add_warden(self, warden: Warden):
        """添加监控者"""
        self.wardens.append(warden)
        self._clear_cache()
    
    def add_entities(self, uavs: List[UAV], users: List[User], wardens: List[Warden]):
        """批量添加实体"""
        self.uavs.extend(uavs)
        self.users.extend(users)
        self.wardens.extend(wardens)
        self._clear_cache()
    
    def _clear_cache(self):
        """清除缓存"""
        self._distance_matrix_cache = None
        self._channel_gain_cache = None
    
    def get_all_entities(self) -> List[Entity]:
        """获取所有实体"""
        return self.uavs + self.users + self.wardens
    
    def calculate_distance(self, entity1: Entity, entity2: Entity) -> float:
        """
        计算两个实体间的三维距离
        
        Args:
            entity1, entity2: 实体对象
            
        Returns:
            距离(米)
        """
        return entity1.distance_to(entity2)
    
    def calculate_channel_gain(self, transmitter: Entity, receiver: Entity) -> float:
        """
        计算信道增益 h = sqrt(ρ_0 / R^2)
        
        Args:
            transmitter: 发射实体
            receiver: 接收实体
            
        Returns:
            信道增益
        """
        distance = self.calculate_distance(transmitter, receiver)
        
        # 避免距离为0的情况
        if distance < 1e-6:
            distance = 1e-6
        
        # h = sqrt(ρ_0 / R^2) = sqrt(ρ_0) / R
        gain = np.sqrt(self.parameters.channel_ref_power_linear) / distance
        return gain
    
    def get_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        获取距离矩阵
        
        Returns:
            {entity1_id: {entity2_id: distance}}
        """
        if self._distance_matrix_cache is not None:
            return self._distance_matrix_cache
        
        all_entities = self.get_all_entities()
        distance_matrix = {}
        
        for entity1 in all_entities:
            distance_matrix[entity1.entity_id] = {}
            for entity2 in all_entities:
                distance_matrix[entity1.entity_id][entity2.entity_id] = \
                    self.calculate_distance(entity1, entity2)
        
        self._distance_matrix_cache = distance_matrix
        return distance_matrix
    
    def get_channel_gain_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        获取信道增益矩阵
        
        Returns:
            {transmitter_id: {receiver_id: gain}}
        """
        if self._channel_gain_cache is not None:
            return self._channel_gain_cache
        
        all_entities = self.get_all_entities()
        gain_matrix = {}
        
        for transmitter in all_entities:
            gain_matrix[transmitter.entity_id] = {}
            for receiver in all_entities:
                gain_matrix[transmitter.entity_id][receiver.entity_id] = \
                    self.calculate_channel_gain(transmitter, receiver)
        
        self._channel_gain_cache = gain_matrix
        return gain_matrix
    
    def get_uav_to_user_gains(self) -> Dict[str, Dict[str, float]]:
        """
        获取UAV到用户的信道增益
        
        Returns:
            {uav_id: {user_id: gain}}
        """
        gains = {}
        for uav in self.uavs:
            gains[uav.entity_id] = {}
            for user in self.users:
                gains[uav.entity_id][user.entity_id] = \
                    self.calculate_channel_gain(uav, user)
        return gains
    
    def get_uav_to_warden_gains(self) -> Dict[str, Dict[str, float]]:
        """
        获取UAV到监控者的信道增益
        
        Returns:
            {uav_id: {warden_id: gain}}
        """
        gains = {}
        for uav in self.uavs:
            gains[uav.entity_id] = {}
            for warden in self.wardens:
                gains[uav.entity_id][warden.entity_id] = \
                    self.calculate_channel_gain(uav, warden)
        return gains
    
    def get_system_layout(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        获取系统布局信息
        
        Returns:
            {entity_type: [(x, y, z), ...]}
        """
        layout = {
            'uavs': [(uav.x, uav.y, uav.z) for uav in self.uavs],
            'users': [(user.x, user.y, user.z) for user in self.users],
            'wardens': [(warden.x, warden.y, warden.z) for warden in self.wardens]
        }
        return layout
    
    def validate_positions(self) -> Tuple[bool, List[str]]:
        """
        验证实体位置的合理性
        
        Returns:
            (是否合理, 问题描述列表)
        """
        issues = []
        
        # 检查区域边界
        area_x, area_y = self.parameters.area_size
        
        for entity in self.get_all_entities():
            if not (0 <= entity.x <= area_x):
                issues.append(f"{entity.entity_id}: x坐标{entity.x}超出区域[0, {area_x}]")
            
            if not (0 <= entity.y <= area_y):
                issues.append(f"{entity.entity_id}: y坐标{entity.y}超出区域[0, {area_y}]")
            
            if entity.z < 0:
                issues.append(f"{entity.entity_id}: z坐标{entity.z}不能为负")
        
        # 检查实体重叠（距离过近）
        min_distance = 1.0  # 最小距离1米
        all_entities = self.get_all_entities()
        
        for i, entity1 in enumerate(all_entities):
            for entity2 in all_entities[i+1:]:
                distance = self.calculate_distance(entity1, entity2)
                if distance < min_distance:
                    issues.append(f"{entity1.entity_id}和{entity2.entity_id}距离过近: {distance:.2f}m")
        
        return len(issues) == 0, issues
    
    def update_positions(self, position_updates: Dict[str, Tuple[float, float, float]]):
        """
        更新实体位置（为动态仿真预留接口）
        
        Args:
            position_updates: {entity_id: (x, y, z)}
        """
        all_entities = {entity.entity_id: entity for entity in self.get_all_entities()}
        
        for entity_id, new_pos in position_updates.items():
            if entity_id in all_entities:
                entity = all_entities[entity_id]
                entity.position.x = new_pos[0]
                entity.position.y = new_pos[1]
                entity.position.z = new_pos[2]
        
        # 清除缓存
        self._clear_cache()
    
    def get_statistics(self) -> Dict[str, any]:
        """获取环境统计信息"""
        distance_matrix = self.get_distance_matrix()
        
        # 收集所有距离
        all_distances = []
        for src_distances in distance_matrix.values():
            for dst_id, distance in src_distances.items():
                if distance > 0:  # 排除自身距离
                    all_distances.append(distance)
        
        all_distances = np.array(all_distances)
        
        stats = {
            'num_uavs': len(self.uavs),
            'num_users': len(self.users),
            'num_wardens': len(self.wardens),
            'total_entities': len(self.get_all_entities()),
            'area_size': self.parameters.area_size,
            'min_distance': np.min(all_distances) if len(all_distances) > 0 else 0,
            'max_distance': np.max(all_distances) if len(all_distances) > 0 else 0,
            'avg_distance': np.mean(all_distances) if len(all_distances) > 0 else 0,
            'uav_heights': [uav.z for uav in self.uavs],
            'user_heights': [user.z for user in self.users],
            'warden_heights': [warden.z for warden in self.wardens]
        }
        
        return stats
    
    def __repr__(self) -> str:
        return f"Environment({len(self.uavs)}UAVs, {len(self.users)}Users, {len(self.wardens)}Wardens)"