"""
感知性能计算模块

计算基于克拉美-罗界(CRB)的感知性能指标 F_sens。
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from ..core.entities import UAV, User
from ..core.parameters import SystemParameters
from ..environment.environment import Environment


class SensingMetrics:
    """感知性能计算"""
    
    def __init__(self, environment: Environment, parameters: SystemParameters):
        """
        初始化感知性能计算器
        
        Args:
            environment: 环境对象
            parameters: 系统参数
        """
        self.environment = environment
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    def calculate_crb_matrix(self, user: User) -> np.ndarray:
        """
        计算用户定位误差的协方差矩阵 C_m^{u,v}
        
        基于费舍尔信息矩阵的逆矩阵计算CRB
        
        Args:
            user: 目标用户
            
        Returns:
            2x2协方差矩阵 [[σ_u^2, σ_uv], [σ_uv, σ_v^2]]
        """
        # 计算费舍尔信息矩阵
        fisher_matrix = self._calculate_fisher_information_matrix(user)
        
        try:
            # CRB = 费舍尔信息矩阵的逆
            crb_matrix = np.linalg.inv(fisher_matrix)
            
            # 检查数值稳定性
            if np.any(np.isnan(crb_matrix)) or np.any(np.isinf(crb_matrix)):
                self.logger.warning(f"用户{user.entity_id}的CRB矩阵包含异常值")
                # 返回一个较大的对角矩阵作为惩罚
                return np.eye(2) * 1e6
            
            return crb_matrix
            
        except np.linalg.LinAlgError:
            self.logger.warning(f"用户{user.entity_id}的费舍尔信息矩阵奇异，使用默认CRB")
            # 费舍尔矩阵奇异，返回一个较大的对角矩阵
            return np.eye(2) * 1e6
    
    def _calculate_fisher_information_matrix(self, user: User) -> np.ndarray:
        """
        计算费舍尔信息矩阵
        
        J = Σ_k (2 * p_k^rad * h_{k,m}^2 / σ^2) * ∇h_{k,m} * ∇h_{k,m}^T
        
        Args:
            user: 目标用户
            
        Returns:
            2x2费舍尔信息矩阵
        """
        fisher_matrix = np.zeros((2, 2))  # 只考虑x,y方向的定位
        
        for uav in self.environment.uavs:
            if not uav.is_power_allocated():
                continue
            
            sensing_power = uav.sensing_power_w
            if sensing_power <= 0:
                continue
            
            # 计算信道增益
            channel_gain = self.environment.calculate_channel_gain(uav, user)
            
            # 计算信道增益的梯度 ∇h_{k,m}
            gradient = self._calculate_channel_gain_gradient(uav, user)
            
            # 计算费舍尔信息矩阵的贡献
            # J_contribution = (2 * p_k^rad * h_{k,m}^2 / σ^2) * ∇h * ∇h^T
            coefficient = (2 * sensing_power * (channel_gain ** 2)) / self.parameters.noise_power_w
            fisher_contribution = coefficient * np.outer(gradient, gradient)
            
            fisher_matrix += fisher_contribution
        
        # 确保矩阵的数值稳定性
        # 添加小的正则化项以避免奇异矩阵
        regularization = 1e-12 * np.eye(2)
        fisher_matrix += regularization
        
        return fisher_matrix
    
    def _calculate_channel_gain_gradient(self, uav: UAV, user: User) -> np.ndarray:
        """
        计算信道增益关于用户位置的梯度
        
        h = sqrt(ρ_0) / R，其中 R = sqrt((x_u - x_k)^2 + (y_u - y_k)^2 + (z_u - z_k)^2)
        
        ∇h = -sqrt(ρ_0) / R^3 * [x_u - x_k, y_u - y_k]
        
        Args:
            uav: 无人机
            user: 用户
            
        Returns:
            梯度向量 [∂h/∂x, ∂h/∂y]
        """
        # 位置差
        dx = user.x - uav.x
        dy = user.y - uav.y
        dz = user.z - uav.z
        
        # 三维距离
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 避免距离为0
        if distance < 1e-6:
            distance = 1e-6
        
        # 梯度计算
        # ∂h/∂x = -sqrt(ρ_0) * (x_u - x_k) / R^3
        # ∂h/∂y = -sqrt(ρ_0) * (y_u - y_k) / R^3
        coefficient = -np.sqrt(self.parameters.channel_ref_power_linear) / (distance ** 3)
        
        gradient = coefficient * np.array([dx, dy])
        
        return gradient
    
    def calculate_sensing_performance(self) -> float:
        """
        计算感知性能指标 F_sens = log(Σ tr(C_m^{u,v}))
        
        Returns:
            感知性能值（越小越好）
        """
        total_trace = 0.0
        
        for user in self.environment.users:
            crb_matrix = self.calculate_crb_matrix(user)
            trace = np.trace(crb_matrix)
            total_trace += trace
        
        # 避免对数运算的数值问题
        if total_trace <= 0:
            self.logger.warning("总迹为非正值，使用默认值")
            total_trace = 1e-12
        
        sensing_performance = np.log(total_trace)
        
        return sensing_performance
    
    def calculate_user_positioning_errors(self) -> Dict[str, Dict[str, float]]:
        """
        计算每个用户的定位误差
        
        Returns:
            {user_id: {'x_var': σ_x^2, 'y_var': σ_y^2, 'xy_cov': σ_xy, 'trace': tr(C)}}
        """
        user_errors = {}
        
        for user in self.environment.users:
            crb_matrix = self.calculate_crb_matrix(user)
            
            user_errors[user.entity_id] = {
                'x_var': crb_matrix[0, 0],
                'y_var': crb_matrix[1, 1],
                'xy_cov': crb_matrix[0, 1],
                'trace': np.trace(crb_matrix),
                'crb_matrix': crb_matrix.tolist()
            }
        
        return user_errors
    
    def calculate_position_error_ellipse(self, user: User, confidence: float = 0.95) -> Dict[str, float]:
        """
        计算定位误差椭圆参数
        
        Args:
            user: 用户
            confidence: 置信度
            
        Returns:
            椭圆参数字典
        """
        crb_matrix = self.calculate_crb_matrix(user)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(crb_matrix)
        
        # 确保特征值为正
        eigenvalues = np.abs(eigenvalues)
        
        # 置信度对应的卡方分布临界值
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence, df=2)
        
        # 椭圆半轴长度
        semi_major = np.sqrt(chi2_val * np.max(eigenvalues))
        semi_minor = np.sqrt(chi2_val * np.min(eigenvalues))
        
        # 椭圆角度
        major_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(major_eigenvector[1], major_eigenvector[0])
        
        return {
            'semi_major': semi_major,
            'semi_minor': semi_minor,
            'angle_rad': angle,
            'angle_deg': np.degrees(angle),
            'area': np.pi * semi_major * semi_minor
        }
    
    def get_sensing_statistics(self) -> Dict[str, any]:
        """
        获取感知性能统计信息
        
        Returns:
            统计信息字典
        """
        user_errors = self.calculate_user_positioning_errors()
        
        # 收集统计数据
        x_vars = [errors['x_var'] for errors in user_errors.values()]
        y_vars = [errors['y_var'] for errors in user_errors.values()]
        traces = [errors['trace'] for errors in user_errors.values()]
        
        stats = {
            'sensing_performance': self.calculate_sensing_performance(),
            'total_trace': sum(traces),
            'user_errors': user_errors,
            'num_users': len(self.environment.users),
            'avg_x_var': np.mean(x_vars),
            'avg_y_var': np.mean(y_vars),
            'avg_trace': np.mean(traces),
            'max_trace': np.max(traces),
            'min_trace': np.min(traces)
        }
        
        return stats
    
    def validate_sensing_performance(self, check_power_allocation: bool = True) -> Tuple[bool, List[str]]:
        """
        验证感知性能的合理性
        
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
                    elif uav.sensing_power_w < 0:
                        issues.append(f"UAV {uav.entity_id} 感知功率为负: {uav.sensing_power_w}")
            
            # 检查CRB矩阵（仅在功率已分配时）
            if check_power_allocation and all(uav.is_power_allocated() for uav in self.environment.uavs):
                for user in self.environment.users:
                    crb_matrix = self.calculate_crb_matrix(user)
                    
                    # 检查对称性
                    if not np.allclose(crb_matrix, crb_matrix.T):
                        issues.append(f"用户 {user.entity_id} 的CRB矩阵不对称")
                    
                    # 检查正定性
                    eigenvalues = np.linalg.eigvals(crb_matrix)
                    if np.any(eigenvalues <= 0):
                        issues.append(f"用户 {user.entity_id} 的CRB矩阵不是正定的")
                    
                    # 检查数值异常
                    if np.any(np.isnan(crb_matrix)) or np.any(np.isinf(crb_matrix)):
                        issues.append(f"用户 {user.entity_id} 的CRB矩阵包含异常值")
                
                # 检查感知性能
                sensing_perf = self.calculate_sensing_performance()
                if np.isnan(sensing_perf) or np.isinf(sensing_perf):
                    issues.append(f"感知性能异常: {sensing_perf}")
            
        except Exception as e:
            if check_power_allocation:
                issues.append(f"感知性能计算出错: {str(e)}")
        
        return len(issues) == 0, issues