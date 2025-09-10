"""
系统可视化模块

提供系统布局、信道增益等的可视化功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
import logging

from ..environment.environment import Environment
from ..core.entities import UAV, User, Warden


class SystemVisualizer:
    """系统可视化"""
    
    def __init__(self, environment: Environment):
        """
        初始化系统可视化器
        
        Args:
            environment: 环境对象
        """
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_system_layout(self, figsize: Tuple[int, int] = (12, 10), 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制系统布局图
        
        Args:
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制无人机
        if self.environment.uavs:
            uav_positions = np.array([(uav.x, uav.y, uav.z) for uav in self.environment.uavs])
            ax.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                      c='red', marker='^', s=200, label='无人机', alpha=0.8)
            
            # 添加无人机标签
            for uav in self.environment.uavs:
                ax.text(uav.x, uav.y, uav.z + 5, uav.entity_id, 
                       fontsize=10, ha='center')
        
        # 绘制用户
        if self.environment.users:
            user_positions = np.array([(user.x, user.y, user.z) for user in self.environment.users])
            ax.scatter(user_positions[:, 0], user_positions[:, 1], user_positions[:, 2],
                      c='blue', marker='o', s=100, label='用户', alpha=0.8)
            
            # 添加用户标签
            for user in self.environment.users:
                ax.text(user.x, user.y, user.z + 5, user.entity_id, 
                       fontsize=10, ha='center')
        
        # 绘制监控者
        if self.environment.wardens:
            warden_positions = np.array([(warden.x, warden.y, warden.z) for warden in self.environment.wardens])
            ax.scatter(warden_positions[:, 0], warden_positions[:, 1], warden_positions[:, 2],
                      c='orange', marker='s', s=150, label='监控者', alpha=0.8)
            
            # 添加监控者标签
            for warden in self.environment.wardens:
                ax.text(warden.x, warden.y, warden.z + 5, warden.entity_id, 
                       fontsize=10, ha='center')
        
        # 绘制通信链路
        self._draw_communication_links(ax)
        
        # 设置坐标轴
        area_x, area_y = self.environment.parameters.area_size
        ax.set_xlim(0, area_x)
        ax.set_ylim(0, area_y)
        ax.set_zlim(0, max(120, max([entity.z for entity in self.environment.get_all_entities()]) + 20))
        
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_zlabel('Z (米)')
        ax.set_title('静态多无人机隐蔽通感系统布局图')
        
        # 添加图例
        ax.legend()
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"系统布局图已保存到: {save_path}")
        
        return fig
    
    def _draw_communication_links(self, ax):
        """绘制通信链路"""
        # UAV到用户的通信链路
        for uav in self.environment.uavs:
            for user in self.environment.users:
                ax.plot([uav.x, user.x], [uav.y, user.y], [uav.z, user.z],
                       'b-', alpha=0.3, linewidth=0.8)
        
        # UAV到监控者的潜在检测链路
        for uav in self.environment.uavs:
            for warden in self.environment.wardens:
                ax.plot([uav.x, warden.x], [uav.y, warden.y], [uav.z, warden.z],
                       'r--', alpha=0.5, linewidth=1.0)
    
    def plot_channel_heatmap(self, entity_type: str = 'user', 
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制信道增益热力图
        
        Args:
            entity_type: 'user' 或 'warden'
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        area_x, area_y = self.environment.parameters.area_size
        
        # 创建网格
        x = np.linspace(0, area_x, 100)
        y = np.linspace(0, area_y, 100)
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(1, len(self.environment.uavs), figsize=figsize)
        if len(self.environment.uavs) == 1:
            axes = [axes]
        
        for i, uav in enumerate(self.environment.uavs):
            # 计算每个网格点的信道增益
            Z = np.zeros_like(X)
            for j in range(len(x)):
                for k in range(len(y)):
                    # 创建虚拟接收者
                    if entity_type == 'user':
                        z_coord = self.environment.parameters.user_height
                    else:
                        z_coord = self.environment.parameters.warden_height
                    
                    distance = np.sqrt((X[k, j] - uav.x)**2 + (Y[k, j] - uav.y)**2 + (z_coord - uav.z)**2)
                    if distance < 1e-6:
                        distance = 1e-6
                    
                    gain = np.sqrt(self.environment.parameters.channel_ref_power_linear) / distance
                    Z[k, j] = 20 * np.log10(gain)  # 转换为dB
            
            # 绘制热力图
            im = axes[i].contourf(X, Y, Z, levels=50, cmap='viridis')
            axes[i].set_title(f'{uav.entity_id} 信道增益 (dB)')
            axes[i].set_xlabel('X (米)')
            axes[i].set_ylabel('Y (米)')
            
            # 添加无人机位置
            axes[i].plot(uav.x, uav.y, 'r^', markersize=15, label='无人机')
            
            # 添加实体位置
            if entity_type == 'user':
                for user in self.environment.users:
                    axes[i].plot(user.x, user.y, 'bo', markersize=8, label='用户' if user == self.environment.users[0] else "")
            else:
                for warden in self.environment.wardens:
                    axes[i].plot(warden.x, warden.y, 'rs', markersize=10, label='监控者' if warden == self.environment.wardens[0] else "")
            
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"信道增益热力图已保存到: {save_path}")
        
        return fig
    
    def plot_distance_matrix(self, figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制距离矩阵热力图
        
        Args:
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        distance_matrix = self.environment.get_distance_matrix()
        entities = self.environment.get_all_entities()
        entity_ids = [entity.entity_id for entity in entities]
        
        # 转换为numpy数组
        n = len(entity_ids)
        dist_array = np.zeros((n, n))
        
        for i, id1 in enumerate(entity_ids):
            for j, id2 in enumerate(entity_ids):
                dist_array[i, j] = distance_matrix[id1][id2]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        im = ax.imshow(dist_array, cmap='plasma', interpolation='nearest')
        
        # 设置刻度标签
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(entity_ids, rotation=45)
        ax.set_yticklabels(entity_ids)
        
        # 添加数值标注
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{dist_array[i, j]:.1f}',
                              ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title('实体间距离矩阵 (米)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"距离矩阵图已保存到: {save_path}")
        
        return fig
    
    def plot_coverage_map(self, power_threshold_dbm: float = -80,
                         figsize: Tuple[int, int] = (12, 10),
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制覆盖地图
        
        Args:
            power_threshold_dbm: 功率阈值(dBm)
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        area_x, area_y = self.environment.parameters.area_size
        
        # 创建网格
        x = np.linspace(0, area_x, 200)
        y = np.linspace(0, area_y, 200)
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算每个网格点的最强信号
        max_power = np.full_like(X, -np.inf)
        
        for uav in self.environment.uavs:
            if not uav.is_power_allocated():
                continue
            
            tx_power_w = uav.communication_power_w
            if tx_power_w <= 0:
                continue
            
            for i in range(len(x)):
                for j in range(len(y)):
                    distance = np.sqrt((X[j, i] - uav.x)**2 + (Y[j, i] - uav.y)**2 + 
                                     (self.environment.parameters.user_height - uav.z)**2)
                    if distance < 1e-6:
                        distance = 1e-6
                    
                    # 计算接收功率
                    gain = np.sqrt(self.environment.parameters.channel_ref_power_linear) / distance
                    rx_power_w = tx_power_w * (gain ** 2)
                    rx_power_dbm = 10 * np.log10(rx_power_w * 1000)
                    
                    max_power[j, i] = max(max_power[j, i], rx_power_dbm)
        
        # 创建覆盖掩码
        coverage = max_power >= power_threshold_dbm
        
        # 绘制覆盖图
        ax.contourf(X, Y, max_power, levels=50, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, coverage, levels=[0.5], colors='red', linewidths=2)
        
        # 添加实体位置
        for uav in self.environment.uavs:
            ax.plot(uav.x, uav.y, 'r^', markersize=15, label='无人机' if uav == self.environment.uavs[0] else "")
        
        for user in self.environment.users:
            ax.plot(user.x, user.y, 'bo', markersize=10, label='用户' if user == self.environment.users[0] else "")
        
        for warden in self.environment.wardens:
            ax.plot(warden.x, warden.y, 'ks', markersize=12, label='监控者' if warden == self.environment.wardens[0] else "")
        
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_title(f'通信覆盖地图 (阈值: {power_threshold_dbm} dBm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
        cbar.set_label('接收功率 (dBm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"覆盖地图已保存到: {save_path}")
        
        return fig