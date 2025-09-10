"""
配置文件加载和管理

支持从JSON文件加载系统配置，以及参数验证。
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from .parameters import SystemParameters
from .entities import UAV, User, Warden


class ConfigurationLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if self.config_path is None or not self.config_path.exists():
            print("使用默认配置")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            print(f"成功加载配置文件: {self.config_path}")
            return self.config_data
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "system_parameters": {
                "area_size": [2000, 2000],
                "uav_height": 100.0,
                "user_height": 0.0,
                "warden_height": 105.0,
                "channel_ref_power": -60.0,
                "bandwidth": 51200000,
                "noise_power_dbm": -110.0,
                "power_min_dbm": 0.0,
                "power_max_dbm": 20.0,
                "covert_threshold": 0.9
            },
            "entities": {
                "uavs": [
                    {"id": "UAV1", "position": [500, 500, 100], "max_power": 20},
                    {"id": "UAV2", "position": [1500, 1500, 100], "max_power": 20}
                ],
                "users": [
                    {"id": "User1", "position": [400, 600, 0]},
                    {"id": "User2", "position": [700, 800, 0]},
                    {"id": "User3", "position": [1200, 400, 0]},
                    {"id": "User4", "position": [1600, 1200, 0]}
                ],
                "wardens": [
                    {"id": "Warden1", "position": [1000, 1000, 105]}
                ]
            },
            "optimization": {
                "weight_range": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "solver": "SLSQP",
                "tolerance": 1e-6,
                "max_iterations": 1000
            }
        }
    
    def create_system_parameters(self, config_override: Optional[Dict[str, Any]] = None) -> SystemParameters:
        """
        创建系统参数对象
        
        Args:
            config_override: 覆盖配置
            
        Returns:
            SystemParameters对象
        """
        config = self.load_config()
        
        # 应用覆盖配置
        if config_override:
            self._deep_update(config, config_override)
        
        sys_params = config.get("system_parameters", {})
        opt_params = config.get("optimization", {})
        
        return SystemParameters(
            area_size=tuple(sys_params.get("area_size", [2000, 2000])),
            uav_height=sys_params.get("uav_height", 100.0),
            user_height=sys_params.get("user_height", 0.0),
            warden_height=sys_params.get("warden_height", 105.0),
            channel_ref_power=sys_params.get("channel_ref_power", -60.0),
            bandwidth=sys_params.get("bandwidth", 51.2e6),
            noise_power_dbm=sys_params.get("noise_power_dbm", -110.0),
            power_min_dbm=sys_params.get("power_min_dbm", 0.0),
            power_max_dbm=sys_params.get("power_max_dbm", 20.0),
            covert_threshold=sys_params.get("covert_threshold", 0.9),
            weight_range=np.array(opt_params.get("weight_range", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
            optimization_tolerance=opt_params.get("tolerance", 1e-6),
            max_iterations=opt_params.get("max_iterations", 1000),
            solver_method=opt_params.get("solver", "SLSQP")
        )
    
    def create_entities(self, config_override: Optional[Dict[str, Any]] = None) -> tuple[List[UAV], List[User], List[Warden]]:
        """
        创建实体对象
        
        Args:
            config_override: 覆盖配置
            
        Returns:
            (UAV列表, User列表, Warden列表)
        """
        config = self.load_config()
        
        # 应用覆盖配置
        if config_override:
            self._deep_update(config, config_override)
        
        entities_config = config.get("entities", {})
        
        # 创建UAV
        uavs = []
        for uav_config in entities_config.get("uavs", []):
            pos = uav_config["position"]
            uav = UAV(
                x=pos[0], y=pos[1], z=pos[2],
                uav_id=uav_config["id"],
                max_power_dbm=uav_config.get("max_power", 20.0)
            )
            uavs.append(uav)
        
        # 创建User
        users = []
        for user_config in entities_config.get("users", []):
            pos = user_config["position"]
            user = User(
                x=pos[0], y=pos[1], z=pos[2],
                user_id=user_config["id"]
            )
            users.append(user)
        
        # 创建Warden
        wardens = []
        for warden_config in entities_config.get("wardens", []):
            pos = warden_config["position"]
            warden = Warden(
                x=pos[0], y=pos[1], z=pos[2],
                warden_id=warden_config["id"]
            )
            wardens.append(warden)
        
        return uavs, users, wardens
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, config: Dict[str, Any], output_path: Path):
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存到: {output_path}")


def create_default_config_file(output_path: Path):
    """创建默认配置文件"""
    loader = ConfigurationLoader()
    default_config = loader._get_default_config()
    loader.save_config(default_config, output_path)