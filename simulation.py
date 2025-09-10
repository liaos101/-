"""
仿真控制器

协调各模块，控制完整的仿真流程。
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import logging
import time

from ..core.config import ConfigurationLoader
from ..core.parameters import SystemParameters
from ..core.entities import UAV, User, Warden
from ..environment.environment import Environment
from ..metrics.calculator import MetricsCalculator
from ..optimization.optimizer import MultiObjectiveOptimizer, OptimizationResult
from ..optimization.analysis import ParetoAnalyzer, ResultExporter
from ..visualization.system_viz import SystemVisualizer
from ..visualization.results_viz import ResultsVisualizer
from ..visualization.report import ReportGenerator


class SimulationController:
    """仿真控制器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化仿真控制器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config_loader = ConfigurationLoader(config_path)
        self.parameters = self.config_loader.create_system_parameters()
        
        # 创建实体
        self.uavs, self.users, self.wardens = self.config_loader.create_entities()
        
        # 初始化环境
        self.environment = Environment(self.parameters)
        self.environment.add_entities(self.uavs, self.users, self.wardens)
        
        # 初始化性能计算器
        self.metrics_calculator = MetricsCalculator(self.environment, self.parameters)
        
        # 初始化优化器
        self.optimizer = MultiObjectiveOptimizer(
            self.metrics_calculator, self.uavs, self.parameters
        )
        
        # 初始化可视化组件
        self.system_viz = SystemVisualizer(self.environment)
        self.results_viz = ResultsVisualizer()
        self.report_generator = ReportGenerator()
        self.report_generator.set_environment(self.environment)
        
        # 分析器
        self.analyzer = ParetoAnalyzer()
        self.exporter = ResultExporter()
        
        # 仿真状态
        self.simulation_results: Dict[str, Any] = {}
        self.current_experiment = None
        
        self.logger.info("仿真控制器初始化完成")
        self._log_system_info()
    
    def _log_system_info(self):
        """记录系统信息"""
        self.logger.info(f"系统配置:")
        self.logger.info(f"  区域大小: {self.parameters.area_size}")
        self.logger.info(f"  无人机数量: {len(self.uavs)}")
        self.logger.info(f"  用户数量: {len(self.users)}")
        self.logger.info(f"  监控者数量: {len(self.wardens)}")
        self.logger.info(f"  权重范围: {self.parameters.weight_range}")
        self.logger.info(f"  隐蔽阈值: {self.parameters.covert_threshold}")
    
    def validate_system(self) -> bool:
        """
        验证系统配置
        
        Returns:
            是否有效
        """
        self.logger.info("验证系统配置...")
        
        issues = []
        
        # 验证参数
        try:
            self.parameters.validate()
        except ValueError as e:
            issues.append(f"参数验证失败: {e}")
        
        # 验证环境
        env_valid, env_issues = self.environment.validate_positions()
        if not env_valid:
            issues.extend(env_issues)
        
        # 验证性能计算
        metrics_valid, metrics_issues = self.metrics_calculator.validate_all_metrics(check_power_allocation=False)
        if not metrics_valid:
            for module, module_issues in metrics_issues.items():
                issues.extend([f"{module}: {issue}" for issue in module_issues])
        
        # 验证优化设置
        opt_valid, opt_issues = self.optimizer.validate_optimization_setup()
        if not opt_valid:
            issues.extend(opt_issues)
        
        if issues:
            self.logger.error("系统验证失败:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            return False
        else:
            self.logger.info("系统验证通过")
            return True
    
    def run_baseline_simulation(self, progress_callback: Optional[Callable] = None) -> OptimizationResult:
        """
        运行基准仿真
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            优化结果
        """
        self.logger.info("开始运行基准仿真...")
        
        if not self.validate_system():
            raise RuntimeError("系统验证失败，无法运行仿真")
        
        start_time = time.time()
        
        # 生成帕累托前沿
        result = self.optimizer.generate_pareto_front(progress_callback)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 记录结果
        self.simulation_results['baseline'] = {
            'result': result,
            'elapsed_time': elapsed_time,
            'timestamp': time.time()
        }
        
        self.current_experiment = 'baseline'
        
        self.logger.info(f"基准仿真完成，用时 {elapsed_time:.2f} 秒")
        
        # 记录统计信息
        stats = result.get_statistics()
        self.logger.info(f"结果统计:")
        self.logger.info(f"  解的数量: {stats['num_solutions']}")
        self.logger.info(f"  可行解数量: {stats['num_feasible']}")
        self.logger.info(f"  可行率: {stats['feasible_ratio']:.2%}")
        
        return result
    
    def run_sensitivity_analysis(self, analysis_type: str, 
                               parameter_values: List[Any],
                               progress_callback: Optional[Callable] = None) -> Dict[str, OptimizationResult]:
        """
        运行敏感性分析
        
        Args:
            analysis_type: 分析类型 ('warden_position', 'covert_threshold', 'power_limit')
            parameter_values: 参数值列表
            progress_callback: 进度回调函数
            
        Returns:
            {参数值: 优化结果} 字典
        """
        self.logger.info(f"开始运行敏感性分析: {analysis_type}")
        
        if not self.validate_system():
            raise RuntimeError("系统验证失败，无法运行敏感性分析")
        
        results = {}
        
        # 保存原始配置
        original_config = self._save_current_config()
        
        for i, param_value in enumerate(parameter_values):
            self.logger.info(f"运行参数值 {i+1}/{len(parameter_values)}: {param_value}")
            
            try:
                # 应用参数变化
                self._apply_parameter_change(analysis_type, param_value)
                
                # 重新验证系统
                if not self.validate_system():
                    self.logger.warning(f"参数值 {param_value} 导致系统无效，跳过")
                    continue
                
                # 运行优化
                result = self.optimizer.generate_pareto_front(
                    lambda current, total, weight, obj, feasible: (
                        progress_callback(i, len(parameter_values), current, total, weight, obj, feasible)
                        if progress_callback else None
                    )
                )
                
                results[str(param_value)] = result
                
            except Exception as e:
                self.logger.error(f"参数值 {param_value} 的仿真失败: {e}")
                continue
            
            finally:
                # 恢复原始配置
                self._restore_config(original_config)
        
        # 记录结果
        self.simulation_results[f'sensitivity_{analysis_type}'] = {
            'results': results,
            'parameter_values': parameter_values,
            'analysis_type': analysis_type,
            'timestamp': time.time()
        }
        
        self.logger.info(f"敏感性分析完成，成功运行 {len(results)}/{len(parameter_values)} 个参数值")
        
        return results
    
    def _save_current_config(self) -> Dict[str, Any]:
        """保存当前配置"""
        return {
            'parameters': self.parameters,
            'uav_positions': [(uav.x, uav.y, uav.z) for uav in self.uavs],
            'user_positions': [(user.x, user.y, user.z) for user in self.users],
            'warden_positions': [(warden.x, warden.y, warden.z) for warden in self.wardens]
        }
    
    def _restore_config(self, config: Dict[str, Any]):
        """恢复配置"""
        self.parameters = config['parameters']
        
        # 恢复位置
        for i, pos in enumerate(config['uav_positions']):
            if i < len(self.uavs):
                self.uavs[i].position.x, self.uavs[i].position.y, self.uavs[i].position.z = pos
        
        for i, pos in enumerate(config['user_positions']):
            if i < len(self.users):
                self.users[i].position.x, self.users[i].position.y, self.users[i].position.z = pos
        
        for i, pos in enumerate(config['warden_positions']):
            if i < len(self.wardens):
                self.wardens[i].position.x, self.wardens[i].position.y, self.wardens[i].position.z = pos
        
        # 清除环境缓存
        self.environment._clear_cache()
    
    def _apply_parameter_change(self, analysis_type: str, param_value: Any):
        """应用参数变化"""
        if analysis_type == 'warden_position':
            # 改变监控者位置
            if self.wardens:
                x, y = param_value
                self.wardens[0].position.x = x
                self.wardens[0].position.y = y
                self.environment._clear_cache()
        
        elif analysis_type == 'covert_threshold':
            # 改变隐蔽阈值
            self.parameters.covert_threshold = param_value
        
        elif analysis_type == 'power_limit':
            # 改变功率限制
            self.parameters.power_max_dbm = param_value
            for uav in self.uavs:
                uav.max_power_dbm = param_value
        
        else:
            raise ValueError(f"未知的分析类型: {analysis_type}")
    
    def generate_visualization(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        生成可视化图表
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            生成的图表信息
        """
        if experiment_name is None:
            experiment_name = self.current_experiment or 'default'
        
        self.logger.info(f"生成可视化图表: {experiment_name}")
        
        figures = {}
        
        # 1. 系统布局图
        figures['system_layout'] = self.system_viz.plot_system_layout()
        
        # 2. 信道增益热力图
        figures['channel_heatmap'] = self.system_viz.plot_channel_heatmap()
        
        # 3. 覆盖地图
        figures['coverage_map'] = self.system_viz.plot_coverage_map()
        
        # 4. 如果有仿真结果，生成结果图表
        if experiment_name in self.simulation_results:
            exp_data = self.simulation_results[experiment_name]
            
            if 'result' in exp_data:
                result = exp_data['result']
                
                # 帕累托前沿
                figures['pareto_front'] = self.results_viz.plot_pareto_front(result)
                
                # 收敛历史
                figures['convergence'] = self.results_viz.plot_convergence_history(result)
                
                # 功率分配
                figures['power_allocation'] = self.results_viz.plot_power_allocation(result)
                
                # 约束分析
                figures['constraint_analysis'] = self.results_viz.plot_constraint_analysis(result)
        
        return figures
    
    def generate_report(self, experiment_name: Optional[str] = None, 
                       output_format: str = 'full') -> str:
        """
        生成实验报告
        
        Args:
            experiment_name: 实验名称
            output_format: 输出格式 ('full', 'summary')
            
        Returns:
            报告路径
        """
        if experiment_name is None:
            experiment_name = self.current_experiment or 'default'
        
        self.logger.info(f"生成实验报告: {experiment_name}")
        
        if experiment_name not in self.simulation_results:
            raise ValueError(f"找不到实验结果: {experiment_name}")
        
        exp_data = self.simulation_results[experiment_name]
        
        if output_format == 'full':
            if 'result' in exp_data:
                # 单个结果的完整报告
                result = exp_data['result']
                report_path = self.report_generator.generate_full_report(result, experiment_name=experiment_name)
            elif 'results' in exp_data:
                # 敏感性分析的完整报告
                results = list(exp_data['results'].values())
                labels = list(exp_data['results'].keys())
                report_path = self.report_generator.generate_full_report(results, labels, experiment_name)
            else:
                raise ValueError("实验数据格式错误")
        
        elif output_format == 'summary':
            if 'result' in exp_data:
                result = exp_data['result']
                report_path = self.report_generator.generate_quick_summary(result, experiment_name)
            else:
                raise ValueError("摘要报告仅支持单个结果")
        
        else:
            raise ValueError(f"未知的输出格式: {output_format}")
        
        return report_path
    
    def export_results(self, experiment_name: Optional[str] = None, 
                      output_path: Optional[str] = None) -> str:
        """
        导出结果数据
        
        Args:
            experiment_name: 实验名称
            output_path: 输出路径
            
        Returns:
            导出文件路径
        """
        if experiment_name is None:
            experiment_name = self.current_experiment or 'default'
        
        if experiment_name not in self.simulation_results:
            raise ValueError(f"找不到实验结果: {experiment_name}")
        
        exp_data = self.simulation_results[experiment_name]
        
        if output_path is None:
            output_path = f"results/{experiment_name}_data.csv"
        
        if 'result' in exp_data:
            result = exp_data['result']
            self.exporter.export_to_csv(result, output_path)
        else:
            raise ValueError("暂不支持导出敏感性分析结果")
        
        self.logger.info(f"结果已导出到: {output_path}")
        
        return output_path
    
    def get_experiment_summary(self, experiment_name: Optional[str] = None) -> str:
        """
        获取实验摘要
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            摘要文本
        """
        if experiment_name is None:
            experiment_name = self.current_experiment or 'default'
        
        if experiment_name not in self.simulation_results:
            return f"找不到实验结果: {experiment_name}"
        
        exp_data = self.simulation_results[experiment_name]
        
        if 'result' in exp_data:
            result = exp_data['result']
            return self.exporter.export_summary(result)
        else:
            return f"实验 {experiment_name} 的结果摘要暂不支持"
    
    def list_experiments(self) -> List[str]:
        """列出所有实验"""
        return list(self.simulation_results.keys())
    
    def clear_results(self):
        """清除所有结果"""
        self.simulation_results.clear()
        self.current_experiment = None
        self.logger.info("所有结果已清除")
    
    def __repr__(self) -> str:
        return f"SimulationController(实验数: {len(self.simulation_results)}, 当前: {self.current_experiment})"