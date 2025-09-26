#!/usr/bin/env python3
"""
地图配置加载器
用于解析和加载协作举升任务的地图配置
"""

import yaml
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

class CollaborativeLiftingMapLoader:
    """协作举升地图加载器"""
    
    def __init__(self, map_config_path: str):
        """
        初始化地图加载器
        
        Args:
            map_config_path: 地图配置文件路径
        """
        self.map_config_path = Path(map_config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载地图配置文件"""
        if not self.map_config_path.exists():
            raise FileNotFoundError(f"地图配置文件不存在: {self.map_config_path}")
            
        with open(self.map_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get_table_positions(self) -> List[List[float]]:
        """获取桌面位置列表"""
        return self.config.get('TABLE', [])
    
    def get_board_start_position(self) -> List[float]:
        """获取木板起始位置"""
        return self.config.get('BOARD_START', [1.0, 0.0, 0.85])
    
    def get_robot_start_position(self) -> List[float]:
        """获取机器人起始位置"""
        return self.config.get('ROBOT_START', [0.5, 0.0, 0.8])
    
    def get_human_start_position(self) -> List[float]:
        """获取人类起始位置"""
        return self.config.get('HUMAN_START', [1.5, 0.0, 0.8])
    
    def get_collaboration_zone(self) -> Dict[str, Any]:
        """获取协作区域配置"""
        return self.config.get('COLLABORATION_ZONE', {
            'CENTER': [1.0, 0.0, 0.8],
            'RADIUS': 0.8
        })
    
    def get_lift_targets(self) -> List[List[float]]:
        """获取举升目标位置列表"""
        return self.config.get('LIFT_TARGETS', [])
    
    def get_safety_zones(self) -> Dict[str, List[float]]:
        """获取安全区域配置"""
        return self.config.get('SAFETY_ZONES', {})
    
    def get_boundaries(self) -> Dict[str, float]:
        """获取环境边界"""
        return self.config.get('BOUNDARIES', {})
    
    def get_task_params(self) -> Dict[str, Any]:
        """获取任务参数"""
        return self.config.get('TASK_PARAMS', {})
    
    def get_board_size(self) -> List[float]:
        """获取木板尺寸"""
        task_params = self.get_task_params()
        return task_params.get('BOARD_SIZE', [1.0, 0.4, 0.03])
    
    def get_table_size(self) -> List[float]:
        """获取桌面尺寸"""
        task_params = self.get_task_params()
        return task_params.get('TABLE_SIZE', [0.4, 1.5, 0.05])
    
    def get_min_balance(self) -> float:
        """获取最小平衡阈值"""
        task_params = self.get_task_params()
        return task_params.get('MIN_BALANCE', 0.8)
    
    def get_collaboration_distance(self) -> float:
        """获取协作距离"""
        task_params = self.get_task_params()
        return task_params.get('COLLABORATION_DISTANCE', 0.6)
    
    def validate_config(self) -> bool:
        """验证地图配置的有效性"""
        try:
            # 检查必要的配置项
            required_keys = ['TABLE', 'BOARD_START', 'ROBOT_START', 'HUMAN_START']
            for key in required_keys:
                if key not in self.config:
                    print(f"缺少必要的配置项: {key}")
                    return False
            
            # 检查桌面位置
            table_positions = self.get_table_positions()
            if not table_positions:
                print("桌面位置配置为空")
                return False
            
            # 检查举升目标
            lift_targets = self.get_lift_targets()
            if not lift_targets:
                print("举升目标配置为空")
                return False
            
            # 检查边界配置
            boundaries = self.get_boundaries()
            if not boundaries:
                print("边界配置为空")
                return False
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("=" * 60)
        print("协作举升地图配置摘要")
        print("=" * 60)
        
        print(f"地图文件: {self.map_config_path}")
        print(f"配置验证: {'通过' if self.validate_config() else '失败'}")
        print()
        
        print("桌面位置:")
        for i, pos in enumerate(self.get_table_positions()):
            print(f"  桌面 {i+1}: {pos}")
        print()
        
        print("起始位置:")
        print(f"  木板: {self.get_board_start_position()}")
        print(f"  机器人: {self.get_robot_start_position()}")
        print(f"  人类: {self.get_human_start_position()}")
        print()
        
        print("举升目标:")
        for i, target in enumerate(self.get_lift_targets()):
            print(f"  目标 {i+1}: {target}")
        print()
        
        print("协作区域:")
        collab_zone = self.get_collaboration_zone()
        print(f"  中心: {collab_zone['CENTER']}")
        print(f"  半径: {collab_zone['RADIUS']}")
        print()
        
        print("任务参数:")
        task_params = self.get_task_params()
        for key, value in task_params.items():
            print(f"  {key}: {value}")
        print()
        
        print("环境边界:")
        boundaries = self.get_boundaries()
        for key, value in boundaries.items():
            print(f"  {key}: {value}")
        print("=" * 60)

def load_collaborative_lifting_map(map_config_path: str) -> CollaborativeLiftingMapLoader:
    """
    加载协作举升地图配置
    
    Args:
        map_config_path: 地图配置文件路径
        
    Returns:
        CollaborativeLiftingMapLoader: 地图加载器实例
    """
    return CollaborativeLiftingMapLoader(map_config_path)

if __name__ == "__main__":
    # 测试地图加载器
    map_path = "/home/ubuntu/IRoMan/flow_copolicy/flow_policy/env/cooperative_transport/gym_table/config/maps/collaborative_lifting.yml"
    
    try:
        map_loader = load_collaborative_lifting_map(map_path)
        map_loader.print_config_summary()
    except Exception as e:
        print(f"地图加载失败: {e}")
