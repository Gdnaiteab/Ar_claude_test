import yaml
import os
from typing import Dict, Any
import logging

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"成功加载配置文件: {config_path}")
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")

def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置文件的有效性
    
    Args:
        config: 配置字典
    """
    required_keys = ['data', 'model', 'training']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必要字段: {key}")
    
    # 验证数据配置
    data_config = config['data']
    if 'data_root' not in data_config:
        raise ValueError("数据配置缺少data_root字段")
    
    # 验证变量配置
    variables = data_config.get('variables', {})
    single_level = variables.get('single_level', [])
    multi_level = variables.get('multi_level', [])
    
    total_vars = len(single_level)
    for var in multi_level:
        total_vars += len(var.get('levels', []))
    
    if total_vars < 9:
        raise ValueError(f"总变量数量不足9个，当前为{total_vars}个")
    
    logging.info("配置文件验证通过")