import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import calendar

class WeatherDataset(Dataset):
    """气象数据集类"""
    
    def __init__(
        self,
        data_root: str,
        variables_config: Dict[str, Any],
        years: List[int],
        time_window: int,
        time_interval: int = 6,
        shuffle: bool = True,
        split: str = "train"
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            variables_config: 变量配置
            years: 使用的年份列表
            time_window: 时间窗口大小
            time_interval: 时间间隔（小时）
            shuffle: 是否随机打乱
            split: 数据集类型 ("train", "val", "test")
        """
        self.data_root = data_root
        self.variables_config = variables_config
        self.years = years
        self.time_window = time_window
        self.time_interval = time_interval
        self.shuffle = shuffle
        self.split = split
        
        self.logger = logging.getLogger(f"WeatherDataset_{split}")
        
        # 解析变量配置
        self.variable_paths, self.normalization_stats = self._parse_variables()
        self.logger.info(f"总变量数量: {len(self.variable_paths)}")
        
        # 生成时间索引
        self.time_indices = self._generate_time_indices()
        self.logger.info(f"总样本数: {len(self.time_indices)}")
        
        # 调整样本数使其能被时间窗口整除
        self._adjust_samples()
        self.logger.info(f"调整后样本数: {len(self.time_indices)}")
    
    def _parse_variables(self) -> Tuple[List[str], Dict[str, Dict]]:
        """解析变量配置，返回变量路径和归一化统计信息"""
        variable_paths = []
        normalization_stats = {}
        
        # 处理单层变量
        for var_name in self.variables_config.get('single_level', []):
            var_path = os.path.join(self.data_root, var_name)
            if not os.path.exists(var_path):
                self.logger.warning(f"单层变量路径不存在: {var_path}")
                continue
            
            variable_paths.append(var_path)
            
            # 加载归一化统计信息
            stats_file = os.path.join(var_path, 'mean_std.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                normalization_stats[var_path] = stats
            else:
                self.logger.warning(f"归一化统计文件不存在: {stats_file}")
        
        # 处理多层变量
        for var_config in self.variables_config.get('multi_level', []):
            var_name = var_config['variable']
            levels = var_config['levels']
            
            for level in levels:
                var_level_path = os.path.join(self.data_root, var_name, level)
                if not os.path.exists(var_level_path):
                    self.logger.warning(f"多层变量路径不存在: {var_level_path}")
                    continue
                
                variable_paths.append(var_level_path)
                
                # 加载归一化统计信息
                stats_file = os.path.join(var_level_path, 'mean_std.json')
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    normalization_stats[var_level_path] = stats
                else:
                    self.logger.warning(f"归一化统计文件不存在: {stats_file}")
        
        return variable_paths, normalization_stats
    
    def _generate_time_indices(self) -> List[Tuple[str, int]]:
        """生成时间索引列表"""
        time_indices = []
        
        for year in self.years:
            # 判断是否为闰年
            is_leap = calendar.isleap(year)
            hours_in_year = 8784 if is_leap else 8760
            
            # 根据时间间隔计算样本数
            samples_per_year = hours_in_year // self.time_interval
            
            for i in range(0, samples_per_year, self.time_interval // self.time_interval):
                hour_index = i * self.time_interval
                if hour_index < hours_in_year:
                    time_indices.append((str(year), hour_index))
        
        return time_indices
    
    def _adjust_samples(self):
        """调整样本数使其能被时间窗口整除"""
        total_samples = len(self.time_indices)
        remainder = total_samples % self.time_window
        
        if remainder != 0:
            # 丢弃最前面的数据
            self.time_indices = self.time_indices[remainder:]
            self.logger.info(f"丢弃了前{remainder}个样本以满足时间窗口要求")
    
    def _load_npy_file(self, var_path: str, year: str, hour_index: int) -> np.ndarray:
        """
        加载单个npy文件
        
        Args:
            var_path: 变量路径
            year: 年份
            hour_index: 小时索引
        
        Returns:
            加载的数据 (128, 256)
        """
        file_path = os.path.join(var_path, year, f"{year}-{hour_index:04d}.npy")
        
        if os.path.exists(file_path):
            try:
                data = np.load(file_path).astype(np.float32)
                return data
            except Exception as e:
                self.logger.error(f"加载文件失败: {file_path}, 错误: {e}")
        
        # 如果文件不存在，尝试插值
        return self._interpolate_missing_data(var_path, year, hour_index)
    
    def _interpolate_missing_data(self, var_path: str, year: str, hour_index: int) -> np.ndarray:
        """插值缺失数据"""
        self.logger.warning(f"缺失数据，进行插值: {var_path}/{year}/{year}-{hour_index:04d}.npy")
        
        # 寻找最近的两个时间点
        prev_hour = hour_index - self.time_interval
        next_hour = hour_index + self.time_interval
        
        prev_data = None
        next_data = None
        
        # 尝试加载前一个时间点
        if prev_hour >= 0:
            prev_file = os.path.join(var_path, year, f"{year}-{prev_hour:04d}.npy")
            if os.path.exists(prev_file):
                try:
                    prev_data = np.load(prev_file).astype(np.float32)
                except:
                    pass
        
        # 尝试加载后一个时间点
        max_hours = 8784 if calendar.isleap(int(year)) else 8760
        if next_hour < max_hours:
            next_file = os.path.join(var_path, year, f"{year}-{next_hour:04d}.npy")
            if os.path.exists(next_file):
                try:
                    next_data = np.load(next_file).astype(np.float32)
                except:
                    pass
        
        # 执行插值
        if prev_data is not None and next_data is not None:
            return (prev_data + next_data) / 2.0
        elif prev_data is not None:
            return prev_data
        elif next_data is not None:
            return next_data
        else:
            # 如果都无法获取，返回零数组
            self.logger.error(f"无法获取插值数据，返回零数组: {var_path}/{year}/{hour_index}")
            return np.zeros((128, 256), dtype=np.float32)
    
    def _normalize_data(self, data: np.ndarray, var_path: str) -> np.ndarray:
        """标准化数据"""
        if var_path in self.normalization_stats:
            stats = self.normalization_stats[var_path]
            mean = np.array(stats['mean'], dtype=np.float32)
            std = np.array(stats['std'], dtype=np.float32)
            
            # 防止除零
            std = np.where(std == 0, 1.0, std)
            
            return (data - mean) / std
        else:
            self.logger.warning(f"未找到归一化统计信息: {var_path}")
            return data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if self.shuffle and self.split == "train":
            # 训练时返回可能的时间窗口数量
            return len(self.time_indices) - self.time_window + 1
        else:
            # 验证和测试时返回不重叠的时间窗口数量
            return len(self.time_indices) // self.time_window
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            (input_sequence, target): 输入序列和目标
            input_sequence: (time_window-1, 128, 256, num_variables)
            target: (128, 256, num_variables)
        """
        if self.shuffle and self.split == "train":
            # 训练时随机选择连续的时间窗口
            start_idx = idx
        else:
            # 验证和测试时使用不重叠的时间窗口
            start_idx = idx * self.time_window
        
        # 获取时间窗口内的所有时间点
        sequence_data = []
        for i in range(self.time_window):
            time_idx = start_idx + i
            if time_idx >= len(self.time_indices):
                # 处理边界情况
                time_idx = len(self.time_indices) - 1
            
            year, hour_index = self.time_indices[time_idx]
            
            # 加载所有变量的数据
            sample_data = []
            for var_path in self.variable_paths:
                var_data = self._load_npy_file(var_path, year, hour_index)
                var_data = self._normalize_data(var_data, var_path)
                sample_data.append(var_data)
            
            # 堆叠所有变量 (128, 256, num_variables)
            sample = np.stack(sample_data, axis=-1)
            sequence_data.append(sample)
        
        # 转换为张量
        sequence_tensor = torch.from_numpy(np.stack(sequence_data, axis=0))  # (time_window, 128, 256, num_variables)
        
        # 分离输入和目标
        input_sequence = sequence_tensor[:-1]  # (time_window-1, 128, 256, num_variables)
        target = sequence_tensor[-1]           # (128, 256, num_variables)
        
        return input_sequence, target


def get_dataloader(
    config: Dict[str, Any],
    split: str = "train"
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        split: 数据集类型 ("train", "val", "test")
    
    Returns:
        数据加载器
    """
    data_config = config['data']
    model_config = config['model']
    
    # 根据split确定使用的年份
    if split == "train":
        years = list(range(data_config['years']['train'][0], data_config['years']['train'][1] + 1))
        shuffle = data_config.get('shuffle', True)
    elif split == "val":
        years = list(range(data_config['years']['val'][0], data_config['years']['val'][1] + 1))
        shuffle = False
    elif split == "test":
        years = list(range(data_config['years']['test'][0], data_config['years']['test'][1] + 1))
        shuffle = False
    else:
        raise ValueError(f"不支持的数据集类型: {split}")
    
    # 创建数据集
    dataset = WeatherDataset(
        data_root=data_config['data_root'],
        variables_config=data_config['variables'],
        years=years,
        time_window=model_config['time_window'],
        time_interval=data_config['time_interval'],
        shuffle=shuffle,
        split=split
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=data_config['batch_size'],
        shuffle=shuffle,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True if split == "train" else False
    )
    
    return dataloader