#!/usr/bin/env python3
"""
AR预测视频模型推理脚本
"""

import os
import sys
import argparse
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, setup_logger
from models import ARPredVideoVanilla


class WeatherInference:
    """气象数据推理类"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """
        初始化推理器
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            device: 计算设备
        """
        self.config = load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.logger = setup_logger(
            name="inference_ar_pred",
            log_dir="log",
            log_level=self.config['logging']['log_level']
        )
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.logger.info(f"推理器初始化完成，使用设备: {self.device}")
        
        # 模型参数
        self.time_window = self.config['model']['time_window']
        self.img_size = (128, 256)
        self.num_variables = self._calculate_num_variables()
        
        # 加载归一化统计信息
        self.normalization_stats = self._load_normalization_stats()
    
    def _calculate_num_variables(self) -> int:
        """计算总变量数量"""
        variables_config = self.config['data']['variables']
        total_vars = len(variables_config.get('single_level', []))
        
        for var_config in variables_config.get('multi_level', []):
            total_vars += len(var_config.get('levels', []))
        
        return total_vars
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """加载训练好的模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 创建模型
        model = ARPredVideoVanilla(self.config)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"成功加载模型检查点: {checkpoint_path}")
        self.logger.info(f"检查点epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"检查点损失: {checkpoint.get('loss', 'unknown')}")
        
        return model
    
    def _load_normalization_stats(self) -> Dict[str, Dict]:
        """加载归一化统计信息"""
        data_root = self.config['data']['data_root']
        variables_config = self.config['data']['variables']
        normalization_stats = {}
        
        # 处理单层变量
        for var_name in variables_config.get('single_level', []):
            var_path = os.path.join(data_root, var_name)
            stats_file = os.path.join(var_path, 'mean_std.json')
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                normalization_stats[var_name] = stats
            else:
                self.logger.warning(f"归一化统计文件不存在: {stats_file}")
        
        # 处理多层变量
        for var_config in variables_config.get('multi_level', []):
            var_name = var_config['variable']
            levels = var_config['levels']
            
            for level in levels:
                var_level_key = f"{var_name}_{level}"
                var_level_path = os.path.join(data_root, var_name, level)
                stats_file = os.path.join(var_level_path, 'mean_std.json')
                
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    normalization_stats[var_level_key] = stats
                else:
                    self.logger.warning(f"归一化统计文件不存在: {stats_file}")
        
        return normalization_stats
    
    def _normalize_data(self, data: np.ndarray, var_key: str) -> np.ndarray:
        """标准化数据"""
        if var_key in self.normalization_stats:
            stats = self.normalization_stats[var_key]
            mean = np.array(stats['mean'], dtype=np.float32)
            std = np.array(stats['std'], dtype=np.float32)
            
            # 防止除零
            std = np.where(std == 0, 1.0, std)
            
            return (data - mean) / std
        else:
            self.logger.warning(f"未找到归一化统计信息: {var_key}")
            return data
    
    def _denormalize_data(self, data: np.ndarray, var_key: str) -> np.ndarray:
        """反标准化数据"""
        if var_key in self.normalization_stats:
            stats = self.normalization_stats[var_key]
            mean = np.array(stats['mean'], dtype=np.float32)
            std = np.array(stats['std'], dtype=np.float32)
            
            return data * std + mean
        else:
            self.logger.warning(f"未找到归一化统计信息: {var_key}")
            return data
    
    def load_npy_data(self, file_paths: List[str]) -> np.ndarray:
        """
        加载npy数据文件
        
        Args:
            file_paths: npy文件路径列表，按变量顺序排列
        
        Returns:
            data: (time_steps, height, width, num_variables)
        """
        if len(file_paths) == 0:
            raise ValueError("文件路径列表不能为空")
        
        # 检查所有文件是否存在
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 获取时间步数
        time_steps = len(file_paths) // self.num_variables
        
        if len(file_paths) != time_steps * self.num_variables:
            raise ValueError(f"文件数量({len(file_paths)})不符合预期({time_steps * self.num_variables})")
        
        # 加载数据
        data_list = []
        
        for t in range(time_steps):
            time_data = []
            
            # 获取变量键的顺序
            var_keys = self._get_variable_keys()
            
            for v, var_key in enumerate(var_keys):
                file_idx = t * self.num_variables + v
                file_path = file_paths[file_idx]
                
                # 加载npy文件
                var_data = np.load(file_path).astype(np.float32)
                
                # 标准化
                var_data = self._normalize_data(var_data, var_key)
                
                time_data.append(var_data)
            
            # 堆叠变量维度
            time_sample = np.stack(time_data, axis=-1)  # (H, W, V)
            data_list.append(time_sample)
        
        # 堆叠时间维度
        data = np.stack(data_list, axis=0)  # (T, H, W, V)
        
        self.logger.info(f"成功加载数据，形状: {data.shape}")
        return data
    
    def _get_variable_keys(self) -> List[str]:
        """获取变量键列表"""
        variables_config = self.config['data']['variables']
        var_keys = []
        
        # 添加单层变量
        for var_name in variables_config.get('single_level', []):
            var_keys.append(var_name)
        
        # 添加多层变量
        for var_config in variables_config.get('multi_level', []):
            var_name = var_config['variable']
            levels = var_config['levels']
            
            for level in levels:
                var_keys.append(f"{var_name}_{level}")
        
        return var_keys
    
    def predict_single_step(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        单步预测
        
        Args:
            input_sequence: 输入序列 (time_window-1, height, width, num_variables)
        
        Returns:
            prediction: 预测结果 (height, width, num_variables)
        """
        if input_sequence.shape[0] != self.time_window - 1:
            raise ValueError(f"输入序列长度应为{self.time_window-1}，实际为{input_sequence.shape[0]}")
        
        if input_sequence.shape[1:] != (self.img_size[0], self.img_size[1], self.num_variables):
            raise ValueError(f"输入序列形状应为{(self.img_size[0], self.img_size[1], self.num_variables)}，"
                           f"实际为{input_sequence.shape[1:]}")
        
        # 转换为张量并添加batch维度
        input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).to(self.device)
        
        self.logger.debug(f"输入张量形状: {input_tensor.shape}")
        
        # 模型预测
        with torch.no_grad():
            prediction, _ = self.model(input_tensor, is_training=False)
        
        # 转换回numpy
        prediction_np = prediction[0].cpu().numpy()
        
        self.logger.debug(f"预测结果形状: {prediction_np.shape}")
        
        return prediction_np
    
    def predict_multi_step(
        self, 
        initial_sequence: np.ndarray, 
        num_steps: int,
        use_true_data: bool = False,
        true_sequence: np.ndarray = None
    ) -> List[np.ndarray]:
        """
        多步预测
        
        Args:
            initial_sequence: 初始序列 (time_window-1, height, width, num_variables)
            num_steps: 预测步数
            use_true_data: 是否使用真实数据进行预测（不使用预测数据）
            true_sequence: 真实数据序列 (total_time_steps, height, width, num_variables)
        
        Returns:
            predictions: 预测结果列表
        """
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for step in range(num_steps):
            # 单步预测
            prediction = self.predict_single_step(current_sequence)
            predictions.append(prediction)
            
            if use_true_data and true_sequence is not None:
                # 使用真实数据更新序列
                if step + self.time_window - 1 < true_sequence.shape[0]:
                    next_true = true_sequence[step + self.time_window - 1]
                    current_sequence = np.concatenate([
                        current_sequence[1:],
                        next_true[np.newaxis, :]
                    ], axis=0)
                else:
                    # 如果没有更多真实数据，使用预测数据
                    current_sequence = np.concatenate([
                        current_sequence[1:],
                        prediction[np.newaxis, :]
                    ], axis=0)
            else:
                # 使用预测数据更新序列
                current_sequence = np.concatenate([
                    current_sequence[1:],
                    prediction[np.newaxis, :]
                ], axis=0)
            
            self.logger.info(f"完成第{step+1}步预测")
        
        return predictions
    
    def denormalize_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """
        反标准化预测结果
        
        Args:
            prediction: 标准化的预测结果 (height, width, num_variables)
        
        Returns:
            denormalized: 反标准化的预测结果
        """
        var_keys = self._get_variable_keys()
        denormalized = prediction.copy()
        
        for v, var_key in enumerate(var_keys):
            denormalized[:, :, v] = self._denormalize_data(prediction[:, :, v], var_key)
        
        return denormalized
    
    def visualize_prediction(
        self, 
        prediction: np.ndarray, 
        target: np.ndarray = None,
        save_path: str = None,
        var_names: List[str] = None
    ) -> plt.Figure:
        """
        可视化预测结果
        
        Args:
            prediction: 预测结果 (height, width, num_variables)
            target: 真实值 (height, width, num_variables)，可选
            save_path: 保存路径，可选
            var_names: 变量名称列表，可选
        
        Returns:
            fig: matplotlib图形对象
        """
        num_variables = prediction.shape[-1]
        
        if var_names is None:
            var_names = [f'变量{i}' for i in range(num_variables)]
        
        if target is not None:
            # 显示预测和真实值对比
            fig, axes = plt.subplots(2, num_variables, figsize=(4*num_variables, 8))
            if num_variables == 1:
                axes = axes.reshape(2, 1)
            
            for v in range(num_variables):
                # 真实值
                im1 = axes[0, v].imshow(target[:, :, v], cmap='viridis')
                axes[0, v].set_title(f'真实值 - {var_names[v]}')
                axes[0, v].set_xlabel('经度')
                axes[0, v].set_ylabel('纬度')
                plt.colorbar(im1, ax=axes[0, v])
                
                # 预测值
                im2 = axes[1, v].imshow(prediction[:, :, v], cmap='viridis')
                axes[1, v].set_title(f'预测值 - {var_names[v]}')
                axes[1, v].set_xlabel('经度')
                axes[1, v].set_ylabel('纬度')
                plt.colorbar(im2, ax=axes[1, v])
        else:
            # 只显示预测值
            fig, axes = plt.subplots(1, num_variables, figsize=(4*num_variables, 4))
            if num_variables == 1:
                axes = [axes]
            
            for v in range(num_variables):
                im = axes[v].imshow(prediction[:, :, v], cmap='viridis')
                axes[v].set_title(f'预测值 - {var_names[v]}')
                axes[v].set_xlabel('经度')
                axes[v].set_ylabel('纬度')
                plt.colorbar(im, ax=axes[v])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"可视化结果保存到: {save_path}")
        
        return fig
    
    def save_prediction(self, prediction: np.ndarray, save_path: str):
        """
        保存预测结果
        
        Args:
            prediction: 预测结果
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, prediction)
        self.logger.info(f"预测结果保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='AR预测视频模型推理')
    parser.add_argument('--config', type=str, default='config/ar_pred_video_vanilla.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                       help='输入npy文件路径列表')
    parser.add_argument('--output_dir', type=str, default='tmp/inference_results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化结果')
    parser.add_argument('--multi_step', type=int, default=1,
                       help='多步预测步数')
    parser.add_argument('--use_true_data', action='store_true',
                       help='多步预测时是否使用真实数据')
    args = parser.parse_args()
    
    # 创建推理器
    inference = WeatherInference(args.config, args.checkpoint)
    
    # 加载输入数据
    input_data = inference.load_npy_data(args.input_files)
    
    # 检查数据长度
    required_length = inference.time_window - 1 + args.multi_step
    if input_data.shape[0] < required_length:
        raise ValueError(f"输入数据长度不足，需要至少{required_length}个时间步，实际{input_data.shape[0]}")
    
    # 准备初始序列
    initial_sequence = input_data[:inference.time_window-1]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.multi_step == 1:
        # 单步预测
        inference.logger.info("开始单步预测...")
        prediction = inference.predict_single_step(initial_sequence)
        
        # 反标准化
        prediction_denorm = inference.denormalize_prediction(prediction)
        
        # 保存结果
        save_path = os.path.join(args.output_dir, 'prediction_step_1.npy')
        inference.save_prediction(prediction_denorm, save_path)
        
        # 可视化
        if args.visualize:
            # 准备真实值（如果有）
            target = None
            if input_data.shape[0] >= inference.time_window:
                target = input_data[inference.time_window-1]
                target_denorm = inference.denormalize_prediction(target)
            else:
                target_denorm = None
            
            fig = inference.visualize_prediction(
                prediction_denorm, 
                target_denorm,
                save_path=os.path.join(args.output_dir, 'prediction_step_1.png')
            )
            plt.close(fig)
        
        inference.logger.info("单步预测完成")
    
    else:
        # 多步预测
        inference.logger.info(f"开始{args.multi_step}步预测...")
        
        # 准备真实数据（如果使用）
        true_sequence = None
        if args.use_true_data and input_data.shape[0] >= required_length:
            true_sequence = input_data
        
        predictions = inference.predict_multi_step(
            initial_sequence, 
            args.multi_step,
            args.use_true_data,
            true_sequence
        )
        
        # 处理每个预测结果
        for step, prediction in enumerate(predictions):
            step_num = step + 1
            
            # 反标准化
            prediction_denorm = inference.denormalize_prediction(prediction)
            
            # 保存结果
            save_path = os.path.join(args.output_dir, f'prediction_step_{step_num}.npy')
            inference.save_prediction(prediction_denorm, save_path)
            
            # 可视化
            if args.visualize:
                # 准备真实值（如果有）
                target_denorm = None
                if input_data.shape[0] > inference.time_window - 1 + step:
                    target = input_data[inference.time_window - 1 + step]
                    target_denorm = inference.denormalize_prediction(target)
                
                fig = inference.visualize_prediction(
                    prediction_denorm,
                    target_denorm,
                    save_path=os.path.join(args.output_dir, f'prediction_step_{step_num}.png')
                )
                plt.close(fig)
        
        inference.logger.info(f"{args.multi_step}步预测完成")
    
    inference.logger.info(f"所有结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()