#!/usr/bin/env python3
"""
AR预测视频模型训练脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import logging
from datetime import datetime
import random
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, setup_logger, get_dataloader
from models import ARPredVideoVanilla


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """创建优化器"""
    training_config = config['training']
    optimizer_type = training_config.get('optimizer', 'adam').lower()
    lr = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 1e-5)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """创建学习率调度器"""
    training_config = config['training']
    scheduler_config = training_config.get('lr_scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        num_epochs = training_config['num_epochs']
        warmup_epochs = scheduler_config.get('warmup_epochs', 10)
        
        # 使用CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=num_epochs - warmup_epochs, T_mult=1
        )
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    
    return scheduler


def compute_loss(prediction: torch.Tensor, target: torch.Tensor, loss_type: str = 'mse') -> torch.Tensor:
    """计算损失"""
    if loss_type.lower() == 'mse':
        return nn.MSELoss()(prediction, target)
    elif loss_type.lower() == 'mae':
        return nn.L1Loss()(prediction, target)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


def visualize_samples(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    num_samples: int = 4,
    split: str = "val"
) -> Dict[str, Any]:
    """可视化样本"""
    model.eval()
    visualizations = {}
    
    with torch.no_grad():
        for i, (input_seq, target) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            # 模型预测
            prediction, _ = model(torch.cat([input_seq, target.unsqueeze(1)], dim=1), is_training=False)
            
            # 转换为numpy用于可视化
            target_np = target[0].cpu().numpy()  # (H, W, V)
            pred_np = prediction[0].cpu().numpy()  # (H, W, V)
            
            # 创建可视化图
            fig, axes = plt.subplots(2, target_np.shape[-1], figsize=(4*target_np.shape[-1], 8))
            if target_np.shape[-1] == 1:
                axes = axes.reshape(2, 1)
            
            for v in range(target_np.shape[-1]):
                # 真实值
                im1 = axes[0, v].imshow(target_np[:, :, v], cmap='viridis')
                axes[0, v].set_title(f'真实值 - 变量{v}')
                axes[0, v].set_xlabel('经度')
                axes[0, v].set_ylabel('纬度')
                plt.colorbar(im1, ax=axes[0, v])
                
                # 预测值
                im2 = axes[1, v].imshow(pred_np[:, :, v], cmap='viridis')
                axes[1, v].set_title(f'预测值 - 变量{v}')
                axes[1, v].set_xlabel('经度')
                axes[1, v].set_ylabel('纬度')
                plt.colorbar(im2, ax=axes[1, v])
            
            plt.tight_layout()
            
            # 保存到wandb
            visualizations[f'{split}_sample_{i}'] = wandb.Image(fig)
            plt.close(fig)
    
    return visualizations


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    loss_type: str,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'训练 Epoch {epoch}')
    
    for batch_idx, (input_seq, target) in enumerate(pbar):
        # 数据移到设备
        input_seq = input_seq.to(device)  # (B, T-1, H, W, V)
        target = target.to(device)        # (B, H, W, V)
        
        # 拼接输入序列和目标作为完整序列
        full_sequence = torch.cat([input_seq, target.unsqueeze(1)], dim=1)  # (B, T, H, W, V)
        
        logger.debug(f"Batch {batch_idx}: input_seq形状={input_seq.shape}, target形状={target.shape}, full_sequence形状={full_sequence.shape}")
        
        # 前向传播
        optimizer.zero_grad()
        prediction, model_target = model(full_sequence, is_training=True)
        
        logger.debug(f"Batch {batch_idx}: prediction形状={prediction.shape}, model_target形状={model_target.shape}")
        
        # 计算损失
        loss = compute_loss(prediction, model_target, loss_type)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6e}'
        })
        
        # 记录详细日志
        if batch_idx % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                       f'Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6e}')
    
    # 更新学习率
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / num_batches
    logger.info(f'训练 Epoch {epoch} 完成, 平均损失: {avg_loss:.6f}')
    
    return {'train_loss': avg_loss, 'learning_rate': optimizer.param_groups[0]['lr']}


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_type: str,
    split: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'评估 {split}')
        
        for batch_idx, (input_seq, target) in enumerate(pbar):
            # 数据移到设备
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            # 对于评估，我们使用input_seq预测target
            prediction, _ = model(input_seq, is_training=False)
            
            # 计算损失
            loss = compute_loss(prediction, target, loss_type)
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / num_batches
    logger.info(f'{split} 评估完成, 平均损失: {avg_loss:.6f}')
    
    return {f'{split}_loss': avg_loss}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False
):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config
    }
    
    # 保存当前检查点
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)


def main():
    parser = argparse.ArgumentParser(description='AR预测视频模型训练')
    parser.add_argument('--config', type=str, default='config/ar_pred_video_vanilla.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['system']['seed'])
    
    # 设置日志
    logger = setup_logger(
        name="train_ar_pred",
        log_dir="log",
        log_level=config['logging']['log_level']
    )
    
    # 设置设备
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 初始化wandb
    wandb_config = config['logging']['wandb']
    if wandb_config.get('project'):
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            config=config
        )
        logger.info("Wandb初始化完成")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    test_loader = get_dataloader(config, split='test')
    
    logger.info(f"训练集批次数: {len(train_loader)}")
    logger.info(f"验证集批次数: {len(val_loader)}")
    logger.info(f"测试集批次数: {len(test_loader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = ARPredVideoVanilla(config)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # 损失函数类型
    loss_type = config['training'].get('loss_function', 'mse')
    
    # 训练参数
    num_epochs = config['training']['num_epochs']
    save_every = config['training'].get('save_every', 10)
    eval_every = config['evaluation'].get('eval_every', 5)
    visualize_samples_num = config['evaluation'].get('visualize_samples', 4)
    
    # 早停参数
    early_stopping_config = config['training'].get('early_stopping', {})
    patience = early_stopping_config.get('patience', 20)
    min_delta = early_stopping_config.get('min_delta', 1e-6)
    
    # 检查点目录
    checkpoint_dir = 'checkpoints'
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # 训练循环
    logger.info("开始训练...")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{num_epochs-1}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, loss_type, epoch, logger
        )
        
        # 记录训练指标
        metrics_to_log = train_metrics.copy()
        
        # 验证
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            logger.info("开始验证...")
            val_metrics = evaluate(model, val_loader, device, loss_type, 'val', logger)
            metrics_to_log.update(val_metrics)
            
            # 检查是否是最佳模型
            val_loss = val_metrics['val_loss']
            is_best = val_loss < best_val_loss - min_delta
            
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"发现更好的模型! 验证损失: {val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"验证损失未改善: {val_loss:.6f}, 耐心计数: {patience_counter}/{patience}")
            
            # 可视化样本
            if wandb.run:
                logger.info("生成可视化样本...")
                visualizations = visualize_samples(
                    model, val_loader, device, visualize_samples_num, 'val'
                )
                metrics_to_log.update(visualizations)
        else:
            is_best = False
        
        # 保存检查点
        if epoch % save_every == 0 or epoch == num_epochs - 1 or is_best:
            logger.info("保存检查点...")
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                train_metrics['train_loss'], config, 
                checkpoint_dir, is_best
            )
        
        # 记录到wandb
        if wandb.run:
            wandb.log(metrics_to_log, step=epoch)
        
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"早停触发! 在epoch {epoch}停止训练")
            break
    
    # 最终测试
    logger.info("\n开始最终测试...")
    
    # 加载最佳模型
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        logger.info("加载最佳模型进行测试...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_metrics = evaluate(model, test_loader, device, loss_type, 'test', logger)
    
    # 测试集可视化
    if wandb.run:
        test_visualizations = visualize_samples(
            model, test_loader, device, visualize_samples_num, 'test'
        )
        test_metrics.update(test_visualizations)
        wandb.log(test_metrics)
    
    logger.info(f"最终测试结果: {test_metrics}")
    
    # 关闭wandb
    if wandb.run:
        wandb.finish()
    
    logger.info("训练完成!")


if __name__ == "__main__":
    main()