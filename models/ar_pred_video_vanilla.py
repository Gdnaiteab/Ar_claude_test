import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np


class PatchEmbedding(nn.Module):
    """Patch嵌入层，将输入token化"""
    
    def __init__(
        self, 
        img_size: Tuple[int, int] = (128, 256),
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 9,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 计算patch数量
        self.num_patches_h = img_size[0] // patch_size[0]  # 8
        self.num_patches_w = img_size[1] // patch_size[1]  # 16
        self.num_patches = self.num_patches_h * self.num_patches_w  # 128
        
        # 投影层
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        self.logger = logging.getLogger("PatchEmbedding")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, height, width)
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        self.logger.debug(f"PatchEmbedding输入形状: {x.shape}")
        
        # 确保输入尺寸正确
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入尺寸 {(H, W)} 与预期尺寸 {self.img_size} 不匹配"
        
        # 卷积投影: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        
        # 展平: (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # 转置: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        self.logger.debug(f"PatchEmbedding输出形状: {x.shape}")
        return x


class PositionalEmbedding(nn.Module):
    """位置编码层"""
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 创建可学习的位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )
        
        self.logger = logging.getLogger("PositionalEmbedding")
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            positions: (batch_size, seq_len) 位置索引，如果为None则使用连续位置
        Returns:
            x + pos_emb: (batch_size, seq_len, embed_dim)
        """
        B, seq_len, _ = x.shape
        self.logger.debug(f"PositionalEmbedding输入形状: {x.shape}")
        
        if positions is None:
            # 使用连续位置编码
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # 使用指定位置的编码
            pos_emb = self.pos_embedding[:, positions, :]
        
        result = x + pos_emb.expand(B, -1, -1)
        self.logger.debug(f"PositionalEmbedding输出形状: {result.shape}")
        return result


class MultiHeadAttention(nn.Module):
    """带自回归掩码的多头注意力"""
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        dropout: float = 0.1,
        block_size: int = 128,
        mask_ratio: float = 0.8
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.logger = logging.getLogger("MultiHeadAttention")
    
    def _create_autoregressive_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建自回归掩码"""
        num_blocks = seq_len // self.block_size
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                if j > i:  # 后续区块
                    # 根据掩码率随机掩盖
                    if torch.rand(1).item() < self.mask_ratio:
                        start_i = i * self.block_size
                        end_i = (i + 1) * self.block_size
                        start_j = j * self.block_size
                        end_j = (j + 1) * self.block_size
                        mask[start_i:end_i, start_j:end_j] = float('-inf')
        
        return mask
    
    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            is_training: 是否在训练模式
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        B, seq_len, _ = x.shape
        self.logger.debug(f"MultiHeadAttention输入形状: {x.shape}")
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_len, seq_len)
        
        # 应用自回归掩码
        if is_training:
            mask = self._create_autoregressive_mask(seq_len, x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        out = (attn @ v).transpose(1, 2).reshape(B, seq_len, self.embed_dim)
        out = self.proj(out)
        
        self.logger.debug(f"MultiHeadAttention输出形状: {out.shape}")
        return out


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        block_size: int = 128,
        mask_ratio: float = 0.8
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, block_size, mask_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.logger = logging.getLogger("TransformerEncoderBlock")
    
    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        self.logger.debug(f"EncoderBlock输入形状: {x.shape}")
        
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x), is_training)
        
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        
        self.logger.debug(f"EncoderBlock输出形状: {x.shape}")
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer解码器块（仅交叉注意力，无自注意力）"""
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.logger = logging.getLogger("TransformerDecoderBlock")
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch_size, query_len, embed_dim)
            key_value: (batch_size, kv_len, embed_dim)
        Returns:
            output: (batch_size, query_len, embed_dim)
        """
        self.logger.debug(f"DecoderBlock query形状: {query.shape}, key_value形状: {key_value.shape}")
        
        # 交叉注意力 + 残差连接
        attn_out, _ = self.cross_attn(self.norm1(query), key_value, key_value)
        query = query + attn_out
        
        # MLP + 残差连接
        query = query + self.mlp(self.norm2(query))
        
        self.logger.debug(f"DecoderBlock输出形状: {query.shape}")
        return query


class ARPredVideoVanilla(nn.Module):
    """AR预测视频Vanilla模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_config = config['model']
        
        # 基本参数
        self.time_window = model_config['time_window']
        self.patch_size = tuple(model_config['patch_size'])
        self.embed_dim = model_config['embed_dim']
        self.img_size = (128, 256)
        
        # 计算变量数量
        self.num_variables = self._calculate_num_variables(config['data']['variables'])
        
        # 计算输入通道数（时间窗口 * 变量数）
        self.in_channels = self.num_variables * self.time_window
        
        # Patch嵌入参数
        self.num_patches_per_frame = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.total_patches = self.num_patches_per_frame * self.time_window
        
        # 随机patchify设置
        self.random_patchify = model_config.get('random_patchify', True)
        
        # 构建模型组件
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim
        )
        
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.total_patches,
            embed_dim=self.embed_dim
        )
        
        # Encoder
        encoder_config = model_config['encoder']
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=encoder_config['num_heads'],
                mlp_ratio=encoder_config['mlp_ratio'],
                dropout=encoder_config['dropout'],
                block_size=self.num_patches_per_frame,
                mask_ratio=model_config['mask_ratio']
            ) for _ in range(encoder_config['num_layers'])
        ])
        
        # Decoder
        decoder_config = model_config['decoder']
        self.decoder_dim = decoder_config['hidden_dim']
        
        # 维度映射层
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.decoder_dim)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=self.decoder_dim,
                num_heads=decoder_config['num_heads'],
                mlp_ratio=encoder_config['mlp_ratio'],
                dropout=decoder_config['dropout']
            ) for _ in range(decoder_config['num_layers'])
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.decoder_dim),
            nn.Linear(
                self.decoder_dim, 
                self.patch_size[0] * self.patch_size[1] * self.num_variables
            )
        )
        
        # 随机初始化的解码器查询
        self.decoder_query = nn.Parameter(
            torch.randn(1, self.num_patches_per_frame, self.decoder_dim) * 0.02
        )
        
        self.logger = logging.getLogger("ARPredVideoVanilla")
        self.logger.info(f"模型初始化完成: 变量数={self.num_variables}, "
                        f"总patches={self.total_patches}, "
                        f"每帧patches={self.num_patches_per_frame}")
    
    def _calculate_num_variables(self, variables_config: Dict[str, Any]) -> int:
        """计算总变量数量"""
        total_vars = len(variables_config.get('single_level', []))
        
        for var_config in variables_config.get('multi_level', []):
            total_vars += len(var_config.get('levels', []))
        
        return total_vars
    
    def _reshape_for_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入重塑为patch格式
        Args:
            x: (batch_size, time_window, height, width, num_variables)
        Returns:
            reshaped: (batch_size, in_channels, height, width)
        """
        B, T, H, W, V = x.shape
        self.logger.debug(f"重塑前形状: {x.shape}")
        
        # 在变量维度上叠加时间: (B, T, H, W, V) -> (B, H, W, T*V)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, T * V)
        
        # 转换为通道优先: (B, H, W, T*V) -> (B, T*V, H, W)
        x = x.permute(0, 3, 1, 2)
        
        self.logger.debug(f"重塑后形状: {x.shape}")
        return x
    
    def _random_patchify(self, x: torch.Tensor, original_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        随机移除一个区块作为缺失块
        Args:
            x: (batch_size, total_patches, embed_dim)
            original_sequence: (batch_size, time_window, height, width, num_variables)
        Returns:
            masked_x: (batch_size, (time_window-1)*num_patches_per_frame, embed_dim)
            missing_target: (batch_size, height, width, num_variables)
            missing_block_idx: 缺失区块的索引
        """
        B, total_patches, embed_dim = x.shape
        
        if self.random_patchify and self.training:
            # 随机选择一个区块移除
            missing_block_idx = torch.randint(0, self.time_window, (1,)).item()
        else:
            # 移除最后一个区块
            missing_block_idx = self.time_window - 1
        
        self.logger.debug(f"移除区块索引: {missing_block_idx}")
        
        # 计算区块范围
        start_idx = missing_block_idx * self.num_patches_per_frame
        end_idx = (missing_block_idx + 1) * self.num_patches_per_frame
        
        # 移除缺失区块
        masked_x = torch.cat([
            x[:, :start_idx],
            x[:, end_idx:]
        ], dim=1)
        
        # 获取对应的原始目标数据
        missing_target = original_sequence[:, missing_block_idx]  # (B, H, W, V)
        
        self.logger.debug(f"Patchify后形状: {masked_x.shape}, 目标形状: {missing_target.shape}")
        
        return masked_x, missing_target, missing_block_idx
    
    def _get_missing_block_pos_embedding(self, missing_block_idx: int, batch_size: int) -> torch.Tensor:
        """获取缺失区块的位置编码"""
        start_idx = missing_block_idx * self.num_patches_per_frame
        end_idx = (missing_block_idx + 1) * self.num_patches_per_frame
        
        # 获取缺失区块的位置编码
        missing_pos_emb = self.pos_embed.pos_embedding[:, start_idx:end_idx, :]
        
        # 投影到解码器维度
        missing_pos_emb = self.encoder_to_decoder(missing_pos_emb)
        
        return missing_pos_emb.expand(batch_size, -1, -1)
    
    def forward(self, x: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: (batch_size, time_window, height, width, num_variables) 或
               (batch_size, time_window-1, height, width, num_variables) for inference
        Returns:
            prediction: (batch_size, height, width, num_variables)
            target: (batch_size, height, width, num_variables) or None for inference
        """
        self.logger.debug(f"模型输入形状: {x.shape}")
        
        original_sequence = x
        batch_size = x.shape[0]
        
        # 如果是推理模式且输入是time_window-1，需要特殊处理
        if not is_training and x.shape[1] == self.time_window - 1:
            # 推理模式：输入是time_window-1个样本
            # 添加一个dummy的最后帧
            dummy_frame = torch.zeros_like(x[:, -1:])
            x = torch.cat([x, dummy_frame], dim=1)
            missing_block_idx = self.time_window - 1
            missing_target = None
        else:
            # 训练模式：正常处理
            missing_target = None
            missing_block_idx = None
        
        # 重塑输入
        x_reshaped = self._reshape_for_patches(x)  # (B, T*V, H, W)
        
        # Patch嵌入
        x_patches = self.patch_embed(x_reshaped)  # (B, total_patches, embed_dim)
        
        # 位置编码
        x_with_pos = self.pos_embed(x_patches)  # (B, total_patches, embed_dim)
        
        # Random patchify
        if is_training:
            masked_x, missing_target, missing_block_idx = self._random_patchify(x_with_pos, original_sequence)
        else:
            # 推理时固定移除最后一个区块
            masked_x, _, missing_block_idx = self._random_patchify(x_with_pos, original_sequence)
        
        # Encoder
        encoder_out = masked_x
        for encoder_block in self.encoder_blocks:
            encoder_out = encoder_block(encoder_out, is_training)
        
        self.logger.debug(f"Encoder输出形状: {encoder_out.shape}")
        
        # 映射到解码器维度
        encoder_out = self.encoder_to_decoder(encoder_out)
        
        # 准备解码器查询
        decoder_query = self.decoder_query.expand(batch_size, -1, -1)
        
        # 添加缺失区块的位置编码到查询
        missing_pos_emb = self._get_missing_block_pos_embedding(missing_block_idx, batch_size)
        decoder_query = decoder_query + missing_pos_emb
        
        # Decoder
        decoder_out = decoder_query
        for decoder_block in self.decoder_blocks:
            decoder_out = decoder_block(decoder_out, encoder_out)
        
        self.logger.debug(f"Decoder输出形状: {decoder_out.shape}")
        
        # 输出头
        output = self.output_head(decoder_out)  # (B, num_patches_per_frame, patch_size*patch_size*num_variables)
        
        # 重塑为图像格式
        prediction = self._reshape_patches_to_image(output)  # (B, H, W, V)
        
        self.logger.debug(f"最终预测形状: {prediction.shape}")
        
        if missing_target is not None:
            self.logger.debug(f"目标形状: {missing_target.shape}")
            return prediction, missing_target
        else:
            return prediction, None
    
    def _reshape_patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        将patch重塑回图像格式
        Args:
            patches: (batch_size, num_patches_per_frame, patch_size*patch_size*num_variables)
        Returns:
            image: (batch_size, height, width, num_variables)
        """
        B, num_patches, patch_dim = patches.shape
        
        patch_h, patch_w = self.patch_size
        num_patches_h = self.img_size[0] // patch_h
        num_patches_w = self.img_size[1] // patch_w
        
        # 重塑: (B, num_patches, patch_h * patch_w * V) -> (B, num_patches_h, num_patches_w, patch_h, patch_w, V)
        patches = patches.reshape(B, num_patches_h, num_patches_w, patch_h, patch_w, self.num_variables)
        
        # 重新排列: (B, num_patches_h, patch_h, num_patches_w, patch_w, V)
        patches = patches.permute(0, 1, 3, 2, 4, 5)
        
        # 合并空间维度: (B, H, W, V)
        image = patches.reshape(B, self.img_size[0], self.img_size[1], self.num_variables)
        
        return image