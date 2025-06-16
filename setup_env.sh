#!/bin/bash

# 创建conda虚拟环境
echo "创建conda虚拟环境: ar_pred"
conda create -n ar_pred python=3.10 -y

echo "虚拟环境创建完成！"
echo "激活环境请运行: conda activate ar_pred"
echo "然后安装依赖: pip install -r requirements.txt"