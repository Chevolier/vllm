#!/bin/bash
# 步骤 1: 安装依赖
echo “正在卸载 flash-attn...”
uv pip uninstall -y flash-attn
echo “正在安装 flash-attn...”
uv pip install flash-attn==2.5.8 --no-build-isolation
echo “正在安装项目（可编辑模式）...”
uv pip install -e .
echo “项目安装完成。”
echo “”
echo “正在安装所需的 Python 库...”
uv pip install -r requirements/audio.txt
echo “依赖安装完成。”
echo “”