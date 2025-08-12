#!/bin/bash

# 确保脚本在错误时退出
set -e

# 清理并重新构建
make clean
make -j$(nproc)

# 运行基准测试
echo "Starting benchmark..."
./main