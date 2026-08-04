#!/usr/bin/env bash
# =============================================================================
# install_deps.sh — 通过 isaaclab.sh 安装 Python 依赖（使用国内镜像加速）
#
# 用法:
#   cd /path/to/IsaacLab2.3
#   ./scripts/tools/setup_env/install_deps.sh
#
# 依赖来源: scripts/tools/setup_env/requirements.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "[ERROR] 依赖文件不存在: $REQUIREMENTS"
    exit 1
fi

echo "============================================"
echo "  Install Python Dependencies"
echo "============================================"
echo "  requirements: $REQUIREMENTS"
echo "  mirrors: tsinghua + aliyun + tencent + sjtu"
echo "============================================"
echo ""
echo "将要安装的依赖:"
cat "$REQUIREMENTS"
echo ""

export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
export PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/ https://mirrors.cloud.tencent.com/pypi/simple/ https://mirror.sjtu.edu.cn/pypi/web/simple"

echo "[INFO] PIP_INDEX_URL=$PIP_INDEX_URL"
echo "[INFO] PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL"
echo ""

# 回到项目根目录执行 isaaclab.sh
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "[INFO] 项目根目录: $PROJECT_ROOT"
echo "[INFO] 执行: ./isaaclab.sh -p -m pip install -r $REQUIREMENTS"
echo ""

./isaaclab.sh -p -m pip install -r "$REQUIREMENTS"

echo ""
echo "[OK] 依赖安装完成。"
