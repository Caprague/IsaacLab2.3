#!/usr/bin/env bash
# =============================================================================
# switch_env.sh — 切换项目中的 environment 标记（local ↔ server）
#
# 用法:
#   ./scripts/tools/switch_env.sh local     # 切换为本地环境
#   ./scripts/tools/switch_env.sh server    # 切换为服务器环境
#
# 影响的文件（硬编码，精确匹配 = "xxx" 赋值语句，不触碰 == 比较）:
#   - source/isaaclab_assets/isaaclab_assets/robots/unitree.py
#   - source/isaaclab_tasks/.../go2_loco_skill_walk_mid360_depth_10hz_cfg.py
#   - source/isaaclab_tasks/.../go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py
# =============================================================================

set -euo pipefail

# ── 参数校验 ────────────────────────────────────────────────────────────────
MODE="${1:-}"

if [[ "$MODE" != "local" && "$MODE" != "server" ]]; then
    echo "用法: $0 <local|server>"
    echo ""
    echo "  local   — 将所有 environment 标记切换为 \"local\""
    echo "  server  — 将所有 environment 标记切换为 \"server\""
    echo ""
    echo "示例:"
    echo "  $0 server   # 切换到服务器环境"
    echo "  $0 local    # 切换回本地环境"
    exit 1
fi

# ── 项目根目录 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── 需要切换的文件列表 ──────────────────────────────────────────────────────
FILES=(
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_cfg.py"
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py"
)

# ── 确定源/目标字符串 ───────────────────────────────────────────────────────
if [[ "$MODE" == "server" ]]; then
    FROM_STR='"local"'
    TO_STR='"server"'
else
    FROM_STR='"server"'
    TO_STR='"local"'
fi

echo "============================================"
echo "  Environment Switch: → $MODE"
echo "============================================"
echo "  替换: $FROM_STR → $TO_STR"
echo "============================================"
echo ""

# ── 执行替换 ────────────────────────────────────────────────────────────────
CHANGED_FILES=()

for rel_path in "${FILES[@]}"; do
    abs_path="$PROJECT_ROOT/$rel_path"

    if [[ ! -f "$abs_path" ]]; then
        echo "[WARN] 文件不存在，跳过: $rel_path"
        continue
    fi

    # 检查是否包含目标字符串（ = "xxx" 赋值，排除 == "xxx" 比较）
    if grep -q '[^=]= '"$FROM_STR" "$abs_path"; then
        # 执行替换：仅替换 = "xxx" 赋值形式，保留 == "xxx" 比较不动
        sed -i 's/\([^=]\)= '"$FROM_STR"'/\1= '"$TO_STR"'/g' "$abs_path"
        CHANGED_FILES+=("$rel_path")
        echo "[OK] 已切换: $rel_path"
    else
        echo "[SKIP] 无需切换: $rel_path （已是 $MODE 或不含该模式）"
    fi
done

# ── 结果汇总 ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
if [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
    echo "  没有文件被修改（可能已处于 $MODE 环境）。"
else
    echo "  已修改 ${#CHANGED_FILES[@]} 个文件:"
    for f in "${CHANGED_FILES[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================"
