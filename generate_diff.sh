#!/usr/bin/env bash
# =============================================================================
# generate_diff.sh — 生成当前分支所有未提交修改的 diff 文件
#
# 用法:
#   ./generate_diff.sh               # 输出 git.diff
#   ./generate_diff.sh my.diff        # 指定输出文件名
#
# 生成内容: 工作区中已追踪文件的所有未提交变更（含已暂存 + 未暂存）
#
# 典型使用场景:
#   开发主机 → 生成 git.diff → 传输到训练主机 → git apply git.diff
# =============================================================================

set -euo pipefail

OUTPUT_FILE="${1:-git.diff}"

# ── 预检 ────────────────────────────────────────────────────────────────────
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "[ERROR] 当前目录不是 Git 仓库，请在项目根目录下运行此脚本。"
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# ── 统计 ────────────────────────────────────────────────────────────────────
STAGED_FILES=$(git diff --cached --name-only)
UNSTAGED_FILES=$(git diff --name-only)

STAGED_COUNT=$(echo "$STAGED_FILES" | grep -c '.' 2>/dev/null || echo "0")
UNSTAGED_COUNT=$(echo "$UNSTAGED_FILES" | grep -c '.' 2>/dev/null || echo "0")

echo "============================================"
echo "  Git Diff 生成器 (未提交修改)"
echo "============================================"
echo "  当前分支:   $CURRENT_BRANCH"
echo "  已暂存变更: $STAGED_COUNT 个文件"
echo "  未暂存变更: $UNSTAGED_COUNT 个文件"
echo "  输出文件:   $OUTPUT_FILE"
echo "============================================"

if [[ "$STAGED_COUNT" == "0" && "$UNSTAGED_COUNT" == "0" ]]; then
    echo ""
    echo "[INFO] 工作区干净，没有未提交的修改。无需生成 diff。"
    exit 0
fi

# ── 生成 diff ───────────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

{
    echo "# ============================================================================="
    echo "# Git Diff — 未提交修改"
    echo "# 分支: $CURRENT_BRANCH"
    echo "# 生成时间: $TIMESTAMP"
    echo "# 已暂存: $STAGED_COUNT  未暂存: $UNSTAGED_COUNT"
    echo "# ============================================================================="
    echo ""

    if [[ "$STAGED_COUNT" -gt 0 ]]; then
        echo "# ── 已暂存变更 (git diff --cached) ─────────────────────────────────────"
        echo "$STAGED_FILES" | while read -r f; do echo "#   $f"; done
        echo ""
        git diff --cached
        echo ""
    fi

    if [[ "$UNSTAGED_COUNT" -gt 0 ]]; then
        echo "# ── 未暂存变更 (git diff) ──────────────────────────────────────────────"
        echo "$UNSTAGED_FILES" | while read -r f; do echo "#   $f"; done
        echo ""
        git diff
        echo ""
    fi

} > "$OUTPUT_FILE"

# ── 结果 ────────────────────────────────────────────────────────────────────
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo ""
echo "[OK] Diff 已生成: $OUTPUT_FILE ($FILE_SIZE)"
echo ""
echo "在目标机器上应用:"
echo "  git apply $OUTPUT_FILE"
echo "  # 或先检查:"
echo "  git apply --stat $OUTPUT_FILE"
echo "  git apply --check $OUTPUT_FILE"
