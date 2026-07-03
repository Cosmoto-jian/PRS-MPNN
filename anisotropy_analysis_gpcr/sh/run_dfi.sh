#!/bin/bash
# ============================================================================
# DFI Batch Runner
# 用法:
#   ./run_dfi.sh <filename>    对指定文件运行 DFI 计算
#   ./run_dfi.sh               对该目录下所有 zip/pdb 文件运行 DFI 计算
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PDB_DIR="$PROJECT_DIR/raw/pdb"
DFI_SCRIPT="$PROJECT_DIR/tools/dfi_calc.py"
PYTHON="/opt/anaconda3/envs/simulation_mech/bin/python"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_one() {
    local filepath="$1"
    local filename="$(basename "$filepath")"
    echo -e "${GREEN}>>> Processing: ${filename}${NC}"
    $PYTHON "$DFI_SCRIPT" --pdb "$filepath"
    echo ""
}

# --- main ---
if [ ! -f "$DFI_SCRIPT" ]; then
    echo -e "${RED}Error: dfi_calc.py not found at $DFI_SCRIPT${NC}"
    exit 1
fi

if [ $# -ge 1 ]; then
    # 指定文件模式
    TARGET="$PDB_DIR/$1"
    if [ ! -f "$TARGET" ]; then
        echo -e "${RED}Error: file not found: $TARGET${NC}"
        echo -e "${YELLOW}Available files:${NC}"
        ls "$PDB_DIR"/
        exit 1
    fi
    run_one "$TARGET"
else
    # none 模式: 遍历全部 zip/pdb 文件
    FILES=$(ls "$PDB_DIR"/*.zip "$PDB_DIR"/*.pdb 2>/dev/null)
    if [ -z "$FILES" ]; then
        echo -e "${YELLOW}No zip/pdb files found in $PDB_DIR${NC}"
        exit 0
    fi
    COUNT=$(echo "$FILES" | wc -l | tr -d ' ')
    echo -e "${GREEN}Found ${COUNT} file(s) in ${PDB_DIR}${NC}"
    echo ""
    for f in $FILES; do
        run_one "$f"
    done
    echo -e "${GREEN}=== All done (${COUNT} files processed) ===${NC}"
fi
