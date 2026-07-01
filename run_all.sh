#!/bin/bash
# 全自动 DFI 计算 + 绘图脚本

# ---- 配置 ----
CLAC_DIR="/Users/zizhenghe/Documents/DFI/PRS-MPNN-main/src/PRS"
PDB_DIR="./PDB_file"
OUTPUT_DIR="./results"
PYTHON="python3"

PROTEINS=("aa2ar" "cxcr4" "adrb2" "aa1r" "aa2br" "acm1" "ccr1" "ccr5" "cxcr1" "glp1r" "mc4r")
#PROTEINS=("O75899" "P21731" "Q9H244" "P25021" "P28221" "P28222" "P29275" "P30989" "P49286")
DFI_SCRIPT="${CLAC_DIR}/dfi_calc6.py"
PLOT_SCRIPT="./plot.py"

mkdir -p "$OUTPUT_DIR"
# ---- 1. 运行 dfi_calc5.py ----
for prot in "${PROTEINS[@]}"; do
    pdb_file="$PDB_DIR/${prot}.pdb"
    if [ ! -f "$pdb_file" ]; then
        echo "⚠️  警告: 缺少 $pdb_file，跳过 $prot"
        continue
    fi
    echo "▶️  正在处理 $prot ..."
    $PYTHON "$DFI_SCRIPT" --pdb "$pdb_file" --mode seq --chain A
done

# ---- 1.5 重命名关键条件 ----
for prot in "${PROTEINS[@]}"; do
    mv "${prot}_memb16_Z_anisotropy_analysis.csv"     "${prot}_EXP.csv" 2>/dev/null
    mv "${prot}_noMemb_Z_anisotropy_analysis.csv"     "${prot}_CTRL1.csv" 2>/dev/null
    mv "${prot}_noMemb_total_anisotropy_analysis.csv" "${prot}_CTRL2.csv" 2>/dev/null
    mv "${prot}_memb16_total_anisotropy_analysis.csv" "${prot}_CTRL3.csv" 2>/dev/null
done

# ---- 2. 移动【所有】相关的 CSV 文件到输出目录 ----
echo "📦 移动 CSV 文件到 $OUTPUT_DIR ..."
# 把剩下的分析文件移走
mv *_anisotropy_analysis.csv "$OUTPUT_DIR/" 2>/dev/null
# 核心修复：把刚刚重命名的关键文件也移走
mv *_EXP.csv *_CTRL1.csv *_CTRL2.csv *_CTRL3.csv "$OUTPUT_DIR/" 2>/dev/null
# ---- 3. 执行绘图 ----
cd "$OUTPUT_DIR" || exit
$PYTHON "../$PLOT_SCRIPT" --output_dir ./
cd - > /dev/null

echo "✅ 全部完成！"
