import pandas as pd
import numpy as np

# ─── 1. 配置区 ──────────────────────────────────────────────────────────────
# 原始数据文件路径 (根据你刚才截图的文件名)
RAW_FILE = '05_mutation_records_raw_from_api.csv'
# 输出文件路径 (这个文件可以直接喂给你的 DFI 绘图程序)
OUTPUT_FILE = 'mutations_high_quality.csv'

# 【控制开关】：先不要搞太多，这里只填你想测试的蛋白质（注意 GPCRdb 通常带 _human 后缀）
TARGET_PROTEINS = ['adrb2_human', 'ccr5_human', 'aa1r_human', 'glp1r_human', 'mc4r_human', 'aa2br_human', 'aa2ar_human', 'acm1_human', 'cxcr1_human', 'cxcr4_human', 'ccr1_human']

# ─── 2. 读取数据 ──────────────────────────────────────────────────────────────
print("正在读取原始数据...")
df = pd.read_csv(RAW_FILE)

# 过滤出我们要测试的小批量蛋白质
# 转小写比对，防止大小写不一致
df['protein_lower'] = df['protein'].astype(str).str.lower()
targets_lower = [p.lower() for p in TARGET_PROTEINS]
df = df[df['protein_lower'].isin(targets_lower)].copy()
print(f"已过滤出目标蛋白质数据，共 {len(df)} 条记录。")

# ─── 3. 核心分类逻辑 ──────────────────────────────────────────────────────────
# 确保我们需要用来判断的列没有 NaN 报错，填补为空字符串或 0
df['exp_mu_effect_qual'] = df['exp_mu_effect_qual'].fillna('')
df['exp_func'] = df['exp_func'].fillna('')
df['exp_fold_change'] = pd.to_numeric(df['exp_fold_change'], errors='coerce').fillna(0)

# 🔴 Red 组：高价值致病/破坏性突变（预测位于低 DFI 的刚性枢纽）
# 条件 1: 定性描述明确为 Abolished (完全丧失功能)
cond_red_qual = df['exp_mu_effect_qual'].str.contains('Abolish', case=False, na=False)
# 条件 2: 属于功能测试 (Functional)，且活性下降超过 50 倍
cond_red_func = (df['exp_func'].str.contains('Functional', case=False, na=False)) & (df['exp_fold_change'] >= 50)

df['is_red'] = cond_red_qual | cond_red_func

# 🟢 Green 组：耐受型突变（预测位于高 DFI 的柔性区域）
# 条件 1: 定性描述明确为 No effect 或 Unchanged
cond_green_qual = df['exp_mu_effect_qual'].str.contains('No effect|Unchanged', case=False, regex=True, na=False)
# 条件 2: 仅仅是表面结合测试 (Binding)，且影响极小 (Fold change 在 0 到 5 之间)
cond_green_bind = (df['exp_func'].str.contains('Binding', case=False, na=False)) & (df['exp_fold_change'] > 0) & (df['exp_fold_change'] <= 5)

df['is_green'] = cond_green_qual | cond_green_bind

# ─── 4. 汇总为绘图程序所需的格式 ──────────────────────────────────────────────
results = []
for prot in targets_lower:
    sub_df = df[df['protein_lower'] == prot]
    
    # 提取 Red 和 Green 组的突变位置，去重并转为字符串列表
    red_sites = sorted(list(set(sub_df[sub_df['is_red']]['mutation_pos'].dropna().astype(int))))
    green_sites = sorted(list(set(sub_df[sub_df['is_green']]['mutation_pos'].dropna().astype(int))))
    
    # 转为空格分隔的字符串，完美适配你的正则表达式解析器
    red_str = " ".join(map(str, red_sites))
    green_str = " ".join(map(str, green_sites))
    
    # 去掉 _human 后缀，为了和你的蛋白质字典 (如 mc4r) 保持一致
    clean_prot_name = prot.replace('_human', '')
    
    results.append({
        'pertain': clean_prot_name,
        'red': red_str,
        'green': green_str
    })

# 生成最终的 CSV
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n清洗完成！结果已保存至: {OUTPUT_FILE}")
print(out_df)
