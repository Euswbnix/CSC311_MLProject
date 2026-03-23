#!/usr/bin/env python3
"""
CSC311 Machine Learning Challenge — 数据清洗脚本
=================================================
输入:  ml_challenge_dataset.csv (原始数据)
输出:  cleaned_data.csv          (清洗后的结构化特征 + 目标列)

功能:
  1. 列重命名 (缩短)
  2. 删除无用列 (unique_id)
  3. Likert 量表 → 数值编码
  4. 数值列异常值裁剪 + 缺失值填充
  5. willing_to_pay 文本清洗 → 数值提取
  6. 多选列 (season / room / view_with) → 多标签二值化
  7. 自由文本列保留 (供后续 TF-IDF 使用)
  8. 目标变量编码
"""

import pandas as pd
import numpy as np
import re

# ============================================================
# 0. 加载原始数据
# ============================================================
RAW_PATH = "data/ml_challenge_dataset.csv"
OUTPUT_PATH = "cleaned_data.csv"

df = pd.read_csv(RAW_PATH)
print(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"缺失值总计: {df.isnull().sum().sum()}")

# ============================================================
# 1. 列重命名 — 方便后续操作
# ============================================================
column_rename = {
    df.columns[0]:  'unique_id',
    df.columns[1]:  'painting',
    df.columns[2]:  'emotion_intensity',
    df.columns[3]:  'feeling_desc',
    df.columns[4]:  'sombre',
    df.columns[5]:  'content',
    df.columns[6]:  'calm',
    df.columns[7]:  'uneasy',
    df.columns[8]:  'num_colours',
    df.columns[9]:  'num_objects',
    df.columns[10]: 'willing_to_pay',
    df.columns[11]: 'room',
    df.columns[12]: 'view_with',
    df.columns[13]: 'season',
    df.columns[14]: 'food',
    df.columns[15]: 'soundtrack',
}
df.rename(columns=column_rename, inplace=True)
print("\n✓ 列重命名完成")

# ===========================================================
# 2. 删除无用列 unique_id
# ============================================================
df.drop(columns=['unique_id'], inplace=True)
print("✓ 删除 unique_id 列")

# ============================================================
# 3. 目标变量编码 (painting → 整数标签)
# ============================================================
target_map = {
    'The Persistence of Memory': 0,
    'The Starry Night': 1,
    'The Water Lily Pond': 2,
}
df['target'] = df['painting'].map(target_map)
print(f"✓ 目标变量编码: {target_map}")
print(f"  分布:\n{df['target'].value_counts().sort_index().to_string()}")

# ============================================================
# 4. Likert 量表列 → 数值 1–5
#    原始格式: "1 - Strongly disagree" ... "5 - Strongly agree"
# ============================================================
LIKERT_MAP = {
    '1 - Strongly disagree': 1,
    '2 - Disagree': 2,
    '3 - Neutral/Unsure': 3,
    '4 - Agree': 4,
    '5 - Strongly agree': 5,
}
likert_cols = ['sombre', 'content', 'calm', 'uneasy']

for col in likert_cols:
    original_missing = df[col].isnull().sum()
    df[col] = df[col].map(LIKERT_MAP)         # 映射为数值
    new_missing = df[col].isnull().sum()
    # map 之后只有原本缺失的才会是 NaN (不存在映射不上的情况, 因为只有5种取值)
    print(f"✓ Likert [{col}]: 缺失 {original_missing} → NaN, 映射后类型 {df[col].dtype}")

# ============================================================
# 5. emotion_intensity 清洗
#    原始为 float, 范围应在 [1, 10], 但存在 0 和小数 (6.7, 7.5 等)
#    策略: 保留 [0, 10], 小数不做处理 (合理输入), 0 视为有效回答
# ============================================================
col = 'emotion_intensity'
before = df[col].isnull().sum()
# 不需要额外操作, 值域已经合理 [0, 10]
print(f"\n✓ emotion_intensity: 缺失 {before}, 范围 [{df[col].min()}, {df[col].max()}]")

# ============================================================
# 6. num_colours / num_objects 异常值裁剪
#    问题: 存在极端值 (如 100, 43), 明显是乱填
#    策略: 裁剪到 [0, 20] — 画作中不可能有超过20种颜色/物体
# ============================================================
for col in ['num_colours', 'num_objects']:
    n_outlier = (df[col] > 20).sum()
    df[col] = df[col].clip(lower=0, upper=20)
    print(f"✓ {col}: 裁剪 {n_outlier} 个异常值 (>20 → 20)")

# ============================================================
# 7. willing_to_pay 清洗 — 这是最脏的列
#
#    原始内容极其混乱:
#      纯数字:    "0", "150", "90000"
#      带符号:    "$5", "200$", "$1,000", "5 000 000$"
#      带文字:    "300 dollars.", "100 bucks", "200 CAD"
#      带句子:    "I would pay about $1000 for this painting."
#      纯文本:    "a", "Not sure", "pancakes,"
#      超大数:    "100000000", "100 million"
#
#    清洗策略:
#      a) 移除 $, 逗号, 空格
#      b) 处理 "million" / "billion" 等文字乘数
#      c) 用正则提取第一个数字
#      d) 完全无法提取的 → NaN
#      e) 极端值裁剪: > 10,000,000 视为异常 → NaN
# ============================================================
def clean_willing_to_pay(val):
    """从混乱的 willing_to_pay 文本中提取数值 (加元)"""
    if pd.isna(val):
        return np.nan

    text = str(val).lower().strip()

    # 处理 "million" / "billion" 等词
    multiplier = 1
    if 'billion' in text:
        multiplier = 1_000_000_000
    elif 'million' in text:
        multiplier = 1_000_000

    # 移除常见非数字字符: $, 逗号, CAD, dollars, bucks 等
    text = text.replace('$', '').replace(',', '').replace('cad', '').replace('dollars', '')
    text = text.replace('dollar', '').replace('bucks', '').replace('buck', '')

    # 处理空格分隔的数字, 如 "5 000 000" → "5000000", "100 000" → "100000"
    # 只匹配连续的 "数字+空格+数字" 模式
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    # 提取第一个数字 (整数或小数)
    match = re.search(r'(\d+\.?\d*)', text)
    if not match:
        return np.nan

    value = float(match.group(1))

    # 如果文本里有 million/billion 但提取的数字很小, 应用乘数
    if multiplier > 1 and value < 1000:
        value *= multiplier

    # 极端值过滤: 超过 1000万 的视为异常
    if value > 10_000_000:
        return np.nan

    return value

df['willing_to_pay'] = df['willing_to_pay'].apply(clean_willing_to_pay)

valid_count = df['willing_to_pay'].notna().sum()
nan_count = df['willing_to_pay'].isna().sum()
print(f"\n✓ willing_to_pay 清洗完成:")
print(f"  有效值: {valid_count}, 无法解析/异常: {nan_count}")
print(f"  范围: [{df['willing_to_pay'].min()}, {df['willing_to_pay'].max()}]")
print(f"  中位数: {df['willing_to_pay'].median()}")

# ============================================================
# 8. season — 多选列, 逗号分隔 → 多标签二值化
#    可能的取值: Spring, Summer, Fall, Winter (及其组合)
#    例: "Spring,Summer" → season_Spring=1, season_Summer=1, 其余=0
# ============================================================
SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']
for s in SEASONS:
    df[f'season_{s}'] = df['season'].fillna('').str.contains(s, case=False).astype(int)

print(f"\n✓ season 多标签二值化 → {len(SEASONS)} 列: {['season_' + s for s in SEASONS]}")
# 验证
print(f"  各季节选择频率:")
for s in SEASONS:
    print(f"    season_{s}: {df[f'season_{s}'].sum()} ({df[f'season_{s}'].mean()*100:.1f}%)")

# ============================================================
# 9. room — 多选列 → 多标签二值化
#    可能的取值: Bedroom, Bathroom, Office, Living room, Dining room
# ============================================================
ROOMS = ['Bedroom', 'Bathroom', 'Office', 'Living room', 'Dining room']
for r in ROOMS:
    col_name = 'room_' + r.replace(' ', '_')
    df[col_name] = df['room'].fillna('').str.contains(r, case=False).astype(int)

room_cols = ['room_' + r.replace(' ', '_') for r in ROOMS]
print(f"\n✓ room 多标签二值化 → {len(ROOMS)} 列: {room_cols}")
for rc in room_cols:
    print(f"    {rc}: {df[rc].sum()} ({df[rc].mean()*100:.1f}%)")

# ============================================================
# 10. view_with — 多选列 → 多标签二值化
#     可能的取值: Friends, Family members, Coworkers/Classmates,
#                 Strangers, By yourself
# ============================================================
VIEWERS = ['Friends', 'Family members', 'Coworkers/Classmates', 'Strangers', 'By yourself']
for v in VIEWERS:
    col_name = 'view_' + v.replace(' ', '_').replace('/', '_')
    df[col_name] = df['view_with'].fillna('').str.contains(re.escape(v), case=False).astype(int)

view_cols = ['view_' + v.replace(' ', '_').replace('/', '_') for v in VIEWERS]
print(f"\n✓ view_with 多标签二值化 → {len(VIEWERS)} 列: {view_cols}")
for vc in view_cols:
    print(f"    {vc}: {df[vc].sum()} ({df[vc].mean()*100:.1f}%)")

# ============================================================
# 11. 缺失值填充 — 数值列用中位数填充
#     填充对象: emotion_intensity, num_colours, num_objects,
#              willing_to_pay, sombre, content, calm, uneasy
# ============================================================
print(f"\n--- 缺失值填充 (中位数) ---")
fill_cols = ['emotion_intensity', 'sombre', 'content', 'calm', 'uneasy',
             'num_colours', 'num_objects', 'willing_to_pay']
for col in fill_cols:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  ✓ {col}: 填充 {n_missing} 个缺失值, 中位数={median_val}")
    else:
        print(f"  ✓ {col}: 无缺失")

# ============================================================
# 12. 自由文本列缺失值 → 空字符串
#     feeling_desc, food, soundtrack 留给后续 TF-IDF
# ============================================================
text_cols = ['feeling_desc', 'food', 'soundtrack']
for col in text_cols:
    n_missing = df[col].isnull().sum()
    df[col] = df[col].fillna('')
    print(f"  ✓ {col}: {n_missing} 个缺失 → 空字符串")

# ============================================================
# 13. 删除已拆解的原始多选列 (信息已转移到二值列中)
# ============================================================
df.drop(columns=['painting', 'season', 'room', 'view_with'], inplace=True)
print(f"\n✓ 删除原始多选列: painting, season, room, view_with")

# ============================================================
# 14. 整理列顺序, 输出
# ============================================================
# 最终列: target + 数值特征 + 二值特征 + 文本列
num_feature_cols = ['emotion_intensity', 'sombre', 'content', 'calm', 'uneasy',
                    'num_colours', 'num_objects', 'willing_to_pay']
binary_feature_cols = ([f'season_{s}' for s in SEASONS]
                     + room_cols
                     + view_cols)
final_order = ['target'] + num_feature_cols + binary_feature_cols + text_cols
df = df[final_order]

# ============================================================
# 15. 最终检查
# ============================================================
print(f"\n{'='*60}")
print(f"清洗完成!")
print(f"{'='*60}")
print(f"  输出大小: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  数值特征: {len(num_feature_cols)} 列")
print(f"  二值特征: {len(binary_feature_cols)} 列")
print(f"  文本特征: {len(text_cols)} 列 (待 TF-IDF)")
print(f"  残余缺失值: {df[num_feature_cols + binary_feature_cols].isnull().sum().sum()}")
print(f"\n列一览:")
for i, col in enumerate(df.columns):
    dtype_str = str(df[col].dtype)
    sample = df[col].iloc[5]
    print(f"  {i:2d}. {col:30s} | {dtype_str:8s} | 例: {sample}")

# 保存
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ 已保存到 {OUTPUT_PATH}")