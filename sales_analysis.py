"""
============================================================
  SYNENT TECHNOLOGIES – DATA SCIENCE INTERNSHIP
  Task 5: Sales Data Analysis
  Name   : Palakurthy Shiva Sai Goud
  Dataset: Superstore Sales Dataset
============================================================

Task Objectives:
  - Monthly revenue trends
  - Top-selling products
  - Profit analysis
Output:
  - Business insights report with graphs
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Always resolve paths relative to this script's location
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

# ============================================================
#  CONFIGURATION – UPDATE THIS PATH TO YOUR DATASET
# ============================================================

DATA_PATH = os.path.join(_BASE, "data", "Sample - Superstore.csv")

# ============================================================

# Chart styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12
PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

os.makedirs('data/charts', exist_ok=True)

print("=" * 60)
print("   TASK 5 – SALES DATA ANALYSIS")
print("   Superstore Sales Dataset")
print("   Synent Technologies Data Science Internship")
print("=" * 60)


# ============================================================
#  PHASE 1: DATA CLEANING & PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 1: DATA CLEANING & PREPROCESSING")
print("=" * 60)

# --- Load dataset ---
print(f"\n>> Loading dataset from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"\n   ERROR: File not found -> {DATA_PATH}")
    print("   Please update DATA_PATH at the top of this script.")
    sys.exit(1)

try:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
except Exception as e:
    print(f"   ERROR reading file: {e}")
    sys.exit(1)

print(f"   Raw shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Columns   : {list(df.columns)}")

# --- Standardise column names ---
print("\n>> Standardising column names...")
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
# Map common variants to standard names
col_map = {
    'Sub_Category':  'Sub_Category',
    'Sub-Category':  'Sub_Category',
    'Order_Date':    'Order_Date',
    'Ship_Date':     'Ship_Date',
}
df.rename(columns={c: col_map.get(c, c) for c in df.columns}, inplace=True)
print(f"   Normalised columns: {list(df.columns)}")

# --- Parse dates ---
print("\n>> Parsing date columns...")
for date_col in ['Order_Date', 'Ship_Date']:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        n_bad = df[date_col].isna().sum()
        if n_bad:
            print(f"   WARNING: {n_bad} unparseable dates in '{date_col}' – set to NaT")

if 'Order_Date' in df.columns:
    df['Year']    = df['Order_Date'].dt.year
    df['Month']   = df['Order_Date'].dt.month
    df['Quarter'] = df['Order_Date'].dt.quarter
    df['YearMonth'] = df['Order_Date'].dt.to_period('M')
    print(f"   Date range: {df['Order_Date'].min().date()} -> {df['Order_Date'].max().date()}")

# --- Convert numeric columns ---
print("\n>> Converting numeric columns...")
for col in ['Sales', 'Profit', 'Quantity', 'Discount']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# --- Check missing values ---
print("\n>> Checking for missing values...")
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("   No missing values found!")
else:
    for col, cnt in missing.items():
        print(f"   {col}: {cnt} missing ({cnt/len(df)*100:.1f}%)")
    df.dropna(subset=['Sales', 'Profit'], inplace=True)
    print("   Rows with null Sales/Profit dropped.")

# --- Check duplicates ---
print("\n>> Checking for duplicates...")
dups = df.duplicated().sum()
if dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"   {dups} duplicate rows removed.")
else:
    print("   No duplicate rows found.")

# --- Derived columns ---
df['Profit_Margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'] * 100, 0)
df['Revenue_Band']  = pd.cut(df['Sales'],
                              bins=[0, 100, 500, 1000, 5000, 1e9],
                              labels=['< $100', '$100–500', '$500–1K', '$1K–5K', '> $5K'])

print("\n   [Data Cleaning Complete]")


# ============================================================
#  PHASE 2: SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 2: SUMMARY STATISTICS")
print("=" * 60)

total_sales    = df['Sales'].sum()
total_profit   = df['Profit'].sum()
total_orders   = df['Order_ID'].nunique() if 'Order_ID' in df.columns else len(df)
total_customers= df['Customer_ID'].nunique() if 'Customer_ID' in df.columns else 'N/A'
avg_order_val  = total_sales / total_orders
profit_margin  = (total_profit / total_sales) * 100
total_qty      = df['Quantity'].sum()
loss_orders    = (df['Profit'] < 0).sum()

print(f"""
   Total Revenue          : ${total_sales:,.2f}
   Total Profit           : ${total_profit:,.2f}
   Overall Profit Margin  : {profit_margin:.1f}%
   Total Orders           : {total_orders:,}
   Total Customers        : {total_customers}
   Average Order Value    : ${avg_order_val:,.2f}
   Total Units Sold       : {total_qty:,}
   Orders with Loss       : {loss_orders:,} ({loss_orders/len(df)*100:.1f}%)
""")

print("   Sales by Category:")
cat_summary = df.groupby('Category').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum'),
    Orders=('Sales', 'count')
).round(2)
cat_summary['Margin_%'] = (cat_summary['Profit'] / cat_summary['Sales'] * 100).round(1)
for cat, row in cat_summary.iterrows():
    print(f"     {cat:<20} Sales: ${row['Sales']:>12,.0f}  "
          f"Profit: ${row['Profit']:>10,.0f}  Margin: {row['Margin_%']:.1f}%")

print("\n   Sales by Region:")
if 'Region' in df.columns:
    reg_summary = df.groupby('Region').agg(
        Sales=('Sales', 'sum'), Profit=('Profit', 'sum')).round(2)
    for reg, row in reg_summary.iterrows():
        print(f"     {reg:<12} Sales: ${row['Sales']:>12,.0f}  Profit: ${row['Profit']:>10,.0f}")


# ============================================================
#  PHASE 3: CORE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3: CORE ANALYSIS")
print("=" * 60)

# Monthly trend
print("\n>> Monthly Revenue & Profit Trends")
monthly = (df.groupby('YearMonth')[['Sales', 'Profit']].sum()
             .reset_index().sort_values('YearMonth'))
monthly['YearMonth_str'] = monthly['YearMonth'].astype(str)
best_month  = monthly.loc[monthly['Sales'].idxmax(), 'YearMonth_str']
worst_month = monthly.loc[monthly['Profit'].idxmin(), 'YearMonth_str']
print(f"   Best revenue month : {best_month}  (${monthly['Sales'].max():,.0f})")
print(f"   Worst profit month : {worst_month}  (${monthly['Profit'].min():,.0f})")

# Top products
print("\n>> Top-Selling Products")
if 'Product_Name' in df.columns:
    top_products = df.groupby('Product_Name')['Sales'].sum().sort_values(ascending=False).head(10)
    for i, (prod, sales) in enumerate(top_products.items(), 1):
        print(f"   {i:>2}. {prod[:45]:<45} ${sales:>10,.0f}")

# Sub-category profit
print("\n>> Sub-Category Profit Analysis")
sub_profit = df.groupby('Sub_Category')['Profit'].sum().sort_values()
loss_subs  = sub_profit[sub_profit < 0]
gain_subs  = sub_profit[sub_profit > 0]
print(f"   Loss-making sub-categories : {list(loss_subs.index)}")
print(f"   Most profitable            : {gain_subs.idxmax()} (${gain_subs.max():,.0f})")

# Discount effect
print("\n>> Discount vs Profit Analysis")
df['Discount_Band'] = pd.cut(df['Discount'],
                               bins=[-0.01, 0, 0.2, 0.4, 0.6, 1.01],
                               labels=['No Discount', '1–20%', '21–40%', '41–60%', '>60%'])
disc_profit = df.groupby('Discount_Band', observed=True)['Profit_Margin'].mean().round(1)
for band, margin in disc_profit.items():
    print(f"   {band:<15} -> Avg Profit Margin: {margin:>6.1f}%")

print("\n   [Analysis Complete]")


# ============================================================
#  PHASE 4: VISUALIZATION  (6 Charts)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 4: VISUALIZATION")
print("=" * 60)
print("\n   Generating charts... (close each window to continue)\n")

# Larger base font for readability
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13


# ── OBJECTIVE 1: MONTHLY REVENUE TRENDS ─────────────────────

# --- Chart 1: Monthly Revenue + Profit (dual panel) ---
print(">> Chart 1/6: Monthly Revenue & Profit Trends")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Monthly Revenue & Profit Trends (2014 – 2017)', fontsize=16, fontweight='bold')

# Revenue line
ax1.plot(monthly['YearMonth_str'], monthly['Sales'] / 1e3,
         color='steelblue', linewidth=2.5, marker='o', markersize=4)
ax1.fill_between(monthly['YearMonth_str'], monthly['Sales'] / 1e3,
                 alpha=0.15, color='steelblue')
ax1.set_ylabel('Revenue ($K)', fontsize=13)
ax1.set_title('Monthly Revenue  –  Peak: ' + best_month, fontsize=13)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
ax1.grid(axis='y', alpha=0.4)

# Profit bars (green/red)
step = max(1, len(monthly) // 12)
colors_bar = ['coral' if v < 0 else 'mediumseagreen' for v in monthly['Profit']]
ax2.bar(monthly['YearMonth_str'], monthly['Profit'] / 1e3,
        color=colors_bar, edgecolor='white', width=0.8)
ax2.axhline(0, color='black', linewidth=1, linestyle='--')
ax2.set_ylabel('Profit ($K)', fontsize=13)
ax2.set_title('Monthly Profit  –  Green = Profit, Red = Loss', fontsize=13)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
ax2.set_xticks(monthly['YearMonth_str'][::step])
ax2.set_xticklabels(monthly['YearMonth_str'][::step], rotation=45, ha='right', fontsize=11)
ax2.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('data/charts/01_monthly_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 01_monthly_trends.png")


# ── OBJECTIVE 2: TOP-SELLING PRODUCTS ───────────────────────

# --- Chart 2: Top 10 Products by Revenue ---
print("\n>> Chart 2/6: Top 10 Products by Revenue")
top10_prod = df.groupby('Product_Name')['Sales'].sum().sort_values(ascending=False).head(10)
# Shorten long product names for readability
labels = [n[:40] + '…' if len(n) > 40 else n for n in top10_prod.index[::-1]]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(labels, top10_prod.values[::-1] / 1e3,
               color=sns.color_palette('Blues_r', 10), edgecolor='white', height=0.6)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
            f'${w:.1f}K', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Total Revenue ($K)', fontsize=13)
ax.set_title('Top 10 Best-Selling Products by Revenue', fontsize=15, fontweight='bold', pad=12)
ax.set_xlim(0, top10_prod.max() / 1e3 * 1.15)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig('data/charts/02_top_products.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 02_top_products.png")


# --- Chart 3: Sales by Category & Sub-Category ---
print("\n>> Chart 3/6: Sales by Category")
cat_sales_vals = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
cat_colors = ['#4C72B0', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(cat_sales_vals.index, cat_sales_vals.values / 1e3,
              color=cat_colors, edgecolor='white', width=0.5)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
            f'${h:.0f}K', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Revenue ($K)', fontsize=13)
ax.set_title('Total Revenue by Category\n(Which category drives the most sales?)',
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
ax.set_ylim(0, cat_sales_vals.max() / 1e3 * 1.15)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('data/charts/03_category_revenue.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 03_category_revenue.png")


# ── OBJECTIVE 3: PROFIT ANALYSIS ────────────────────────────

# --- Chart 4: Profit by Sub-Category (all, sorted, green/red) ---
print("\n>> Chart 4/6: Profit by Sub-Category")
sub_prof = df.groupby('Sub_Category')['Profit'].sum().sort_values()
bar_colors4 = ['#e05c5c' if v < 0 else '#4caf7d' for v in sub_prof.values]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(sub_prof.index, sub_prof.values / 1e3,
               color=bar_colors4, edgecolor='white', height=0.65)
ax.axvline(0, color='black', linewidth=1.2)
for bar in bars:
    w = bar.get_width()
    x_off = 0.4 if w >= 0 else -0.4
    ha = 'left' if w >= 0 else 'right'
    ax.text(w + x_off, bar.get_y() + bar.get_height() / 2,
            f'${w:.1f}K', va='center', ha=ha, fontsize=10, fontweight='bold')
ax.set_xlabel('Total Profit ($K)', fontsize=13)
ax.set_title('Profit by Sub-Category\n(Red bars = LOSS  |  Green bars = PROFIT)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('data/charts/04_subcategory_profit.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 04_subcategory_profit.png")


# --- Chart 5: Profit Margin by Category ---
print("\n>> Chart 5/6: Profit Margin by Category")
cat_margin = df.groupby('Category')['Profit_Margin'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(cat_margin.index, cat_margin.values,
              color=['#4caf7d' if v >= 0 else '#e05c5c' for v in cat_margin.values],
              edgecolor='white', width=0.5)
ax.axhline(0, color='black', linewidth=1)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
            f'{h:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Profit Margin (%)', fontsize=13)
ax.set_title('Average Profit Margin by Category\n(Which category is most profitable per dollar sold?)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, cat_margin.max() * 1.25)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('data/charts/05_profit_margin_category.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 05_profit_margin_category.png")


# --- Chart 6: Discount Impact on Profit Margin ---
print("\n>> Chart 6/6: Discount Impact on Profit Margin")
disc_band_avg = df.groupby('Discount_Band', observed=True)['Profit_Margin'].mean()
bar_colors6 = ['#4caf7d' if v >= 0 else '#e05c5c' for v in disc_band_avg.values]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(disc_band_avg.index.astype(str), disc_band_avg.values,
              color=bar_colors6, edgecolor='white', width=0.55)
ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
for i, val in enumerate(disc_band_avg.values):
    y_pos = val + 1.5 if val >= 0 else val - 4
    ax.text(i, y_pos, f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_xlabel('Discount Band', fontsize=13)
ax.set_ylabel('Avg Profit Margin (%)', fontsize=13)
ax.set_title('How Discounts Destroy Profit\n(Higher discounts = deeper losses)',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('data/charts/06_discount_impact.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 06_discount_impact.png")

print("\n   [Visualization Complete]")


# ============================================================
#  FINAL: BUSINESS INSIGHTS REPORT
# ============================================================
print("\n" + "=" * 60)
print("  BUSINESS INSIGHTS REPORT")
print("=" * 60)

top_cat    = cat_sales_vals.idxmax()
cat_profit = df.groupby('Category')['Profit'].sum()
low_cat    = cat_profit.idxmin()
top_subcat = df.groupby('Sub_Category')['Sales'].sum().idxmax()
loss_list  = list(sub_prof[sub_prof < 0].index)
top_region = df.groupby('Region')['Sales'].sum().idxmax() if 'Region' in df.columns else 'N/A'
top_seg    = df.groupby('Segment')['Sales'].sum().idxmax() if 'Segment' in df.columns else 'N/A'

print(f"""
  OBJECTIVE 1 – Monthly Revenue Trends
  -------------------------------------
  Best Revenue Month  : {best_month}  (${monthly['Sales'].max():,.0f})
  Worst Profit Month  : {worst_month}  (${monthly['Profit'].min():,.0f})
  Revenue grows steadily each year with Q4 spikes (holiday season)

  OBJECTIVE 2 – Top-Selling Products
  ------------------------------------
  #1 Product          : {top10_prod.index[0][:50]}
     Revenue          : ${top10_prod.iloc[0]:,.0f}
  Top Category        : {top_cat}  (${cat_sales_vals.max():,.0f})

  OBJECTIVE 3 – Profit Analysis
  --------------------------------
  Total Profit        : ${total_profit:,.0f}  (Margin: {profit_margin:.1f}%)
  Most Profitable     : {sub_prof.idxmax()}  (${sub_prof.max():,.0f})
  Loss-making         : {', '.join(loss_list) if loss_list else 'None'}
  Discount Warning    : Discounts >20% push profit into negative territory
  Orders with Loss    : {loss_orders:,} ({loss_orders/len(df)*100:.1f}% of all orders)
""")

print("=" * 60)
print("  All 6 charts saved in: data/charts/")
print("  Task 5 – Sales Data Analysis Complete!")
print("=" * 60)
sys.exit(0)



fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly['YearMonth_str'], monthly['Sales'] / 1e3,
        color='steelblue', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(monthly['YearMonth_str'], monthly['Sales'] / 1e3,
                alpha=0.15, color='steelblue')
step = max(1, len(monthly) // 12)
ax.set_xticks(monthly['YearMonth_str'][::step])
ax.set_xticklabels(monthly['YearMonth_str'][::step], rotation=45, ha='right')
ax.set_title('Monthly Revenue Trend', fontsize=14, pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Revenue ($K)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
plt.tight_layout()
plt.savefig('data/charts/01_monthly_revenue_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 01_monthly_revenue_trend.png")


# --- Chart 2: Monthly Profit Trend ---
print("\n>> Chart 2/10: Monthly Profit Trend")
fig, ax = plt.subplots(figsize=(14, 5))
colors_bar = ['coral' if v < 0 else 'mediumseagreen' for v in monthly['Profit']]
ax.bar(monthly['YearMonth_str'], monthly['Profit'] / 1e3,
       color=colors_bar, edgecolor='white', width=0.8)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(monthly['YearMonth_str'][::step])
ax.set_xticklabels(monthly['YearMonth_str'][::step], rotation=45, ha='right')
ax.set_title('Monthly Profit Trend (Green = Profit, Red = Loss)', fontsize=14, pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Profit ($K)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
plt.tight_layout()
plt.savefig('data/charts/02_monthly_profit_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 02_monthly_profit_trend.png")


# --- Chart 3: Sales by Category (Pie + Bar) ---
print("\n>> Chart 3/10: Sales by Category")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
axes[0].pie(cat_sales, labels=cat_sales.index, autopct='%1.1f%%',
            colors=PALETTE[:len(cat_sales)],
            wedgeprops=dict(edgecolor='white', linewidth=1.5), startangle=140)
axes[0].set_title('Revenue Share by Category', fontsize=13)
cat_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
bars = axes[1].bar(cat_profit.index, cat_profit.values / 1e3,
                   color=PALETTE[:len(cat_profit)], edgecolor='white', width=0.5)
for bar in bars:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'${h:.1f}K', ha='center', va='bottom', fontsize=10)
axes[1].set_title('Total Profit by Category', fontsize=13)
axes[1].set_ylabel('Profit ($K)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
plt.suptitle('Category Performance Overview', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('data/charts/03_category_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 03_category_performance.png")


# --- Chart 4: Top 10 Sub-Categories by Sales ---
print("\n>> Chart 4/10: Top 10 Sub-Categories by Sales")
sub_sales = df.groupby('Sub_Category')['Sales'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(11, 5))
bars = plt.barh(sub_sales.index[::-1], sub_sales.values[::-1] / 1e3,
                color=sns.color_palette('Blues_r', len(sub_sales)), edgecolor='white')
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.5, bar.get_y() + bar.get_height()/2,
             f'${w:.1f}K', va='center', fontsize=9)
plt.xlabel('Total Sales ($K)')
plt.title('Top 10 Sub-Categories by Sales Revenue', fontsize=13)
plt.tight_layout()
plt.savefig('data/charts/04_subcategory_sales.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 04_subcategory_sales.png")


# --- Chart 5: Profit by Sub-Category (Loss in red) ---
print("\n>> Chart 5/10: Profit by Sub-Category")
sub_prof = df.groupby('Sub_Category')['Profit'].sum().sort_values()
colors5  = ['coral' if v < 0 else 'mediumseagreen' for v in sub_prof.values]
plt.figure(figsize=(12, 6))
bars = plt.barh(sub_prof.index, sub_prof.values / 1e3, color=colors5, edgecolor='white')
plt.axvline(0, color='black', linewidth=0.8)
for bar in bars:
    w = bar.get_width()
    x_pos = w + 0.2 if w >= 0 else w - 0.2
    ha = 'left' if w >= 0 else 'right'
    plt.text(x_pos, bar.get_y() + bar.get_height()/2,
             f'${w:.1f}K', va='center', ha=ha, fontsize=8)
plt.xlabel('Total Profit ($K)')
plt.title('Profit by Sub-Category  (Red = Loss, Green = Profit)', fontsize=13)
plt.tight_layout()
plt.savefig('data/charts/05_subcategory_profit.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 05_subcategory_profit.png")


# --- Chart 6: Sales vs Profit Scatter ---
print("\n>> Chart 6/10: Sales vs Profit Scatter")
plt.figure(figsize=(10, 6))
for cat in df['Category'].unique():
    sub = df[df['Category'] == cat]
    plt.scatter(sub['Sales'], sub['Profit'], alpha=0.35, s=20, label=cat)
plt.axhline(0, color='red', linewidth=1, linestyle='--', label='Break-even')
plt.xlabel('Sales ($)')
plt.ylabel('Profit ($)')
plt.title('Sales vs Profit by Category', fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('data/charts/06_sales_vs_profit.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 06_sales_vs_profit.png")


# --- Chart 7: Regional Performance ---
print("\n>> Chart 7/10: Regional Sales & Profit")
if 'Region' in df.columns:
    region_df = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index()
    x = np.arange(len(region_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, region_df['Sales'] / 1e3,  width,
                label='Sales',  color='steelblue', edgecolor='white')
    b2 = ax.bar(x + width/2, region_df['Profit'] / 1e3, width,
                label='Profit', color='mediumseagreen', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(region_df['Region'])
    ax.set_ylabel('Amount ($K)')
    ax.set_title('Sales & Profit by Region', fontsize=13)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'${h:.0f}K', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('data/charts/07_regional_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Saved: 07_regional_performance.png")
