"""
============================================================
  SYNENT TECHNOLOGIES – DATA SCIENCE INTERNSHIP
  Task 5: Sales Data Analysis
  Name   : Palak
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
#  PHASE 4: VISUALIZATION  (5 Focused Charts)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 4: VISUALIZATION")
print("=" * 60)
print("\n   Generating 5 focused charts...")
print("   Each graph will open on screen. Close the graph window to see the next one.\n")

for file_name in os.listdir('data/charts'):
    if file_name.lower().endswith('.png'):
        os.remove(os.path.join('data/charts', file_name))

# Larger base font for readability
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def add_takeaway(fig, lines):
    """Add a simple insight note below each chart."""
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5, 0.03, "\n".join(lines),
        ha='center', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#cccccc')
    )


# ── OBJECTIVE 1: MONTHLY REVENUE TRENDS ─────────────────────

# --- Chart 1: Monthly Revenue + Profit (dual panel) ---
print(">> Chart 1/5: Monthly Revenue & Profit Trends")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
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
add_takeaway(fig, [
    f"Highest revenue month: {best_month} (${monthly['Sales'].max():,.0f})",
    f"Lowest profit month: {worst_month} (${monthly['Profit'].min():,.0f})"
])

plt.tight_layout(rect=[0, 0.15, 1, 0.96])
plt.savefig('data/charts/01_monthly_trends.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("   Saved: 01_monthly_trends.png")


# ── OBJECTIVE 2: TOP-SELLING PRODUCTS ───────────────────────

# --- Chart 2: Top 10 Products by Revenue ---
print("\n>> Chart 2/5: Top 10 Products by Revenue")
top10_prod = df.groupby('Product_Name')['Sales'].sum().sort_values(ascending=False).head(10)
bottom_product = df.groupby('Product_Name')['Sales'].sum().sort_values().head(1)
# Shorten long product names for readability
labels = [n[:40] + '…' if len(n) > 40 else n for n in top10_prod.index[::-1]]

fig, ax = plt.subplots(figsize=(10, 6))
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
add_takeaway(fig, [
    f"Highest-selling product: {top10_prod.index[0][:55]} (${top10_prod.iloc[0]:,.0f})",
    f"Lowest-selling product: {bottom_product.index[0][:55]} (${bottom_product.iloc[0]:,.0f})"
])
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig('data/charts/02_top_products.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("   Saved: 02_top_products.png")


# --- Chart 3: Sales by Category ---
print("\n>> Chart 3/5: Sales by Category")
cat_sales_vals = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
cat_profit_vals = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
cat_colors = ['#4C72B0', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(7.5, 5))
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
add_takeaway(fig, [
    f"Highest revenue category: {cat_sales_vals.idxmax()} (${cat_sales_vals.max():,.0f})",
    f"Lowest profit category: {cat_profit_vals.idxmin()} (${cat_profit_vals.min():,.0f})"
])
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig('data/charts/03_category_revenue.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("   Saved: 03_category_revenue.png")


# ── OBJECTIVE 3: PROFIT ANALYSIS ────────────────────────────

# --- Chart 4: Profit by Sub-Category (all, sorted, green/red) ---
print("\n>> Chart 4/5: Profit by Sub-Category")
sub_prof = df.groupby('Sub_Category')['Profit'].sum().sort_values()
bar_colors4 = ['#e05c5c' if v < 0 else '#4caf7d' for v in sub_prof.values]

fig, ax = plt.subplots(figsize=(10, 6.5))
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
add_takeaway(fig, [
    f"Most profitable sub-category: {sub_prof.idxmax()} (${sub_prof.max():,.0f})",
    f"Biggest loss sub-category: {sub_prof.idxmin()} (${sub_prof.min():,.0f})"
])
plt.tight_layout(rect=[0, 0.16, 1, 1])
plt.savefig('data/charts/04_subcategory_profit.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("   Saved: 04_subcategory_profit.png")


# --- Chart 5: Discount Impact on Profit Margin ---
print("\n>> Chart 5/5: Discount Impact on Profit Margin")
disc_band_avg = df.groupby('Discount_Band', observed=True)['Profit_Margin'].mean()
best_discount_band = disc_band_avg.idxmax()
worst_discount_band = disc_band_avg.idxmin()
bar_colors6 = ['#4caf7d' if v >= 0 else '#e05c5c' for v in disc_band_avg.values]

fig, ax = plt.subplots(figsize=(8, 5))
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
add_takeaway(fig, [
    f"Best margin discount band: {best_discount_band} ({disc_band_avg.max():.1f}%)",
    f"Worst margin discount band: {worst_discount_band} ({disc_band_avg.min():.1f}%)"
])
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig('data/charts/05_discount_impact.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("   Saved: 05_discount_impact.png")

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
print("  5 focused charts saved in: data/charts/")
print("  Task 5 – Sales Data Analysis Complete!")
print("=" * 60)
