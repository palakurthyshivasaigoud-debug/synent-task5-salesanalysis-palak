# synent-task5-salesanalysis-palak

## Task 5: Superstore Sales Analysis

This project analyzes the Superstore sales dataset for the Synent Technologies Data Science Internship. The goal is to convert raw order data into business insights about revenue, products, categories, discounts, and profit.

## Problem Statement

A retail business needs to understand where revenue is coming from and where profit is being lost. I analyzed monthly sales movement, best-selling products, category performance, regional performance, and discount impact to identify clear business takeaways.

## Dataset

**Source:** [Sample Superstore](https://www.kaggle.com/datasets/naveenkumar20bps1137/sample-superstore) - Kaggle

Local file:

```text
data/Sample - Superstore.csv
```

Important columns include `Order Date`, `Ship Date`, `Segment`, `Region`, `Category`, `Sub-Category`, `Product Name`, `Sales`, `Quantity`, `Discount`, and `Profit`.

## Approach

1. Loaded the Superstore CSV file from the repository `data` folder.
2. Standardized column names for easier coding.
3. Parsed order and ship dates.
4. Checked missing values and duplicate rows.
5. Created new fields such as year, month, quarter, revenue band, discount band, and profit margin.
6. Analyzed monthly revenue, product sales, category performance, region performance, and discount effect.
7. Saved charts for the main business insights.

## Key Results

- Total sales are around 2.30 million dollars and total profit is around 286 thousand dollars.
- Technology is the strongest category by revenue and profit margin.
- Furniture generates high revenue but has weak profitability.
- Tables, Bookcases, and Supplies are loss-making sub-categories.
- High discounts reduce profit sharply and can push orders into loss.
- Q4 months show strong revenue peaks, especially around the holiday period.

## Visualizations

Charts are saved in `data/charts/`.

| Chart | Description |
| --- | --- |
| `01_monthly_trends.png` | Monthly revenue and profit trend |
| `02_top_products.png` | Top products by revenue |
| `03_category_revenue.png` | Revenue by category |
| `04_subcategory_profit.png` | Profit and loss by sub-category |
| `05_discount_impact.png` | Discount band vs profit margin |

## How to Run

```bash
pip install -r requirements.txt
python sales_analysis.py
```

## Repository Structure

```text
synent-task5-salesanalysis-palak/
|-- sales_analysis.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- Sample - Superstore.csv
|   `-- charts/
`-- .gitignore
```

## Internship Requirement Mapping

| Requirement | Status |
| --- | --- |
| Monthly revenue trends | Completed |
| Top-selling products | Completed |
| Profit analysis | Completed |
| Business insights report | Completed |
| Dataset included | Completed |

## Author

Palakurthy Shiva Sai Goud

Submitted for Synent Technologies Data Science Internship - Task 5.
