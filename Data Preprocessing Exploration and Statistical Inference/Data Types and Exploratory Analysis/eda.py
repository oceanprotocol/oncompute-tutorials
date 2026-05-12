# ============================================================
# eda.py — Exploratory Data Analysis: Corporate Financial Records
# ============================================================
# Applies the four EDA steps from the tutorial to the actual dataset:
#   1. Data Types        — dtype inspection and cardinality checks
#   2. Descriptive Stats — summary metrics, value counts, missing data
#   3. Distributions     — histograms, box plots, skewness, IQR outliers
#   4. Correlations      — Pearson correlation matrix and scatter pairs
#
# All plots are saved to plots/ next to this script.
# Run: python eda.py
# Requirements: pip install pandas numpy matplotlib seaborn scipy
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "corporate_financial_data.csv"
PLOTS_DIR    = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8FAFC",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    ":",
    "font.size":         10,
})

C_BLUE   = "#2980B9"
C_RED    = "#C0392B"
C_ORANGE = "#E67E22"
C_GREEN  = "#27AE60"
C_PURPLE = "#8E44AD"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(PLOTS_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → plots/{name}")


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Return boolean mask — True where value is beyond Tukey fences."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


# ── Load and cast types ───────────────────────────────────────
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}\nRun dataset.py first."
    )

df = pd.read_csv(DATASET_PATH)
df["sector"]  = df["sector"].astype("category")
df["country"] = df["country"].astype("category")
df["listed"]  = df["listed"].astype(bool)

NUM_COLS = ["employees", "revenue_mUSD", "profit_margin_pct",
            "rd_spending_mUSD", "debt_ratio"]

print("=" * 65)
print(f"Dataset loaded.  Shape: {df.shape}")
print(df.head(3).to_string())
print("=" * 65)


# ════════════════════════════════════════════════════════════
# SECTION 1 — DATA TYPES
# ════════════════════════════════════════════════════════════
print("\n─── SECTION 1: DATA TYPES ─────────────────────────────\n")

# df.dtypes shows pandas' inferred type for every column.
print("Column dtypes after casting:")
print(df.dtypes.to_string())

# Cardinality: a true categorical column has far fewer unique
# values than rows; a near-unique column is likely free text.
print("\nCardinality — unique values per column:")
for col in df.columns:
    note = ""
    if col == "company_name":
        note = "  ← text / near-unique ID"
    elif df[col].nunique() == 2:
        note = "  ← binary"
    print(f"  {col:22s}: {df[col].nunique():5d} unique  [{df[col].dtype}]{note}")

# Missing values — always check before any further analysis.
print("\nMissing value counts  (df.isnull().sum()):")
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("  No missing values.")
else:
    for col, n in missing.items():
        print(f"  {col:22s}: {n} missing ({n / len(df) * 100:.1f}%)")


# ════════════════════════════════════════════════════════════
# SECTION 2 — DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════
print("\n─── SECTION 2: DESCRIPTIVE STATISTICS ─────────────────\n")

# df.describe() generates mean, std, min, max and percentiles
# for all numerical columns at once.
print("Numerical summary  (df.describe):")
print(df.describe(include=[float, int]).round(2).to_string())

# Categorical summary: value_counts and mode.
print("\nValue counts — sector:")
print(df["sector"].value_counts().to_string())
print(f"\n  Mode: {df['sector'].mode()[0]}")

print("\nValue counts — country:")
print(df["country"].value_counts().to_string())
print(f"\n  Mode: {df['country'].mode()[0]}")

print(f"\nListed firms: {df['listed'].sum()} ({df['listed'].mean()*100:.1f}%)")

# Mean vs median gap: a large gap signals skew or outliers.
# If mean > median the distribution has a long right tail.
print("\nMean vs Median — large gap signals skew or outliers:")
for col in ["revenue_mUSD", "employees", "profit_margin_pct", "debt_ratio"]:
    s      = df[col].dropna()
    mean   = s.mean()
    median = s.median()
    gap    = mean - median
    signal = "right-skew" if gap > 0.01 * abs(mean) else "symmetric"
    print(f"  {col:22s}: mean={mean:10.2f}  median={median:8.2f}  "
          f"gap={gap:+8.1f}  ({signal})")


# ════════════════════════════════════════════════════════════
# SECTION 3 — DISTRIBUTIONS AND OUTLIER DETECTION
# ════════════════════════════════════════════════════════════
print("\n─── SECTION 3: DISTRIBUTIONS AND OUTLIER DETECTION ────\n")

# Skewness: near 0 = symmetric; positive = right tail; negative = left tail.
# IQR outlier rule (Tukey fences): values beyond Q1/Q3 ± 1.5×IQR are flagged.
print("Skewness and IQR outlier counts per numerical column:")
for col in NUM_COLS:
    s     = df[col].dropna()
    skew  = stats.skew(s)
    n_out = iqr_outlier_mask(s).sum()
    print(f"  {col:22s}: skewness = {skew:+.2f}   "
          f"IQR outliers = {n_out:3d} ({n_out/len(s)*100:.1f}%)")

# ── Plot 1: Histograms — all numerical columns ─────────────
# Mean and median lines make the skew direction immediately visible.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.subplots_adjust(hspace=0.50, wspace=0.38)

hist_colors = [C_BLUE, C_BLUE, C_GREEN, C_PURPLE, C_ORANGE]
for ax, col, color in zip(axes.flat, NUM_COLS, hist_colors):
    s = df[col].dropna()
    ax.hist(s, bins=45, color=color, edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.axvline(s.mean(),   color=C_RED,    linestyle="--",
               linewidth=1.8, label=f"Mean   = {s.mean():.2f}")
    ax.axvline(s.median(), color=C_ORANGE, linestyle="--",
               linewidth=1.8, label=f"Median = {s.median():.2f}")
    skew = stats.skew(s)
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title(f"{col}  (n={len(s)})\nskewness = {skew:+.2f}",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5)

axes.flat[5].set_visible(False)

fig.suptitle(
    "Histograms — All Numerical Columns\n"
    "Dashed lines: mean (red) vs median (orange) — gap reveals skew direction",
    fontsize=12, fontweight="bold",
)
save(fig, "01_histograms.png")

# ── Plot 2: Box plots — IQR outlier detection ─────────────
# Each box spans Q1 to Q3 (IQR); dots beyond whiskers are
# flagged by Tukey fences as candidate outliers.
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
fig.subplots_adjust(wspace=0.48)

for ax, col, color in zip(axes, NUM_COLS, hist_colors):
    s = df[col].dropna()
    ax.boxplot(
        s,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.35, linewidth=1.5),
        medianprops=dict(color=C_RED, linewidth=2.2),
        flierprops=dict(marker="o", markerfacecolor=C_RED,
                        markeredgecolor="#8B0000", markersize=3.5, alpha=0.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    n_out = iqr_outlier_mask(s).sum()
    ax.set_title(
        f"{col}\n{n_out} outliers ({n_out/len(s)*100:.1f}%)",
        fontsize=8.5, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_ylabel(col, fontsize=8)

fig.suptitle(
    "Box Plots — IQR Outlier Detection\n"
    "Dots beyond whiskers are flagged by Tukey fences (Q1/Q3 ± 1.5 × IQR)",
    fontsize=11, fontweight="bold",
)
save(fig, "02_boxplots.png")


# ════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATIONS
# ════════════════════════════════════════════════════════════
print("\n─── SECTION 4: CORRELATIONS ────────────────────────────\n")

corr_pearson  = df[NUM_COLS].corr(method="pearson")
corr_spearman = df[NUM_COLS].corr(method="spearman")

print("Pearson correlation matrix (linear relationships):")
print(corr_pearson.round(3).to_string())

print("\nSpearman correlation matrix (rank-based, robust to outliers):")
print(corr_spearman.round(3).to_string())

# ── Plot 3: Pearson correlation heatmap ───────────────────
# df.corr() gives all pairwise correlations; a heatmap makes
# the pattern of strong/weak relationships immediately visible.
fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, mat, title in [
    (ax_p, corr_pearson,  "Pearson  r  (linear)"),
    (ax_s, corr_spearman, "Spearman ρ  (rank-based)"),
]:
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5, mask=mask, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

fig.suptitle(
    "Correlation Matrices — Numerical Variables\n"
    "Pearson captures linear relationships; Spearman is more robust to outliers",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
save(fig, "03_correlation_heatmaps.png")

# ── Plot 4: Scatter pairs ─────────────────────────────────
# Each scatter plot shows one pair of variables coloured by sector.
# Raw values are used — the clustering near zero for revenue/employees
# reflects the true right-skewed nature of the data.
pairs = [
    ("employees",    "revenue_mUSD",
     "Employees vs Revenue"),
    ("debt_ratio",   "profit_margin_pct",
     "Debt Ratio vs Profit Margin"),
    ("revenue_mUSD", "rd_spending_mUSD",
     "Revenue vs R&D Spending"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sector_cats = df["sector"].cat.categories
sector_cmap = plt.cm.Set1(np.linspace(0, 0.85, len(sector_cats)))

for ax, (cx, cy, title) in zip(axes, pairs):
    sub = df[[cx, cy, "sector"]].dropna()
    for sec, color in zip(sector_cats, sector_cmap):
        m = sub["sector"] == sec
        ax.scatter(sub.loc[m, cx], sub.loc[m, cy],
                   alpha=0.45, s=18, color=color, label=str(sec))

    # Pearson r between the raw columns
    r_val = sub[cx].corr(sub[cy])
    ax.set_xlabel(cx, fontsize=9)
    ax.set_ylabel(cy, fontsize=9)
    ax.set_title(f"{title}\nPearson r = {r_val:.3f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, ncol=1, loc="upper left",
              framealpha=0.8, handletextpad=0.3)

fig.suptitle(
    "Key Variable Pairs — Coloured by Sector",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
save(fig, "04_scatter_pairs.png")

# ── Plot 5: Sector profiles ───────────────────────────────
# Descriptive statistics applied to sub-groups: median of key
# metrics per sector reveals between-group differences.
sector_stats = (
    df.groupby("sector", observed=True)[
        ["revenue_mUSD", "profit_margin_pct", "debt_ratio"]
    ]
    .median()
    .sort_values("revenue_mUSD", ascending=False)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.subplots_adjust(wspace=0.38)
sector_colors = plt.cm.Set2(np.linspace(0, 1, len(sector_stats)))

for ax, (col, ylabel, color) in zip(axes, [
    ("revenue_mUSD",      "Median Revenue ($M)",      C_BLUE),
    ("profit_margin_pct", "Median Profit Margin (%)", C_GREEN),
    ("debt_ratio",        "Median Debt Ratio",         C_ORANGE),
]):
    ax.bar(sector_stats.index, sector_stats[col],
           color=sector_colors, edgecolor="white", linewidth=0.5)
    overall = df[col].median()
    ax.axhline(overall, color=C_RED, linestyle="--", linewidth=1.5,
               label=f"Overall median = {overall:.2f}")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(ylabel, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

fig.suptitle(
    "Descriptive Statistics by Sector — Median Revenue, Profit Margin, and Debt Ratio",
    fontsize=11, fontweight="bold",
)
save(fig, "05_sector_profiles.png")


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
rev_out   = iqr_outlier_mask(df["revenue_mUSD"].dropna()).sum()
r_emp_rev = df[["employees", "revenue_mUSD"]].corr().iloc[0, 1]
r_deb_pro = df[["debt_ratio", "profit_margin_pct"]].corr().iloc[0, 1]
r_rev_rd  = df[["revenue_mUSD", "rd_spending_mUSD"]].corr().iloc[0, 1]

print("\n" + "=" * 65)
print("KEY EDA FINDINGS — CORPORATE FINANCIAL DATASET")
print("=" * 65)
print(f"  Shape                     : {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  Missing R&D               : {df['rd_spending_mUSD'].isna().sum()} firms "
      f"({df['rd_spending_mUSD'].isna().mean()*100:.1f}%)")
print(f"  Revenue skewness          : {stats.skew(df['revenue_mUSD']):.2f}  (strong right skew)")
print(f"  Revenue IQR outliers      : {rev_out} firms beyond Tukey upper fence")
print(f"  Revenue mean / median     : {df['revenue_mUSD'].mean():.0f} / "
      f"{df['revenue_mUSD'].median():.0f} $M")
print(f"  Employees ↔ Revenue  r    : {r_emp_rev:.3f}  (strong positive)")
print(f"  Debt ↔ Profit margin r    : {r_deb_pro:.3f}  (moderate negative)")
print(f"  Revenue ↔ R&D        r    : {r_rev_rd:.3f}  (strong positive)")
print(f"  Profit margin skewness    : {stats.skew(df['profit_margin_pct']):.2f}  (approximately symmetric)")
print("=" * 65)
print(f"\nEDA complete. 5 plots saved to: {PLOTS_DIR}")
