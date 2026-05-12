# ============================================================
# cleaning.py — Data Cleaning & Transformation
# ============================================================
# Exercises accompanying cleaning.md.
#  Part I   — String Cleaning (capitalisation, symbols, whitespace, duplicates, joins)
#  Part II  — Missing Data (diagnosis, deletion, imputation)
#  Part III — Feature Engineering (encoding, scaling, correlation)
#  Part IV  — Dimensionality Reduction (PCA, t-SNE, UMAP)
# All plots are saved to plots/ next to this script.
# Run: python cleaning.py
# Requirements: pip install pandas numpy matplotlib seaborn scikit-learn missingno umap-learn
# ============================================================

import re
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import missingno as msno
    MSNO_AVAILABLE = True
except ImportError:
    MSNO_AVAILABLE = False
    print("  [info] missingno not installed — skipping visualisation step.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("  [info] umap-learn not installed — UMAP panel will be skipped.")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_DIR  = Path(__file__).parent
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Helpers ────────────────────────────────────────────────
def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")

def report(label, value):
    print(f"  {label:<45} {value}")

def justify(text):
    """Print a short rationale block."""
    print()
    for line in textwrap.wrap(text, width=60):
        print(f"  > {line}")
    print()

def save_missing_matrix(df, title, filename):
    """Save a missingno matrix plot for *df* to DATA_DIR/<filename>."""
    if not MSNO_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    msno.matrix(df, ax=ax, sparkline=False, color=(0.16, 0.50, 0.73))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualisation saved → {filename}")

# ════════════════════════════════════════════════════════════
# LOAD RAW FILES
# ════════════════════════════════════════════════════════════
section("LOAD RAW FILES")

personal = pd.read_csv(DATA_DIR / "employees_personal_messy.csv", dtype=str)
work     = pd.read_csv(DATA_DIR / "employees_work_messy.csv",     dtype=str)
perf     = pd.read_csv(DATA_DIR / "employees_performance_messy.csv", dtype=str)

report("personal  shape", str(personal.shape))
report("work      shape", str(work.shape))
report("perf      shape", str(perf.shape))

# ════════════════════════════════════════════════════════════
# CLEAN TABLE 1 — Personal Info
# ════════════════════════════════════════════════════════════
section("CLEAN TABLE 1: employees_personal_messy.csv")

# ── Step 1: Lowercase ─────────────────────────────────────
personal["name"]    = personal["name"].str.lower()
personal["city"]    = personal["city"].str.lower()
personal["country"] = personal["country"].str.lower()
personal["email"]   = personal["email"].str.lower()
print("  [1] All text columns lowercased.")

# ── Step 2: Strip whitespace ──────────────────────────────
for col in ["name", "email", "city", "country", "employee_id"]:
    personal[col] = personal[col].str.strip()
    # Collapse internal double-spaces
    personal[col] = personal[col].str.replace(r"\s{2,}", " ", regex=True)
print("  [2] Leading/trailing/internal whitespace stripped.")

# ── Step 3: Remove symbols from name ─────────────────────
# Keep letters, digits, spaces, hyphens, apostrophes
personal["name"] = personal["name"].apply(
    lambda s: re.sub(r"[^a-z0-9 \-\']", "", s).strip() if pd.notna(s) else s
)
personal["name"] = personal["name"].str.replace(r"\s{2,}", " ", regex=True)
print("  [3] Stray symbols removed from 'name'.")

# ── Step 4: Remove full duplicates ────────────────────────
before = len(personal)
personal = personal.drop_duplicates()
report("[4] Full duplicates removed:", f"{before - len(personal)} rows dropped "
       f"({before} → {len(personal)})")

# ── Step 5: Resolve partial duplicates ───────────────────
# Partial dups: same employee_id, different city/country.
# Strategy: keep first occurrence (assumes later rows are noise).
before = len(personal)
personal = personal.drop_duplicates(subset=["employee_id"], keep="first")
report("[5] Partial duplicates resolved (keep first):",
       f"{before - len(personal)} rows dropped ({before} → {len(personal)})")

print(f"\n  personal final shape: {personal.shape}")

# ════════════════════════════════════════════════════════════
# CLEAN TABLE 2 — Work Info
# ════════════════════════════════════════════════════════════
section("CLEAN TABLE 2: employees_work_messy.csv")

# ── Step 1: Lowercase department ─────────────────────────
work["department"] = work["department"].str.lower().str.strip()
print("  [1] 'department' lowercased and stripped.")

# ── Step 2: Standardise missing values ───────────────────
MISSING_STRINGS = {
    "n/a", "na", "null", "none", "nan", "#n/a",
    "not available", "?", "unknown", "",
}

def to_nan(val):
    if pd.isna(val):
        return np.nan
    return np.nan if str(val).strip().lower() in MISSING_STRINGS else val

work["salary"]    = work["salary"].apply(to_nan)
work["hire_date"] = work["hire_date"].apply(to_nan)

report("[2] Standardised missing sentinels to NaN (salary):",
       f"{work['salary'].isna().sum()} missing")
report("    Standardised missing sentinels to NaN (hire_date):",
       f"{work['hire_date'].isna().sum()} missing")

# ── Step 3: Clean salary — remove $ and , then cast ──────
work["salary"] = (
    work["salary"]
    .str.replace(r"[\$,]", "", regex=True)
    .str.strip()
)
work["salary"] = pd.to_numeric(work["salary"], errors="coerce")
report("[3] Salary cleaned and cast to numeric. NaN count:",
       str(work["salary"].isna().sum()))

# ── Missing data visualisation — BEFORE imputation ───────
section("MISSING DATA VISUALISATION — work table (BEFORE imputation)")
save_missing_matrix(
    work,
    "Missing data matrix — work table (before SimpleImputer)",
    "missing_work_before.png",
)

# ── Step 4: Impute missing salary with SimpleImputer ─────
before_miss = work["salary"].isna().sum()
si = SimpleImputer(strategy="median")
work["salary"] = si.fit_transform(work[["salary"]]).ravel()
work["salary"] = work["salary"].astype(int)
report("[4] Missing salary imputed with SimpleImputer (median):",
       f"{before_miss} values filled")

# ── Missing data visualisation — AFTER imputation ────────
section("MISSING DATA VISUALISATION — work table (AFTER imputation)")
save_missing_matrix(
    work,
    "Missing data matrix — work table (after SimpleImputer)",
    "missing_work_after.png",
)

# ── Step 5: Drop rows where hire_date is still missing ────
before = len(work)
work = work.dropna(subset=["hire_date"])
report("[5] Rows with missing hire_date dropped:",
       f"{before - len(work)} rows removed ({before} → {len(work)})")

print(f"\n  work final shape: {work.shape}")

# ════════════════════════════════════════════════════════════
# CLEAN TABLE 3 — Performance Info
# ════════════════════════════════════════════════════════════
section("CLEAN TABLE 3: employees_performance_messy.csv")

# ── Step 1: Standardise missing values ───────────────────
perf["performance_score"] = perf["performance_score"].apply(to_nan)
perf["satisfaction"]      = perf["satisfaction"].apply(to_nan)

# Sentinel -999 for performance_score
perf["performance_score"] = pd.to_numeric(perf["performance_score"], errors="coerce")
perf.loc[perf["performance_score"] == -999, "performance_score"] = np.nan

# Sentinel -1 for satisfaction
perf["satisfaction"] = pd.to_numeric(perf["satisfaction"], errors="coerce")
perf.loc[perf["satisfaction"] == -1, "satisfaction"] = np.nan

report("[1] Standardised missing (including -999 / -1 sentinels):",
       f"perf_score NaN: {perf['performance_score'].isna().sum()}, "
       f"satisfaction NaN: {perf['satisfaction'].isna().sum()}")

# ── Step 2: Remove full duplicates ────────────────────────
before = len(perf)
perf = perf.drop_duplicates()
report("[2] Full duplicates removed:", f"{before - len(perf)} rows dropped")

# ── Step 3: Resolve partial duplicates ───────────────────
# Same employee_id, conflicting scores. Keep row with more complete data.
perf["_complete"] = perf[["performance_score", "satisfaction"]].notna().sum(axis=1)
perf = (perf
        .sort_values("_complete", ascending=False)
        .drop_duplicates(subset=["employee_id"], keep="first")
        .drop(columns="_complete"))
report("[3] Partial duplicates resolved (keep most complete):",
       f"{len(perf)} rows remaining")

# ── Missing data visualisation — BEFORE imputation ───────
section("MISSING DATA VISUALISATION — perf table (BEFORE imputation)")
save_missing_matrix(
    perf,
    "Missing data matrix — perf table (before IterativeImputer)",
    "missing_perf_before.png",
)

# ── Step 4: Impute missing scores with IterativeImputer ──
# IterativeImputer models each feature as a function of the others,
# effectively performing multiple imputation via iterated regression.
p_miss = perf["performance_score"].isna().sum()
s_miss = perf["satisfaction"].isna().sum()

ii = IterativeImputer(random_state=42, max_iter=10)
imputed = ii.fit_transform(perf[["performance_score", "satisfaction"]])
perf["performance_score"] = np.round(imputed[:, 0], 1)
perf["satisfaction"]      = np.round(imputed[:, 1]).astype(int)

report("[4] Missing performance_score imputed with IterativeImputer:",
       f"{p_miss} values filled")
report("    Missing satisfaction imputed with IterativeImputer:",
       f"{s_miss} values filled")

# ── Missing data visualisation — AFTER imputation ────────
section("MISSING DATA VISUALISATION — perf table (AFTER imputation)")
save_missing_matrix(
    perf,
    "Missing data matrix — perf table (after IterativeImputer)",
    "missing_perf_after.png",
)

print(f"\n  perf final shape: {perf.shape}")

# ════════════════════════════════════════════════════════════
# MERGE — personal, work, perf on employee_id
# ════════════════════════════════════════════════════════════
section("MERGE TABLES")

for df in [personal, work, perf]:
    df["employee_id"] = df["employee_id"].str.strip()

merged = personal.merge(work, on="employee_id", how="left").merge(perf, on="employee_id", how="left")
report("Merged shape:", f"{merged.shape}")

# ── Save ──────────────────────────────────────────────────
out_path = DATA_DIR / "employees_recovered.csv"
merged.to_csv(out_path, index=False)
print(f"\n  Recovered dataset saved → employees_recovered.csv")


# ════════════════════════════════════════════════════════════
# LOAD EXTENDED DATASET & MERGE
# ════════════════════════════════════════════════════════════
section("LOAD EXTENDED DATASET & MERGE")

extended = pd.read_csv(DATA_DIR / "employees_extended.csv")
df = merged.merge(extended, on="employee_id", how="inner")
report("Shape after merging with employees_extended:", str(df.shape))


# ════════════════════════════════════════════════════════════
# ENCODINGS
# ════════════════════════════════════════════════════════════

# ── One-hot encoding: work_mode, contract_type ────────────
section("ONE-HOT ENCODING  —  work_mode, contract_type")
justify(
    "Both columns are nominal: their categories carry no "
    "natural rank or distance. Representing them as integers "
    "would falsely imply an ordering (e.g. remote=0 < "
    "hybrid=1 < onsite=2). One-hot encoding creates a binary "
    "indicator per category, making the representation "
    "order-free and compatible with linear models."
)

df = pd.get_dummies(df, columns=["work_mode", "contract_type"], dtype=int)
ohe_cols = [c for c in df.columns
            if c.startswith("work_mode_") or c.startswith("contract_type_")]
print(f"  New columns: {ohe_cols}")


# ── Ordinal encoding: education_level, seniority ─────────
section("ORDINAL ENCODING  —  education_level, seniority")
justify(
    "Both columns have a clear, meaningful rank order "
    "(junior < mid < senior < lead; high_school < bachelor "
    "< master < phd). Assigning consecutive integers "
    "preserves that order without inflating the feature "
    "space as one-hot would. Random label assignment "
    "would distort distance-based algorithms."
)

edu_order = ["high_school", "bachelor", "master", "phd"]
sen_order = ["junior", "mid", "senior", "lead"]

oe = OrdinalEncoder(categories=[edu_order, sen_order])
df[["education_level_enc", "seniority_enc"]] = oe.fit_transform(
    df[["education_level", "seniority"]]
).astype(int)

print(f"  education_level mapping: { {v: i for i, v in enumerate(edu_order)} }")
print(f"  seniority mapping:       { {v: i for i, v in enumerate(sen_order)} }")


# ════════════════════════════════════════════════════════════
# SCALING
# ════════════════════════════════════════════════════════════

# ── Standardisation: years_experience, commute_distance_km
section("STANDARDISATION (Z-score)  —  years_experience, commute_distance_km")
justify(
    "Both columns are continuous and approximately normally "
    "distributed with no hard natural bounds. Z-score "
    "standardisation (mean=0, std=1) is the right choice "
    "here: it centres the distribution and makes the scale "
    "comparable across features without compressing the "
    "shape or clipping any values."
)

ss = StandardScaler()
df[["years_experience_std", "commute_distance_km_std"]] = ss.fit_transform(
    df[["years_experience", "commute_distance_km"]]
)
print(f"  years_experience     — mean: {df['years_experience_std'].mean():.4f}, "
      f"std: {df['years_experience_std'].std():.4f}")
print(f"  commute_distance_km  — mean: {df['commute_distance_km_std'].mean():.4f}, "
      f"std: {df['commute_distance_km_std'].std():.4f}")


# ── Min-max scaling: training_score, engagement_score ─────
section("MIN-MAX SCALING  —  training_score, engagement_score")
justify(
    "training_score is bounded to [0, 100] and "
    "engagement_score to [1, 10]. When the natural range "
    "is fixed and meaningful, min-max scaling maps values "
    "to [0, 1] while fully preserving the original "
    "proportional distances. Z-score would push values "
    "outside the original range, losing the "
    "interpretability of the boundary."
)

mm = MinMaxScaler()
df[["training_score_mm", "engagement_score_mm"]] = mm.fit_transform(
    df[["training_score", "engagement_score"]]
)
print(f"  training_score   — min: {df['training_score_mm'].min():.3f}, "
      f"max: {df['training_score_mm'].max():.3f}")
print(f"  engagement_score — min: {df['engagement_score_mm'].min():.3f}, "
      f"max: {df['engagement_score_mm'].max():.3f}")


# ── Robust scaling: annual_bonus, overtime_hours_year ─────
section("ROBUST SCALING (IQR)  —  annual_bonus, overtime_hours_year")
justify(
    "Both columns are right-skewed with extreme "
    "outliers (~5 % of rows). Z-score and min-max are both "
    "sensitive to outliers: a single extreme value pulls "
    "the mean or stretches the range, compressing the bulk "
    "of the data into a tiny interval. RobustScaler uses "
    "the median and IQR instead, so the central mass of "
    "the distribution is scaled correctly regardless of "
    "the tails."
)

rb = RobustScaler()
df[["annual_bonus_robust", "overtime_hours_year_robust"]] = rb.fit_transform(
    df[["annual_bonus", "overtime_hours_year"]]
)
print(f"  annual_bonus         — median: {df['annual_bonus_robust'].median():.3f}, "
      f"IQR: {df['annual_bonus_robust'].quantile(0.75) - df['annual_bonus_robust'].quantile(0.25):.3f}")
print(f"  overtime_hours_year  — median: {df['overtime_hours_year_robust'].median():.3f}, "
      f"IQR: {df['overtime_hours_year_robust'].quantile(0.75) - df['overtime_hours_year_robust'].quantile(0.25):.3f}")


# ════════════════════════════════════════════════════════════
# CORRELATION MATRICES
# ════════════════════════════════════════════════════════════
section("CORRELATION MATRICES")

CORR_COLS = [
    "salary", "performance_score", "satisfaction",
    "education_level_enc", "seniority_enc",
    "years_experience_std", "commute_distance_km_std",
    "training_score_mm", "engagement_score_mm",
    "annual_bonus_robust", "overtime_hours_year_robust",
    "skill_analytical", "skill_communication", "skill_technical",
    "skill_leadership", "skill_creativity", "skill_teamwork",
]

PRETTY_LABELS = {
    "salary":                     "Salary",
    "performance_score":          "Perf. Score",
    "satisfaction":               "Satisfaction",
    "education_level_enc":        "Education",
    "seniority_enc":              "Seniority",
    "years_experience_std":       "Yrs Exp (std)",
    "commute_distance_km_std":    "Commute (std)",
    "training_score_mm":          "Training (mm)",
    "engagement_score_mm":        "Engagement (mm)",
    "annual_bonus_robust":        "Bonus (rob.)",
    "overtime_hours_year_robust": "Overtime (rob.)",
    "skill_analytical":           "Skill: Analytical",
    "skill_communication":        "Skill: Comm.",
    "skill_technical":            "Skill: Technical",
    "skill_leadership":           "Skill: Leadership",
    "skill_creativity":           "Skill: Creativity",
    "skill_teamwork":             "Skill: Teamwork",
}

corr_df = df[CORR_COLS].rename(columns=PRETTY_LABELS)

def save_corr_heatmap(method, filename):
    mat = corr_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        mat, ax=ax, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.4, annot_kws={"size": 7},
    )
    ax.set_title(f"{method.capitalize()} Correlation Matrix",
                 fontsize=13, pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filename}")

justify(
    "Three correlation coefficients are computed to cross-check "
    "relationships from different angles. Pearson measures "
    "linear association and is sensitive to outliers. "
    "Spearman is rank-based and captures monotonic (not just "
    "linear) relationships, making it robust to outliers and "
    "skewed distributions. Kendall is also rank-based but "
    "uses concordant/discordant pairs, giving a more "
    "conservative estimate that handles ties and small "
    "samples better."
)

save_corr_heatmap("pearson",  "corr_pearson.png")
save_corr_heatmap("spearman", "corr_spearman.png")
save_corr_heatmap("kendall",  "corr_kendall.png")


# ════════════════════════════════════════════════════════════
#  DIMENSIONALITY REDUCTION
# ════════════════════════════════════════════════════════════
section("ADDITIONALS — DIMENSIONALITY REDUCTION")

# The 6 skill columns seem to share a common latent factor, producing
# high inter-correlation. Reducing them to 2 dimensions lets us
# visualise structure and check whether seniority separates
# naturally in the latent skill space.
SKILL_COLS = [
    "skill_analytical", "skill_communication", "skill_technical",
    "skill_leadership", "skill_creativity", "skill_teamwork",
]

skill_scaled = StandardScaler().fit_transform(df[SKILL_COLS])

# Map archetype string → integer for colouring
ARCHETYPE_ORDER  = ["technical", "leader", "creative", "operational"]
ARCHETYPE_COLORS = ["#2171b5", "#cb181d", "#238b45", "#d94801"]
colour_values    = df["archetype"].map({a: i for i, a in enumerate(ARCHETYPE_ORDER)}).values

# ── PCA ───────────────────────────────────────────────────
justify(
    "PCA (Principal Component Analysis) is a linear method "
    "that finds the directions of maximum variance. It is "
    "fast, deterministic, and the explained variance ratio "
    "on each axis is interpretable. It works best when the "
    "structure in the data is approximately linear."
)

pca        = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(skill_scaled)
print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")

# ── t-SNE ─────────────────────────────────────────────────
justify(
    "t-SNE (t-distributed Stochastic Neighbour Embedding) "
    "is a non-linear method optimised for 2D visualisation. "
    "It preserves local neighbourhood structure, making "
    "clusters visually apparent even when the global "
    "geometry is non-linear. Axes are not interpretable "
    "on their own — only relative distances matter."
)

tsne        = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_coords = tsne.fit_transform(skill_scaled)

# ── UMAP ──────────────────────────────────────────────────
if UMAP_AVAILABLE:
    justify(
        "UMAP (Uniform Manifold Approximation and Projection) "
        "is also non-linear but preserves both local and more "
        "of the global structure than t-SNE. It is faster on "
        "large datasets and produces embeddings that are more "
        "consistent across runs (given a fixed random seed)."
    )
    reducer     = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(skill_scaled)

# ── Combined plot ─────────────────────────────────────────
n_panels = 3 if UMAP_AVAILABLE else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
fig.suptitle(
    "Dimensionality reduction of skill battery (6 features → 2D)\n"
    "Coloured by employee archetype",
    fontsize=12,
)

panels = [
    ("PCA",   pca_coords,  f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
                           f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"),
    ("t-SNE", tsne_coords, "Dim 1", "Dim 2"),
]
if UMAP_AVAILABLE:
    panels.append(("UMAP", umap_coords, "Dim 1", "Dim 2"))

for ax, (title, coords, xlabel, ylabel) in zip(axes, panels):
    for i, (archetype, colour) in enumerate(zip(ARCHETYPE_ORDER, ARCHETYPE_COLORS)):
        mask = colour_values == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, label=archetype,
                   s=40, alpha=0.75, edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

axes[0].legend(title="Archetype", fontsize=8, title_fontsize=8)

fig.tight_layout()
fig.savefig(PLOTS_DIR / "dimensionality_reduction.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved → dimensionality_reduction.png")

print("\n" + "=" * 65)
print("  cleaning.py finished successfully.")
print("=" * 65)

