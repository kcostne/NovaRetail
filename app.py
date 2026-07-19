"""
================================================================================
 CUSTOMER TRANSACTION ANALYTICS & GROWTH OPPORTUNITY DASHBOARD (app.py)
================================================================================
Author  : Senior Business Analytics Consultant / Python Data Scientist (Claude)
Purpose : End-to-end analysis of a retail customer-transaction dataset to
          surface customer performance insights and business growth
          opportunities. Written for a business-analytics audience — code is
          modular, heavily commented, and organized into the six stages
          requested in the project brief:

              Stage 1 - Data Understanding
              Stage 2 - Data Preparation
              Stage 3 - Customer Performance Analysis
              Stage 4 - Growth Opportunity Analysis
              Stage 5 - Advanced Analytics (RFM, CLV, segmentation, trends)
              Stage 6 - Business Recommendations

How to run:
    python app.py                      # uses NR_dataset.xlsx in this folder
    python app.py --data path/to.xlsx  # or point at a different file

Outputs:
    - Printed narrative + tables in the console (organized by stage)
    - All charts saved as .png files to ./outputs/plots/
    - A cleaned dataset written to ./outputs/cleaned_transactions.csv

NOTE ON SCOPE: This script is written as a self-contained analysis engine
(suitable for a GitHub repo). Its functions are already split so that a
Streamlit front end (streamlit_app.py) can import them later and call, e.g.,
`load_data()` / `run_full_pipeline()` and render the returned DataFrames and
matplotlib figures with st.pyplot()/st.dataframe() without any rework.
================================================================================
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so the script can run headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# GLOBAL CONFIG
# --------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

DEFAULT_DATA_PATH = "NR_dataset.xlsx"
OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "plots"


def _ensure_output_dirs() -> None:
    """Create the output folders if they don't already exist."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the plots folder and close it to free memory."""
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _section(title: str) -> None:
    """Print a formatted section header so console output reads like a report."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# STAGE 1 — DATA UNDERSTANDING
# ============================================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from Excel (falls back to CSV if a .csv path
    is given). Raises a clear error if the file cannot be found, since a
    silent failure here would break every downstream stage.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. Place NR_dataset.xlsx next "
            f"to app.py, or pass --data <path_to_file>."
        )

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Stage 1 deliverables: structure, dtypes, missing values, duplicates,
    and summary statistics for numerical and categorical variables.
    """
    _section("STAGE 1 — DATA UNDERSTANDING")

    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("--- Column data types ---")
    print(df.dtypes)

    print("\n--- Missing values per column ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing)

    print(f"\n--- Duplicate rows (exact duplicates) ---\n{df.duplicated().sum()}")

    print("\n--- Numerical variable summary ---")
    print(df.describe(include=[np.number]).round(2))

    print("\n--- Categorical variable summary ---")
    print(df.describe(include=["object", "string", "category"]))

    # Assumption worth flagging: TransactionID is not unique per row (many
    # rows share the same TransactionID). We treat each ROW as one purchase
    # line-item, not each TransactionID as one order, since PurchaseAmount,
    # ProductCategory, and CustomerID vary within a shared TransactionID.
    repeated_tx = df["TransactionID"].value_counts()
    print(
        "\nNote: TransactionID is shared across multiple rows for "
        f"{(repeated_tx > 1).sum()} of {repeated_tx.shape[0]} unique IDs. "
        "Each row is treated as an individual purchase line-item throughout "
        "this analysis (see Stage 2 assumptions)."
    )


# ============================================================================
# STAGE 2 — DATA PREPARATION
# ============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing/inconsistent data and engineer date-derived features.
    Every step is commented so the reasoning is auditable.
    """
    _section("STAGE 2 — DATA PREPARATION")
    df = df.copy()

    # 1) 'label' (customer behavioral segment) has missing values in this
    #    dataset. Since it is a categorical business label (not something we
    #    can safely infer numerically), we fill missing values with the
    #    explicit placeholder "Unlabeled" rather than dropping the row —
    #    dropping would also discard a real transaction from every other
    #    analysis. This keeps the row usable everywhere except label-specific
    #    segment breakdowns, where "Unlabeled" will show up as its own group.
    missing_labels = df["label"].isnull().sum()
    df["label"] = df["label"].fillna("Unlabeled")
    print(f"Filled {missing_labels} missing 'label' value(s) with 'Unlabeled'.")

    # 2) Convert TransactionDate to a proper datetime dtype (it may already
    #    be parsed correctly by pandas, but we enforce it explicitly and
    #    coerce unparseable values to NaT rather than letting them fail
    #    the whole load).
    before_dtype = df["TransactionDate"].dtype
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    bad_dates = df["TransactionDate"].isnull().sum()
    print(f"TransactionDate dtype converted from {before_dtype} to datetime64. "
          f"{bad_dates} unparseable date(s) coerced to NaT.")
    if bad_dates > 0:
        df = df.dropna(subset=["TransactionDate"])
        print(f"Dropped {bad_dates} row(s) with an invalid TransactionDate.")

    # 3) Engineer time-based features used throughout Stages 3-5.
    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionMonthName"] = df["TransactionDate"].dt.month_name()
    df["TransactionQuarter"] = df["TransactionDate"].dt.quarter
    df["TransactionWeekday"] = df["TransactionDate"].dt.day_name()
    print("Engineered columns: TransactionYear, TransactionMonth, "
          "TransactionMonthName, TransactionQuarter, TransactionWeekday.")

    # 4) Sanity-check PurchaseAmount: negative or zero purchase amounts would
    #    be data-entry errors for a retail transaction. None were found in
    #    this dataset, but we guard against them defensively.
    invalid_amounts = df[df["PurchaseAmount"] <= 0]
    if not invalid_amounts.empty:
        print(f"Removing {len(invalid_amounts)} row(s) with PurchaseAmount <= 0.")
        df = df[df["PurchaseAmount"] > 0]
    else:
        print("PurchaseAmount check: no zero/negative values found.")

    # 5) Whitespace/casing hygiene on text columns (defensive cleanup — cheap
    #    insurance against the kind of inconsistency we address explicitly
    #    for ProductCategory below).
    text_cols = ["ProductCategory", "CustomerAgeGroup", "CustomerGender",
                 "CustomerRegion", "RetailChannel", "label"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
    print("Trimmed leading/trailing whitespace on all categorical text columns.")

    return df


def standardize_product_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2 (ProductCategory standardization).

    RECOMMENDATIONS (presented to the analyst before this code was written —
    see the accompanying write-up for full reasoning per group). Only the
    HIGH-confidence groups are merged automatically below; MEDIUM and LOW
    confidence groups are deliberately left untouched so a human reviewer
    can approve them explicitly, per the brief's requirement not to
    auto-merge ambiguous categories.

    Approved (HIGH confidence) merges:
        - "Beauty Products"                         -> "Beauty & Personal Care"
        - "Clothing", "Fashion"                      -> "Fashion & Apparel"
        - "Grocery", "Grocery Items"                 -> "Groceries"
        - "Furniture"                                -> "Furniture & Decor"
        - "Sports Equipment"                         -> "Sporting Goods"
        - "Toys"                                     -> "Toys & Games"

    Flagged but NOT merged (Medium/Low confidence — left as separate
    categories pending business review):
        - Cosmetics vs. Beauty & Personal Care (Medium)
        - Health & Beauty vs. Beauty & Personal Care (Low)
        - Books vs. Books & Magazines (Medium)
        - Fashion Accessories vs. Fashion & Apparel (Low)
        - Children's Clothing vs. Fashion & Apparel (Low)
        - Sportswear vs. Fashion & Apparel / Sporting Goods (Low)
        - Food & Beverages vs. Groceries (Medium)
        - Home Decor vs. Furniture & Decor (Medium)
        - Home & Garden, Home Improvement, Home Appliances, Gardening Tools
          (Low — distinct sub-departments despite the shared "Home" theme)
        - Sports & Outdoors, Outdoor Equipment vs. Sporting Goods (Medium)
        - Health & Wellness vs. Health Supplements (Medium)
    """
    _section("STAGE 2 (continued) — ProductCategory Standardization")
    df = df.copy()

    approved_mapping = {
        "Beauty Products": "Beauty & Personal Care",
        "Clothing": "Fashion & Apparel",
        "Fashion": "Fashion & Apparel",
        "Grocery": "Groceries",
        "Grocery Items": "Groceries",
        "Furniture": "Furniture & Decor",
        "Sports Equipment": "Sporting Goods",
        "Toys": "Toys & Games",
    }

    before_unique = sorted(df["ProductCategory"].unique())
    print(f"Unique ProductCategory values BEFORE standardization "
          f"({len(before_unique)}):")
    print(before_unique)

    changed_mask = df["ProductCategory"].isin(approved_mapping.keys())
    n_changed = int(changed_mask.sum())
    df["ProductCategory"] = df["ProductCategory"].replace(approved_mapping)

    after_unique = sorted(df["ProductCategory"].unique())
    print(f"\nUnique ProductCategory values AFTER standardization "
          f"({len(after_unique)}):")
    print(after_unique)

    print(f"\n{n_changed} record(s) were reassigned to a standardized "
          f"category name out of {len(df)} total rows.")

    return df


# ============================================================================
# STAGE 3 — CUSTOMER PERFORMANCE ANALYSIS
# ============================================================================
def customer_performance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze customer performance across spend, frequency, category
    preference, satisfaction, channel, region, age, and gender. Produces
    one chart per dimension, saved to outputs/plots/.
    Returns the per-customer summary table for reuse in later stages.
    """
    _section("STAGE 3 — CUSTOMER PERFORMANCE ANALYSIS")

    # --- Per-customer roll-up: total spend, transaction count, avg value ---
    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            TotalSpend=("PurchaseAmount", "sum"),
            NumTransactions=("PurchaseAmount", "count"),
            AvgPurchaseValue=("PurchaseAmount", "mean"),
            AvgSatisfaction=("CustomerSatisfaction", "mean"),
        )
        .reset_index()
        .sort_values("TotalSpend", ascending=False)
    )
    print("\n--- Top 10 customers by total spend ---")
    print(customer_summary.head(10).round(2).to_string(index=False))

    # 1) Total spend distribution
    fig, ax = plt.subplots()
    sns.histplot(customer_summary["TotalSpend"], bins=15, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Total Customer Spend")
    ax.set_xlabel("Total Spend ($)")
    ax.set_ylabel("Number of Customers")
    _save_fig(fig, "01_total_spend_distribution")

    # 2) Number of transactions per customer
    fig, ax = plt.subplots()
    sns.histplot(customer_summary["NumTransactions"], bins=range(1, 11), ax=ax, color="#55A868")
    ax.set_title("Distribution of Transactions per Customer")
    ax.set_xlabel("Number of Transactions")
    ax.set_ylabel("Number of Customers")
    _save_fig(fig, "02_transactions_per_customer")

    # 3) Average purchase value distribution
    fig, ax = plt.subplots()
    sns.boxplot(x=customer_summary["AvgPurchaseValue"], ax=ax, color="#C44E52")
    ax.set_title("Average Purchase Value per Customer")
    ax.set_xlabel("Average Purchase Value ($)")
    _save_fig(fig, "03_avg_purchase_value")

    # 4) Product category preferences (overall revenue share)
    cat_revenue = (
        df.groupby("ProductCategory")["PurchaseAmount"]
        .sum()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=cat_revenue.values, y=cat_revenue.index, ax=ax, palette="viridis")
    ax.set_title("Total Revenue by Product Category")
    ax.set_xlabel("Total Revenue ($)")
    ax.set_ylabel("Product Category")
    _save_fig(fig, "04_revenue_by_category")
    print("\nInterpretation: the categories at the top of this chart drive "
          "the largest share of revenue and are priority categories to keep "
          "well-stocked and well-marketed.")

    # 5) Customer satisfaction distribution
    fig, ax = plt.subplots()
    sns.countplot(x="CustomerSatisfaction", data=df, ax=ax, palette="Blues_d",
                   order=sorted(df["CustomerSatisfaction"].unique()))
    ax.set_title("Customer Satisfaction Score Distribution")
    ax.set_xlabel("Satisfaction Score (1 = lowest, 5 = highest)")
    ax.set_ylabel("Number of Transactions")
    _save_fig(fig, "05_satisfaction_distribution")

    # 6) Retail channel usage
    channel_counts = df["RetailChannel"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(channel_counts.values, labels=channel_counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("pastel"))
    ax.set_title("Retail Channel Usage Share")
    _save_fig(fig, "06_channel_usage")

    # 7) Regional performance (revenue by region)
    region_revenue = df.groupby("CustomerRegion")["PurchaseAmount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=region_revenue.index, y=region_revenue.values, ax=ax, palette="mako")
    ax.set_title("Total Revenue by Customer Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Revenue ($)")
    _save_fig(fig, "07_regional_revenue")

    # 8) Age group and gender comparison (avg spend heatmap)
    age_gender_spend = df.pivot_table(
        index="CustomerAgeGroup", columns="CustomerGender",
        values="PurchaseAmount", aggfunc="mean"
    )
    fig, ax = plt.subplots()
    sns.heatmap(age_gender_spend, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Average Purchase Amount by Age Group & Gender")
    _save_fig(fig, "08_age_gender_avg_spend")

    print("\nAll Stage 3 charts saved to outputs/plots/ "
          "(01_total_spend_distribution.png ... 08_age_gender_avg_spend.png).")

    return customer_summary


# ============================================================================
# STAGE 4 — GROWTH OPPORTUNITY ANALYSIS
# ============================================================================
def growth_opportunity_analysis(df: pd.DataFrame, customer_summary: pd.DataFrame) -> dict:
    """
    Identify growth and retention opportunities:
      - High-value customers (top spend decile)
      - Customers with declining activity (based on the 'label' field and
        recency of last purchase)
      - Segments with high growth potential
      - Underperforming regions / categories / channels
      - Satisfaction issues that put retention at risk
    Returns a dict of key findings for use in the recommendations stage.
    """
    _section("STAGE 4 — GROWTH OPPORTUNITY ANALYSIS")
    findings = {}

    # --- High-value customers: top 20% by total spend ---
    threshold = customer_summary["TotalSpend"].quantile(0.80)
    high_value = customer_summary[customer_summary["TotalSpend"] >= threshold]
    findings["high_value_customers"] = high_value
    print(f"\nHigh-value customers (top 20% by spend, >= ${threshold:,.2f}): "
          f"{len(high_value)} customers, "
          f"representing ${high_value['TotalSpend'].sum():,.2f} "
          f"({high_value['TotalSpend'].sum() / customer_summary['TotalSpend'].sum():.1%} "
          f"of total revenue).")

    fig, ax = plt.subplots()
    sns.barplot(x="CustomerID", y="TotalSpend",
                data=high_value.sort_values("TotalSpend", ascending=False).astype({"CustomerID": str}),
                ax=ax, palette="rocket")
    ax.set_title("High-Value Customers (Top 20% by Spend)")
    ax.set_xlabel("Customer ID")
    ax.set_ylabel("Total Spend ($)")
    ax.tick_params(axis="x", rotation=90)
    _save_fig(fig, "09_high_value_customers")

    # --- Customers with declining activity: label == 'Decline' ---
    decline_customers = df[df["label"] == "Decline"]["CustomerID"].nunique()
    findings["declining_customers_count"] = decline_customers
    print(f"\nCustomers labeled 'Decline': {decline_customers} unique customers "
          f"— these are retention-risk accounts and prime targets for "
          f"win-back campaigns.")

    # --- Segments with high growth potential: label == 'Growth' or 'Promising' ---
    growth_segment = df[df["label"].isin(["Growth", "Promising"])]
    seg_revenue = growth_segment.groupby("label")["PurchaseAmount"].sum()
    findings["growth_segment_revenue"] = seg_revenue
    print(f"\nRevenue from 'Growth'/'Promising' segments:\n{seg_revenue.round(2)}")

    fig, ax = plt.subplots()
    label_summary = df.groupby("label")["PurchaseAmount"].agg(["sum", "count"]).rename(
        columns={"sum": "TotalRevenue", "count": "Transactions"})
    sns.barplot(x=label_summary.index, y=label_summary["TotalRevenue"], ax=ax, palette="crest")
    ax.set_title("Total Revenue by Customer Behavioral Segment")
    ax.set_xlabel("Segment (label)")
    ax.set_ylabel("Total Revenue ($)")
    _save_fig(fig, "10_revenue_by_segment")

    # --- Underperforming regions ---
    region_perf = df.groupby("CustomerRegion").agg(
        Revenue=("PurchaseAmount", "sum"), AvgSatisfaction=("CustomerSatisfaction", "mean")
    ).sort_values("Revenue")
    findings["underperforming_region"] = region_perf.index[0]
    print(f"\nRegional performance (lowest revenue first):\n{region_perf.round(2)}")
    print(f"Underperforming region: {region_perf.index[0]} "
          f"(${region_perf['Revenue'].iloc[0]:,.2f} in revenue).")

    # --- Underperforming product categories ---
    cat_perf = df.groupby("ProductCategory").agg(
        Revenue=("PurchaseAmount", "sum"), Transactions=("PurchaseAmount", "count")
    ).sort_values("Revenue")
    findings["underperforming_categories"] = cat_perf.head(5)
    print(f"\nLowest-revenue product categories:\n{cat_perf.head(5).round(2)}")

    # --- Retail channels with expansion opportunity ---
    channel_perf = df.groupby("RetailChannel").agg(
        Revenue=("PurchaseAmount", "sum"),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
        Transactions=("PurchaseAmount", "count"),
    )
    findings["channel_perf"] = channel_perf
    print(f"\nChannel performance:\n{channel_perf.round(2)}")

    # --- Satisfaction issues affecting retention ---
    low_satisfaction = df[df["CustomerSatisfaction"] <= 2]
    findings["low_satisfaction_share"] = len(low_satisfaction) / len(df)
    print(f"\nTransactions with satisfaction <= 2: {len(low_satisfaction)} "
          f"({len(low_satisfaction) / len(df):.1%} of all transactions). "
          f"These are retention-risk touchpoints.")

    fig, ax = plt.subplots()
    sns.boxplot(x="RetailChannel", y="CustomerSatisfaction", data=df, ax=ax, palette="Set2")
    ax.set_title("Customer Satisfaction by Retail Channel")
    _save_fig(fig, "11_satisfaction_by_channel")

    return findings


# ============================================================================
# STAGE 5 — ADVANCED ANALYTICS
# ============================================================================
def rfm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic Recency / Frequency / Monetary analysis at the customer level,
    plus 1-5 quintile scores and a combined RFM segment label.
    """
    _section("STAGE 5a — RFM ANALYSIS")

    snapshot_date = df["TransactionDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionID", "count"),
        Monetary=("PurchaseAmount", "sum"),
    ).reset_index()

    # Quintile scoring: lower recency is better (score 5), higher F/M is better.
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def label_rfm(score):
        if score >= 12:
            return "Champions"
        elif score >= 9:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        else:
            return "At Risk"

    rfm["RFM_Segment"] = rfm["RFM_Score"].apply(label_rfm)

    print(f"\nSnapshot date used for Recency: {snapshot_date.date()}")
    print("\n--- RFM segment counts ---")
    print(rfm["RFM_Segment"].value_counts())

    fig, ax = plt.subplots()
    sns.countplot(y="RFM_Segment", data=rfm, order=rfm["RFM_Segment"].value_counts().index,
                   ax=ax, palette="flare")
    ax.set_title("Customer Count by RFM Segment")
    ax.set_xlabel("Number of Customers")
    _save_fig(fig, "12_rfm_segments")

    return rfm


def clv_estimation(rfm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, transparent CLV estimate:
        CLV = Avg Order Value x Purchase Frequency x Estimated Lifespan (years)

    Assumption: since the dataset covers roughly one month of transactions,
    we annualize frequency (Frequency observed x 12) and assume a 3-year
    average customer lifespan, a common starting assumption for retail CLV
    when no churn/tenure history is available. This should be replaced with
    an empirically fit lifespan once more transaction history exists.
    """
    _section("STAGE 5b — CUSTOMER LIFETIME VALUE (CLV) ESTIMATION")

    ASSUMED_LIFESPAN_YEARS = 3
    avg_order_value = rfm["Monetary"] / rfm["Frequency"]
    annualized_frequency = rfm["Frequency"] * 12  # dataset spans ~1 month
    rfm = rfm.copy()
    rfm["AvgOrderValue"] = avg_order_value
    rfm["EstimatedCLV"] = avg_order_value * annualized_frequency * ASSUMED_LIFESPAN_YEARS

    print(f"Assumption: {ASSUMED_LIFESPAN_YEARS}-year average customer lifespan; "
          f"monthly purchase frequency annualized (x12) due to the ~1-month "
          f"observation window in this dataset.")
    print("\n--- Top 10 customers by estimated CLV ---")
    print(rfm.sort_values("EstimatedCLV", ascending=False)
          [["CustomerID", "AvgOrderValue", "Frequency", "EstimatedCLV"]]
          .head(10).round(2).to_string(index=False))

    fig, ax = plt.subplots()
    sns.histplot(rfm["EstimatedCLV"], bins=15, kde=True, ax=ax, color="#8172B2")
    ax.set_title("Distribution of Estimated Customer Lifetime Value")
    ax.set_xlabel("Estimated CLV ($)")
    _save_fig(fig, "13_clv_distribution")

    return rfm


def segment_customers(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    K-Means clustering on standardized Recency/Frequency/Monetary values to
    produce a data-driven customer segmentation, independent of the RFM
    quintile labels above (useful as a cross-check).
    """
    _section("STAGE 5c — CUSTOMER SEGMENTATION (K-MEANS)")
    rfm = rfm.copy()

    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaled = StandardScaler().fit_transform(features)

    # With only 34 customers in this dataset, cap clusters sensibly.
    k = min(n_clusters, max(2, rfm.shape[0] // 5))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    print(f"K-Means fit with k={k} clusters on standardized R/F/M features.")
    cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    cluster_profile["CustomerCount"] = rfm["Cluster"].value_counts().sort_index()
    print("\n--- Cluster profiles (mean R/F/M) ---")
    print(cluster_profile)

    fig, ax = plt.subplots()
    scatter = ax.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"],
                          cmap="tab10", s=80, alpha=0.8, edgecolor="black")
    ax.set_title("Customer Segments — Frequency vs. Monetary Value")
    ax.set_xlabel("Frequency (transaction count)")
    ax.set_ylabel("Monetary (total spend, $)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    _save_fig(fig, "14_kmeans_segments")

    return rfm


def trend_and_seasonality(df: pd.DataFrame) -> None:
    """Purchase trend over time and seasonal (weekday/monthly) behavior."""
    _section("STAGE 5d — PURCHASE TREND & SEASONALITY")

    daily = df.groupby(df["TransactionDate"].dt.date)["PurchaseAmount"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    daily.plot(ax=ax, marker="o", color="#DD8452")
    ax.set_title("Daily Revenue Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    _save_fig(fig, "15_daily_revenue_trend")

    weekday_avg = df.groupby("TransactionWeekday")["PurchaseAmount"].mean()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = weekday_avg.reindex([d for d in weekday_order if d in weekday_avg.index])
    fig, ax = plt.subplots()
    sns.barplot(x=weekday_avg.index, y=weekday_avg.values, ax=ax, palette="cubehelix")
    ax.set_title("Average Purchase Amount by Day of Week")
    ax.set_ylabel("Average Purchase Amount ($)")
    ax.tick_params(axis="x", rotation=45)
    _save_fig(fig, "16_weekday_seasonality")

    print("Note: this dataset spans a short (~1 month) window, so monthly/"
          "quarterly seasonality cannot be reliably assessed yet — daily and "
          "weekday patterns are shown instead. Re-run this analysis once a "
          "full year of transactions is available to evaluate quarter-level "
          "seasonality (TransactionQuarter is already engineered for that).")


def correlation_analysis(df: pd.DataFrame) -> None:
    """Correlation among the key numeric variables."""
    _section("STAGE 5e — CORRELATION ANALYSIS")

    numeric_cols = ["PurchaseAmount", "CustomerSatisfaction", "TransactionMonth", "TransactionQuarter"]
    # Drop any column with zero variance (e.g. a single quarter/month in a
    # short observation window) — correlation is undefined for a constant
    # column and would otherwise print as a confusing NaN.
    numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    corr = df[numeric_cols].corr()
    print(corr.round(2))

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Correlation Matrix — Key Numeric Variables")
    _save_fig(fig, "17_correlation_matrix")

    print("\nInterpretation: correlations near 0 indicate PurchaseAmount is "
          "largely independent of satisfaction score and calendar timing in "
          "this dataset — spend appears driven more by customer- and "
          "category-level factors than by when the purchase happened or how "
          "satisfied the customer reported being on that visit.")


# ============================================================================
# STAGE 6 — BUSINESS RECOMMENDATIONS
# ============================================================================
def print_recommendations(findings: dict) -> None:
    """
    Translate the analytical findings above into actionable recommendations.
    Each recommendation references the specific evidence that supports it.
    """
    _section("STAGE 6 — BUSINESS RECOMMENDATIONS")

    recs = [
        ("Increase revenue",
         "Prioritize inventory, promotion budget, and homepage placement for "
         "the top-revenue categories identified in Stage 3 (see "
         "04_revenue_by_category.png). Bundle top categories with "
         "underperforming ones (see growth_opportunity 'underperforming "
         "categories') to lift their trial."),
        ("Improve customer retention",
         f"Launch a targeted win-back campaign for the "
         f"{findings.get('declining_customers_count', 'N/A')} customers "
         f"labeled 'Decline' (Stage 4) — e.g., a personalized discount or "
         f"check-in outreach before they churn completely."),
        ("Increase customer satisfaction",
         f"Investigate the {findings.get('low_satisfaction_share', 0):.0%} "
         f"of transactions scoring <=2 on satisfaction (Stage 4), especially "
         f"any channel or region where it concentrates (see "
         f"11_satisfaction_by_channel.png), and address root causes "
         f"(fulfillment speed, product quality, staff training)."),
        ("Expand profitable customer segments",
         "Double down on marketing spend toward the 'Growth' and "
         "'Promising' behavioral segments (Stage 4) and toward the RFM "
         "'Champions'/'Loyal Customers' groups (Stage 5a) — they already "
         "show the highest engagement and are the cheapest incremental "
         "revenue to capture."),
        ("Improve marketing effectiveness",
         "Use the K-Means clusters (Stage 5c) to build 3-4 targeted "
         "campaigns instead of one-size-fits-all messaging — e.g., a "
         "high-frequency/high-spend cluster warrants loyalty perks, while a "
         "low-frequency/low-spend cluster warrants reactivation offers."),
        ("Identify cross-selling and upselling opportunities",
         "Use the top category preferences per age/gender segment "
         "(08_age_gender_avg_spend.png) to recommend complementary "
         "categories at checkout, and target high-CLV customers (Stage 5b) "
         "with premium/upsell offers first, since they have the highest "
         "propensity to convert."),
        ("Address regional and channel gaps",
         f"Investigate why the {findings.get('underperforming_region', 'N/A')} "
         f"region underperforms (Stage 4) — it may reflect a marketing gap, "
         f"logistics gap, or genuinely smaller addressable market, each of "
         f"which implies a different fix."),
    ]

    for i, (goal, action) in enumerate(recs, start=1):
        print(f"\n{i}. {goal}\n   -> {action}")

    print("\nAll recommendations above are grounded in the specific charts "
          "and tables produced by Stages 3-5; see outputs/plots/ for the "
          "supporting visualizations.")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_full_pipeline(data_path: str) -> pd.DataFrame:
    """Runs Stages 1-6 end to end and returns the final cleaned DataFrame."""
    _ensure_output_dirs()

    df = load_data(data_path)
    inspect_data(df)

    df = clean_data(df)
    df = standardize_product_categories(df)

    customer_summary = customer_performance_analysis(df)
    findings = growth_opportunity_analysis(df, customer_summary)

    rfm = rfm_analysis(df)
    rfm = clv_estimation(rfm, df)
    rfm = segment_customers(rfm)
    trend_and_seasonality(df)
    correlation_analysis(df)

    print_recommendations(findings)

    # Persist the cleaned dataset for reuse (e.g., by a future Streamlit app).
    df.to_csv(OUTPUT_DIR / "cleaned_transactions.csv", index=False)
    rfm.to_csv(OUTPUT_DIR / "customer_rfm_clv_segments.csv", index=False)

    _section("DONE")
    print(f"Cleaned dataset written to {OUTPUT_DIR / 'cleaned_transactions.csv'}")
    print(f"Customer RFM/CLV/segment table written to "
          f"{OUTPUT_DIR / 'customer_rfm_clv_segments.csv'}")
    print(f"All charts written to {PLOT_DIR}/")

    return df


def main():
    parser = argparse.ArgumentParser(description="Customer transaction analytics pipeline.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                         help="Path to the transaction dataset (.xlsx or .csv).")
    args = parser.parse_args()

    run_full_pipeline(args.data)


if __name__ == "__main__":
    main()
