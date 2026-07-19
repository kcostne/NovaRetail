"""
================================================================================
 CUSTOMER TRANSACTION ANALYTICS DASHBOARD (app.py)
================================================================================
A Streamlit dashboard that analyzes a retail customer-transaction dataset and
surfaces customer performance insights and business growth opportunities.

Built with exactly three libraries: pandas, plotly, and streamlit.
(Only the Python standard library is used besides these three.)

Organized into the six stages requested in the project brief:
    Stage 1 - Data Understanding
    Stage 2 - Data Preparation
    Stage 3 - Customer Performance Analysis
    Stage 4 - Growth Opportunity Analysis
    Stage 5 - Advanced Analytics (RFM, CLV, segmentation, trends, correlation)
    Stage 6 - Business Recommendations

How to run:
    streamlit run app.py

By default the app loads "NR_dataset.xlsx" from the same folder. If that file
isn't found, a file-uploader appears in the sidebar so any .xlsx/.csv with the
same column layout can be analyzed instead.

NOTE ON THE K-MEANS SUBSTITUTION: the original version of this project used
scikit-learn's KMeans for customer segmentation. Since this rebuild is
restricted to pandas + plotly + streamlit (no scikit-learn, no numpy),
Stage 5's "customer segmentation" is instead produced with a transparent,
rule-based RFM-quartile method (implemented with pandas.qcut) — it needs no
extra library, and it is easier for a non-technical stakeholder to audit
than a black-box clustering result. See rfm_analysis() below.
================================================================================
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Transaction Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = "NR_dataset.xlsx"
PLOTLY_TEMPLATE = "plotly_white"


# ============================================================================
# STAGE 1 — DATA UNDERSTANDING
# ============================================================================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    """
    Load the transaction dataset from an uploaded file object or a path
    string. Supports .xlsx and .csv.
    """
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def render_stage1(df: pd.DataFrame) -> None:
    st.header("Stage 1 — Data Understanding")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Column data types")
    dtypes_df = pd.DataFrame({"Column": df.dtypes.index, "Dtype": df.dtypes.astype(str).values})
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    st.subheader("Missing values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing.rename("Missing count"), use_container_width=True)

    st.subheader("Numerical variable summary")
    st.dataframe(df.describe(include=["number"]).round(2), use_container_width=True)

    st.subheader("Categorical variable summary")
    st.dataframe(df.describe(include=["object", "string"]), use_container_width=True)

    # Data-quality note: TransactionID is not unique per row in this dataset.
    repeated_tx = df["TransactionID"].value_counts()
    n_shared = int((repeated_tx > 1).sum())
    if n_shared:
        st.info(
            f"Note: TransactionID is shared across multiple rows for "
            f"{n_shared} of {repeated_tx.shape[0]} unique IDs. Each **row** "
            f"is treated as one purchase line-item throughout this "
            f"dashboard, not each TransactionID."
        )


# ============================================================================
# STAGE 2 — DATA PREPARATION
# ============================================================================
def clean_data(df: pd.DataFrame):
    """
    Clean missing/inconsistent data and engineer date-derived features.
    Returns the cleaned DataFrame plus a log of what was done, so the log
    can be displayed in the UI (keeps the transformation auditable).
    """
    df = df.copy()
    log = []

    # 1) 'label' (behavioral segment) can contain missing values. It's a
    #    categorical business label, not something safe to infer
    #    numerically, so missing values are filled with an explicit
    #    "Unlabeled" placeholder rather than dropping the row (dropping
    #    would also discard a real transaction from every other analysis).
    missing_labels = int(df["label"].isnull().sum())
    df["label"] = df["label"].fillna("Unlabeled")
    log.append(f"Filled {missing_labels} missing 'label' value(s) with 'Unlabeled'.")

    # 2) Enforce TransactionDate as a real datetime; unparseable values
    #    become NaT and are dropped (each such row cannot be time-analyzed).
    before_dtype = str(df["TransactionDate"].dtype)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    bad_dates = int(df["TransactionDate"].isnull().sum())
    log.append(f"TransactionDate dtype converted from {before_dtype} to datetime64. "
               f"{bad_dates} unparseable date(s) coerced to NaT.")
    if bad_dates:
        df = df.dropna(subset=["TransactionDate"])
        log.append(f"Dropped {bad_dates} row(s) with an invalid TransactionDate.")

    # 3) Engineer time-based features used throughout Stages 3-5.
    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionMonthName"] = df["TransactionDate"].dt.month_name()
    df["TransactionQuarter"] = df["TransactionDate"].dt.quarter
    df["TransactionWeekday"] = df["TransactionDate"].dt.day_name()
    log.append("Engineered columns: TransactionYear, TransactionMonth, "
                "TransactionMonthName, TransactionQuarter, TransactionWeekday.")

    # 4) Guard against non-positive purchase amounts (data-entry errors).
    invalid_amounts = df[df["PurchaseAmount"] <= 0]
    if not invalid_amounts.empty:
        log.append(f"Removed {len(invalid_amounts)} row(s) with PurchaseAmount <= 0.")
        df = df[df["PurchaseAmount"] > 0]
    else:
        log.append("PurchaseAmount check: no zero/negative values found.")

    # 5) Whitespace hygiene on text columns.
    text_cols = ["ProductCategory", "CustomerAgeGroup", "CustomerGender",
                 "CustomerRegion", "RetailChannel", "label"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
    log.append("Trimmed leading/trailing whitespace on all categorical text columns.")

    return df, log


def standardize_product_categories(df: pd.DataFrame):
    """
    ProductCategory standardization.

    Only HIGH-confidence groups are merged automatically; MEDIUM and LOW
    confidence groups are deliberately left untouched so a human reviewer
    can approve them explicitly (per the brief's requirement not to
    auto-merge ambiguous categories).

    Approved (HIGH confidence) merges:
        "Beauty Products"              -> "Beauty & Personal Care"
        "Clothing", "Fashion"          -> "Fashion & Apparel"
        "Grocery", "Grocery Items"     -> "Groceries"
        "Furniture"                    -> "Furniture & Decor"
        "Sports Equipment"             -> "Sporting Goods"
        "Toys"                         -> "Toys & Games"

    Flagged but NOT merged (Medium/Low confidence — left for business
    review): Cosmetics vs. Beauty & Personal Care (Medium); Health & Beauty
    vs. Beauty & Personal Care (Low); Books vs. Books & Magazines (Medium);
    Fashion Accessories (Low); Children's Clothing (Low); Sportswear (Low);
    Food & Beverages vs. Groceries (Medium); Home Decor vs. Furniture &
    Decor (Medium); Home & Garden / Home Improvement / Home Appliances /
    Gardening Tools (Low, distinct sub-departments); Sports & Outdoors /
    Outdoor Equipment vs. Sporting Goods (Medium); Health & Wellness vs.
    Health Supplements (Medium).
    """
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
    n_changed = int(df["ProductCategory"].isin(approved_mapping.keys()).sum())
    df["ProductCategory"] = df["ProductCategory"].replace(approved_mapping)
    return df, approved_mapping, n_changed


def render_stage2(raw_df: pd.DataFrame) -> pd.DataFrame:
    st.header("Stage 2 — Data Preparation")

    df, log = clean_data(raw_df)

    st.subheader("Cleaning steps performed")
    for line in log:
        st.write(f"- {line}")

    st.subheader("ProductCategory standardization")
    before_unique = sorted(raw_df["ProductCategory"].astype(str).str.strip().unique())
    df, mapping, n_changed = standardize_product_categories(df)
    after_unique = sorted(df["ProductCategory"].unique())

    map_df = pd.DataFrame({
        "Original name": list(mapping.keys()),
        "Standardized name": list(mapping.values()),
        "Confidence": ["High"] * len(mapping),
    })
    st.write("Approved (high-confidence) merges applied automatically:")
    st.dataframe(map_df, use_container_width=True, hide_index=True)
    st.caption(
        "Medium/Low-confidence look-alikes (e.g. Cosmetics vs. Beauty & "
        "Personal Care, Books vs. Books & Magazines, Home Decor vs. "
        "Furniture & Decor) were intentionally left unmerged for business "
        "review — see the module docstring in the code for the full list "
        "and reasoning."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Before** ({len(before_unique)} unique categories)")
        st.dataframe(pd.DataFrame({"ProductCategory": before_unique}), height=250,
                     use_container_width=True, hide_index=True)
    with col2:
        st.write(f"**After** ({len(after_unique)} unique categories)")
        st.dataframe(pd.DataFrame({"ProductCategory": after_unique}), height=250,
                     use_container_width=True, hide_index=True)

    st.info(f"{n_changed} record(s) were reassigned to a standardized category "
            f"name out of {len(df)} total rows.")

    return df


# ============================================================================
# STAGE 3 — CUSTOMER PERFORMANCE ANALYSIS
# ============================================================================
def build_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
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


def render_stage3(df: pd.DataFrame) -> pd.DataFrame:
    st.header("Stage 3 — Customer Performance Analysis")

    customer_summary = build_customer_summary(df)

    st.subheader("Top 10 customers by total spend")
    st.dataframe(customer_summary.head(10).round(2), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(customer_summary, x="TotalSpend", nbins=15,
                            title="Distribution of Total Customer Spend",
                            template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Total Spend ($)", yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(customer_summary, x="NumTransactions",
                            title="Distribution of Transactions per Customer",
                            template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Number of Transactions", yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(customer_summary, x="AvgPurchaseValue",
                 title="Average Purchase Value per Customer",
                 template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product category preferences")
    cat_revenue = (
        df.groupby("ProductCategory")["PurchaseAmount"].sum()
        .sort_values(ascending=False).reset_index()
    )
    fig = px.bar(cat_revenue, x="PurchaseAmount", y="ProductCategory", orientation="h",
                 title="Total Revenue by Product Category", template=PLOTLY_TEMPLATE,
                 color="PurchaseAmount", color_continuous_scale="Viridis")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Total Revenue ($)",
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The categories at the top drive the largest share of revenue and are "
               "priority categories to keep well-stocked and well-marketed.")

    c3, c4 = st.columns(2)
    with c3:
        sat_counts = df["CustomerSatisfaction"].value_counts().sort_index().reset_index()
        sat_counts.columns = ["CustomerSatisfaction", "Count"]
        fig = px.bar(sat_counts, x="CustomerSatisfaction", y="Count",
                     title="Customer Satisfaction Score Distribution", template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Satisfaction Score (1 = lowest, 5 = highest)",
                           yaxis_title="Number of Transactions")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        channel_counts = df["RetailChannel"].value_counts().reset_index()
        channel_counts.columns = ["RetailChannel", "Count"]
        fig = px.pie(channel_counts, names="RetailChannel", values="Count",
                     title="Retail Channel Usage Share", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        region_revenue = df.groupby("CustomerRegion")["PurchaseAmount"].sum() \
            .sort_values(ascending=False).reset_index()
        fig = px.bar(region_revenue, x="CustomerRegion", y="PurchaseAmount",
                     title="Total Revenue by Customer Region", template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Region", yaxis_title="Total Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        age_gender_spend = df.pivot_table(
            index="CustomerAgeGroup", columns="CustomerGender",
            values="PurchaseAmount", aggfunc="mean"
        ).round(0)
        fig = px.imshow(age_gender_spend, text_auto=True, aspect="auto",
                         title="Average Purchase Amount by Age Group & Gender",
                         template=PLOTLY_TEMPLATE, color_continuous_scale="YlGnBu")
        st.plotly_chart(fig, use_container_width=True)

    return customer_summary


# ============================================================================
# STAGE 4 — GROWTH OPPORTUNITY ANALYSIS
# ============================================================================
def render_stage4(df: pd.DataFrame, customer_summary: pd.DataFrame) -> dict:
    st.header("Stage 4 — Growth Opportunity Analysis")
    findings = {}

    # --- High-value customers: top 20% by total spend ---
    threshold = customer_summary["TotalSpend"].quantile(0.80)
    high_value = customer_summary[customer_summary["TotalSpend"] >= threshold]
    findings["high_value_customers"] = high_value
    total_rev = customer_summary["TotalSpend"].sum()
    st.subheader("High-value customers")
    st.write(
        f"Top 20% by spend (>= ${threshold:,.2f}): **{len(high_value)} customers**, "
        f"representing **${high_value['TotalSpend'].sum():,.2f}** "
        f"({high_value['TotalSpend'].sum() / total_rev:.1%} of total revenue)."
    )
    hv_plot = high_value.sort_values("TotalSpend", ascending=False).copy()
    hv_plot["CustomerID"] = hv_plot["CustomerID"].astype(str)
    fig = px.bar(hv_plot, x="CustomerID", y="TotalSpend",
                 title="High-Value Customers (Top 20% by Spend)", template=PLOTLY_TEMPLATE,
                 color="TotalSpend", color_continuous_scale="Reds")
    fig.update_layout(xaxis_title="Customer ID", yaxis_title="Total Spend ($)",
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Customers with declining activity ---
    decline_customers = int(df[df["label"] == "Decline"]["CustomerID"].nunique())
    findings["declining_customers_count"] = decline_customers
    st.info(f"Customers labeled **'Decline'**: {decline_customers} unique customers — "
            f"retention-risk accounts and prime win-back targets.")

    # --- Segment revenue (growth potential) ---
    st.subheader("Revenue by customer behavioral segment")
    label_summary = df.groupby("label").agg(
        TotalRevenue=("PurchaseAmount", "sum"), Transactions=("PurchaseAmount", "count")
    ).reset_index()
    findings["growth_segment_revenue"] = label_summary
    fig = px.bar(label_summary, x="label", y="TotalRevenue",
                 title="Total Revenue by Customer Behavioral Segment", template=PLOTLY_TEMPLATE)
    fig.update_layout(xaxis_title="Segment (label)", yaxis_title="Total Revenue ($)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Underperforming regions ---
    st.subheader("Regional performance")
    region_perf = df.groupby("CustomerRegion").agg(
        Revenue=("PurchaseAmount", "sum"), AvgSatisfaction=("CustomerSatisfaction", "mean")
    ).sort_values("Revenue").round(2)
    findings["underperforming_region"] = region_perf.index[0]
    st.dataframe(region_perf, use_container_width=True)
    st.write(f"Underperforming region: **{region_perf.index[0]}** "
             f"(${region_perf['Revenue'].iloc[0]:,.2f} in revenue).")

    # --- Underperforming product categories ---
    st.subheader("Lowest-revenue product categories")
    cat_perf = df.groupby("ProductCategory").agg(
        Revenue=("PurchaseAmount", "sum"), Transactions=("PurchaseAmount", "count")
    ).sort_values("Revenue").round(2)
    findings["underperforming_categories"] = cat_perf.head(5)
    st.dataframe(cat_perf.head(5), use_container_width=True)

    # --- Retail channels with expansion opportunity ---
    st.subheader("Channel performance")
    channel_perf = df.groupby("RetailChannel").agg(
        Revenue=("PurchaseAmount", "sum"),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
        Transactions=("PurchaseAmount", "count"),
    ).round(2)
    findings["channel_perf"] = channel_perf
    st.dataframe(channel_perf, use_container_width=True)

    # --- Satisfaction issues affecting retention ---
    low_satisfaction = df[df["CustomerSatisfaction"] <= 2]
    low_sat_share = len(low_satisfaction) / len(df)
    findings["low_satisfaction_share"] = low_sat_share
    st.warning(f"Transactions with satisfaction <= 2: {len(low_satisfaction)} "
               f"({low_sat_share:.1%} of all transactions) — retention-risk touchpoints.")

    fig = px.box(df, x="RetailChannel", y="CustomerSatisfaction",
                 title="Customer Satisfaction by Retail Channel", template=PLOTLY_TEMPLATE,
                 color="RetailChannel")
    st.plotly_chart(fig, use_container_width=True)

    return findings


# ============================================================================
# STAGE 5 — ADVANCED ANALYTICS
# ============================================================================
def rfm_analysis(df: pd.DataFrame):
    """
    Recency / Frequency / Monetary analysis at the customer level, with 1-5
    quintile scores and a combined RFM segment label.

    This RFM-quartile segmentation also serves as this rebuild's
    "customer segmentation" deliverable — see the module docstring for why
    it replaces the scikit-learn KMeans clustering used in the prior
    version (this build is pandas + plotly + streamlit only).
    """
    snapshot_date = df["TransactionDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionID", "count"),
        Monetary=("PurchaseAmount", "sum"),
    ).reset_index()

    # Lower recency is better (score 5); higher frequency/monetary is better.
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                              labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def label_rfm(score):
        if score >= 12:
            return "Champions"
        elif score >= 9:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        return "At Risk"

    rfm["RFM_Segment"] = rfm["RFM_Score"].apply(label_rfm)
    return rfm, snapshot_date


def clv_estimation(rfm: pd.DataFrame):
    """
    CLV = Avg Order Value x Annualized Purchase Frequency x Estimated Lifespan.

    Assumption: this dataset spans roughly one month, so monthly frequency is
    annualized (x12), and a 3-year average customer lifespan is assumed (a
    standard starting point for retail CLV absent churn/tenure history).
    """
    ASSUMED_LIFESPAN_YEARS = 3
    rfm = rfm.copy()
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    annualized_frequency = rfm["Frequency"] * 12
    rfm["EstimatedCLV"] = rfm["AvgOrderValue"] * annualized_frequency * ASSUMED_LIFESPAN_YEARS
    return rfm, ASSUMED_LIFESPAN_YEARS


def render_stage5(df: pd.DataFrame) -> pd.DataFrame:
    st.header("Stage 5 — Advanced Analytics")

    # --- RFM ---
    st.subheader("RFM analysis & rule-based customer segmentation")
    rfm, snapshot_date = rfm_analysis(df)
    st.caption(f"Snapshot date used for Recency: {snapshot_date.date()}")

    seg_counts = rfm["RFM_Segment"].value_counts().reset_index()
    seg_counts.columns = ["RFM_Segment", "Customers"]
    fig = px.bar(seg_counts, x="Customers", y="RFM_Segment", orientation="h",
                 title="Customer Count by RFM Segment", template=PLOTLY_TEMPLATE,
                 color="RFM_Segment")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(rfm, x="Frequency", y="Monetary", color="RFM_Segment",
                      size="Recency", hover_data=["CustomerID"],
                      title="Customer Segments — Frequency vs. Monetary Value "
                            "(bubble size = Recency, larger = less recent)",
                      template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    # --- CLV ---
    st.subheader("Customer Lifetime Value (CLV) estimation")
    rfm, lifespan_years = clv_estimation(rfm)
    st.caption(f"Assumption: {lifespan_years}-year average customer lifespan; monthly "
               f"purchase frequency annualized (x12) due to the ~1-month observation "
               f"window in this dataset.")
    top_clv = rfm.sort_values("EstimatedCLV", ascending=False)[
        ["CustomerID", "AvgOrderValue", "Frequency", "EstimatedCLV", "RFM_Segment"]
    ].head(10).round(2)
    st.dataframe(top_clv, use_container_width=True, hide_index=True)

    fig = px.histogram(rfm, x="EstimatedCLV", nbins=15,
                        title="Distribution of Estimated Customer Lifetime Value",
                        template=PLOTLY_TEMPLATE)
    fig.update_layout(xaxis_title="Estimated CLV ($)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Trend & seasonality ---
    st.subheader("Purchase trend & seasonality")
    daily = df.groupby(df["TransactionDate"].dt.date)["PurchaseAmount"].sum().reset_index()
    daily.columns = ["Date", "Revenue"]
    fig = px.line(daily, x="Date", y="Revenue", markers=True,
                  title="Daily Revenue Trend", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = df.groupby("TransactionWeekday")["PurchaseAmount"].mean() \
        .reindex([d for d in weekday_order if d in df["TransactionWeekday"].unique()]).reset_index()
    weekday_avg.columns = ["Weekday", "AvgPurchaseAmount"]
    fig = px.bar(weekday_avg, x="Weekday", y="AvgPurchaseAmount",
                 title="Average Purchase Amount by Day of Week", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This dataset spans a short (~1 month) window, so monthly/quarterly "
               "seasonality cannot be reliably assessed yet — daily and weekday "
               "patterns are shown instead. TransactionQuarter is already engineered "
               "for future re-analysis once a full year of data is available.")

    # --- Correlation ---
    st.subheader("Correlation analysis")
    numeric_cols = ["PurchaseAmount", "CustomerSatisfaction", "TransactionMonth", "TransactionQuarter"]
    numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]  # drop constant columns
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Matrix — Key Numeric Variables", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlations near 0 indicate PurchaseAmount is largely independent of "
               "satisfaction score and calendar timing in this dataset — spend appears "
               "driven more by customer- and category-level factors than by when the "
               "purchase happened or how satisfied the customer reported being.")

    return rfm


# ============================================================================
# STAGE 6 — BUSINESS RECOMMENDATIONS
# ============================================================================
def render_stage6(findings: dict) -> None:
    st.header("Stage 6 — Business Recommendations")

    recs = [
        ("Increase revenue",
         "Prioritize inventory, promotion budget, and homepage placement for the "
         "top-revenue categories identified in Stage 3. Bundle top categories with "
         "underperforming ones (Stage 4) to lift their trial."),
        ("Improve customer retention",
         f"Launch a targeted win-back campaign for the "
         f"{findings.get('declining_customers_count', 'N/A')} customers labeled "
         f"'Decline' (Stage 4) — e.g., a personalized discount or check-in outreach "
         f"before they churn completely."),
        ("Increase customer satisfaction",
         f"Investigate the {findings.get('low_satisfaction_share', 0):.0%} of "
         f"transactions scoring <=2 on satisfaction (Stage 4), especially any "
         f"channel or region where it concentrates, and address root causes "
         f"(fulfillment speed, product quality, staff training)."),
        ("Expand profitable customer segments",
         "Double down on marketing spend toward the 'Growth'/'Promising' "
         "behavioral segments (Stage 4) and the RFM 'Champions'/'Loyal Customers' "
         "groups (Stage 5) — they already show the highest engagement and are the "
         "cheapest incremental revenue to capture."),
        ("Improve marketing effectiveness",
         "Use the RFM segments (Stage 5) to build targeted campaigns instead of "
         "one-size-fits-all messaging — e.g., 'At Risk' customers warrant "
         "reactivation offers, while 'Champions' warrant loyalty perks."),
        ("Identify cross-selling and upselling opportunities",
         "Use category preferences by age/gender segment (Stage 3) to recommend "
         "complementary categories at checkout, and target high-CLV customers "
         "(Stage 5) with premium/upsell offers first, since they have the highest "
         "propensity to convert."),
        ("Address regional and channel gaps",
         f"Investigate why the {findings.get('underperforming_region', 'N/A')} "
         f"region underperforms (Stage 4) — it may reflect a marketing gap, "
         f"logistics gap, or genuinely smaller addressable market, each implying "
         f"a different fix."),
    ]

    for i, (goal, action) in enumerate(recs, start=1):
        st.markdown(f"**{i}. {goal}**")
        st.write(action)

    st.success("All recommendations above are grounded in the specific charts and "
               "tables produced in Stages 3-5 of this dashboard.")


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("Customer Transaction Analytics & Growth Opportunity Dashboard")
    st.caption("Built with pandas, plotly, and streamlit.")

    st.sidebar.header("Data source")
    default_path = Path(DEFAULT_DATA_PATH)
    uploaded_file = st.sidebar.file_uploader("Upload a transaction file (.xlsx or .csv)",
                                              type=["xlsx", "csv"])

    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
    elif default_path.exists():
        raw_df = load_data(str(default_path))
        st.sidebar.info(f"Using default dataset: {DEFAULT_DATA_PATH}")
    else:
        st.warning("No dataset found. Upload a .xlsx or .csv file with the expected "
                    "columns (Idx, label, CustomerID, TransactionID, TransactionDate, "
                    "ProductCategory, PurchaseAmount, CustomerAgeGroup, CustomerGender, "
                    "CustomerRegion, CustomerSatisfaction, RetailChannel) to begin.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Data Understanding", "2. Data Preparation", "3. Customer Performance",
        "4. Growth Opportunities", "5. Advanced Analytics", "6. Recommendations",
    ])

    with tab1:
        render_stage1(raw_df)

    with tab2:
        df = render_stage2(raw_df)

    with tab3:
        customer_summary = render_stage3(df)

    with tab4:
        findings = render_stage4(df, customer_summary)

    with tab5:
        render_stage5(df)

    with tab6:
        render_stage6(findings)


if __name__ == "__main__":
    main()
