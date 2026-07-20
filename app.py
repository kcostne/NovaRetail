"""
NovaRetail Customer Intelligence Dashboard
--------------------------------------------
Executive dashboard built for NovaRetail's Director of Customer Intelligence.
Reads NR_dataset_edited.xlsx, cleans/standardizes it, and renders KPI cards,
interactive Plotly charts, a Top 5 customers table, and an auto-generated
Business Insights section — all filterable from the sidebar.

Run locally:    streamlit run app.py
Deploy:         push app.py, requirements.txt, NR_dataset_edited.xlsx to GitHub
                and point Streamlit Community Cloud at app.py.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NovaRetail | Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "NR_dataset_edited.xlsx"

# ----------------------------------------------------------------------------
# Category standardization mapping
# ----------------------------------------------------------------------------
# Only HIGH and MEDIUM confidence groups (identified via manual review of the
# ProductCategory column) are merged automatically. LOW confidence groups are
# intentionally left untouched so the Director can review them separately.
#
#   HIGH confidence:
#     Groceries / Grocery / Grocery Items          -> Groceries
#     Toys / Toys & Games                          -> Toys & Games
#     Sporting Goods / Sports & Outdoors /
#       Sports Equipment                           -> Sports & Outdoors
#     Furniture / Furniture & Decor                -> Furniture & Decor
#
#   MEDIUM confidence:
#     Fashion / Fashion & Apparel / Clothing        -> Fashion & Apparel
#     Beauty Products / Beauty & Personal Care /
#       Cosmetics                                   -> Beauty & Personal Care
#     Health & Wellness / Health Supplements        -> Health & Wellness
#     Books / Books & Magazines                     -> Books & Magazines
#
#   LOW confidence (NOT merged): Home Decor, Home & Garden, Home Improvement,
#     Fashion Accessories, Children's Clothing, Sportswear, Health & Beauty,
#     Food & Beverages, Gardening Tools, Outdoor Equipment.
CATEGORY_MAPPING = {
    # High confidence
    "Groceries": "Groceries",
    "Grocery": "Groceries",
    "Grocery Items": "Groceries",
    "Toys": "Toys & Games",
    "Toys & Games": "Toys & Games",
    "Sporting Goods": "Sports & Outdoors",
    "Sports & Outdoors": "Sports & Outdoors",
    "Sports Equipment": "Sports & Outdoors",
    "Furniture": "Furniture & Decor",
    "Furniture & Decor": "Furniture & Decor",
    # Medium confidence
    "Fashion": "Fashion & Apparel",
    "Fashion & Apparel": "Fashion & Apparel",
    "Clothing": "Fashion & Apparel",
    "Beauty Products": "Beauty & Personal Care",
    "Beauty & Personal Care": "Beauty & Personal Care",
    "Cosmetics": "Beauty & Personal Care",
    "Health & Wellness": "Health & Wellness",
    "Health Supplements": "Health & Wellness",
    "Books": "Books & Magazines",
    "Books & Magazines": "Books & Magazines",
}


# ----------------------------------------------------------------------------
# Data loading & cleaning (cached so filters don't re-read the file each time)
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str):
    """Read the raw Excel file, clean it, and standardize ProductCategory.

    Returns:
        df            -- cleaned dataframe ready for the dashboard
        cats_before    -- sorted unique ProductCategory values before mapping
        cats_after     -- sorted unique ProductCategory values after mapping
        n_changed      -- number of records whose category value changed
    """
    df = pd.read_excel(path)

    # --- Handle missing values -------------------------------------------------
    # 'label' (customer segment) is the only column with missing values in
    # this dataset. Rather than dropping the row (and losing its revenue/
    # satisfaction data), we tag it explicitly so it's visible in filters.
    df["label"] = df["label"].fillna("Unclassified")

    # Defensive cleaning in case other columns ever contain blanks/nulls.
    text_cols = [
        "ProductCategory",
        "CustomerAgeGroup",
        "CustomerGender",
        "CustomerRegion",
        "RetailChannel",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["PurchaseAmount"] = pd.to_numeric(df["PurchaseAmount"], errors="coerce")
    df["CustomerSatisfaction"] = pd.to_numeric(
        df["CustomerSatisfaction"], errors="coerce"
    )
    df = df.dropna(subset=["PurchaseAmount", "CustomerSatisfaction"])

    # --- Convert TransactionDate to datetime -----------------------------------
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df = df.dropna(subset=["TransactionDate"])

    # --- Standardize ProductCategory using the approved mapping -----------------
    cats_before = sorted(df["ProductCategory"].unique().tolist())

    # Only replace values that are actually in the approved mapping; anything
    # not in CATEGORY_MAPPING (including all "Low confidence" categories) is
    # left exactly as-is.
    original = df["ProductCategory"].copy()
    df["ProductCategory"] = df["ProductCategory"].apply(
        lambda x: CATEGORY_MAPPING.get(x, x)
    )
    n_changed = int((original != df["ProductCategory"]).sum())

    cats_after = sorted(df["ProductCategory"].unique().tolist())

    return df, cats_before, cats_after, n_changed


df, cats_before, cats_after, n_changed = load_and_clean_data(DATA_PATH)

# ----------------------------------------------------------------------------
# Sidebar — filters
# ----------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

segments = sorted(df["label"].unique().tolist())
regions = sorted(df["CustomerRegion"].unique().tolist())
categories = sorted(df["ProductCategory"].unique().tolist())
channels = sorted(df["RetailChannel"].unique().tolist())
age_groups = sorted(df["CustomerAgeGroup"].unique().tolist())
genders = sorted(df["CustomerGender"].unique().tolist())

selected_segments = st.sidebar.multiselect(
    "Customer Segment", segments, default=segments
)
selected_regions = st.sidebar.multiselect("Customer Region", regions, default=regions)
selected_categories = st.sidebar.multiselect(
    "Product Category", categories, default=categories
)
selected_channels = st.sidebar.multiselect(
    "Retail Channel", channels, default=channels
)
selected_age_groups = st.sidebar.multiselect(
    "Customer Age Group", age_groups, default=age_groups
)
selected_genders = st.sidebar.multiselect(
    "Customer Gender", genders, default=genders
)

min_date = df["TransactionDate"].min().date()
max_date = df["TransactionDate"].max().date()
selected_date_range = st.sidebar.date_input(
    "Transaction Date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
# date_input can return a single date while the user is mid-selection; guard it
if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date, end_date = min_date, max_date

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing data from **{min_date}** to **{max_date}** · "
    f"{len(df):,} total transactions in the cleaned dataset."
)

# --- Apply filters -----------------------------------------------------------
mask = (
    df["label"].isin(selected_segments)
    & df["CustomerRegion"].isin(selected_regions)
    & df["ProductCategory"].isin(selected_categories)
    & df["RetailChannel"].isin(selected_channels)
    & df["CustomerAgeGroup"].isin(selected_age_groups)
    & df["CustomerGender"].isin(selected_genders)
    & (df["TransactionDate"].dt.date >= start_date)
    & (df["TransactionDate"].dt.date <= end_date)
)
fdf = df.loc[mask].copy()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛍️ NovaRetail Customer Intelligence Dashboard")
st.caption(
    "Executive view of customer behavior, revenue performance, and retention "
    "signals — built for the Director of Customer Intelligence."
)

if fdf.empty:
    st.warning("No transactions match the current filters. Adjust the sidebar filters to see results.")
    st.stop()

# ----------------------------------------------------------------------------
# Data quality note — category standardization audit trail
# ----------------------------------------------------------------------------
with st.expander("🧹 Data Quality: Product Category Standardization", expanded=False):
    st.write(
        "Categories that clearly represented the same business category "
        "(High/Medium confidence naming variants) were standardized. "
        "Low-confidence potential matches were left unchanged for manual review."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Before standardization**")
        st.write(cats_before)
    with col_b:
        st.markdown("**After standardization**")
        st.write(cats_after)
    st.info(f"**{n_changed}** records were re-labeled to a standardized category name.")

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
total_revenue = fdf["PurchaseAmount"].sum()
num_customers = fdf["CustomerID"].nunique()
avg_purchase = fdf["PurchaseAmount"].mean()
avg_satisfaction = fdf["CustomerSatisfaction"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Revenue", f"${total_revenue:,.2f}")
kpi2.metric("Number of Customers", f"{num_customers:,}")
kpi3.metric("Average Purchase Amount", f"${avg_purchase:,.2f}")
kpi4.metric("Average Customer Satisfaction", f"{avg_satisfaction:.2f} / 5")

st.markdown("---")

# ----------------------------------------------------------------------------
# Row 1: Revenue trend + Revenue by segment
# ----------------------------------------------------------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Revenue Trend Over Time")
    trend = (
        fdf.groupby(fdf["TransactionDate"].dt.date)["PurchaseAmount"]
        .sum()
        .reset_index()
        .rename(columns={"TransactionDate": "Date", "PurchaseAmount": "Revenue"})
    )
    fig_trend = px.line(
        trend, x="Date", y="Revenue", markers=True, template="plotly_white"
    )
    fig_trend.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_trend, use_container_width=True)

with row1_col2:
    st.subheader("Revenue by Customer Segment")
    seg_rev = (
        fdf.groupby("label")["PurchaseAmount"]
        .sum()
        .reset_index()
        .sort_values("PurchaseAmount", ascending=False)
    )
    fig_seg = px.bar(
        seg_rev,
        x="label",
        y="PurchaseAmount",
        color="label",
        template="plotly_white",
        labels={"label": "Segment", "PurchaseAmount": "Revenue"},
    )
    fig_seg.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_seg, use_container_width=True)

# ----------------------------------------------------------------------------
# Row 2: Segment distribution + Revenue by product category
# ----------------------------------------------------------------------------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Customer Segment Distribution")
    seg_counts = fdf["label"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_seg_dist = px.pie(
        seg_counts, names="Segment", values="Count", hole=0.45, template="plotly_white"
    )
    fig_seg_dist.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_seg_dist, use_container_width=True)

with row2_col2:
    st.subheader("Revenue by Product Category")
    cat_rev = (
        fdf.groupby("ProductCategory")["PurchaseAmount"]
        .sum()
        .reset_index()
        .sort_values("PurchaseAmount", ascending=True)
    )
    fig_cat = px.bar(
        cat_rev,
        x="PurchaseAmount",
        y="ProductCategory",
        orientation="h",
        template="plotly_white",
        labels={"PurchaseAmount": "Revenue", "ProductCategory": "Category"},
    )
    fig_cat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cat, use_container_width=True)

# ----------------------------------------------------------------------------
# Row 3: Revenue by region + Revenue by retail channel
# ----------------------------------------------------------------------------
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("Revenue by Region")
    region_rev = (
        fdf.groupby("CustomerRegion")["PurchaseAmount"]
        .sum()
        .reset_index()
        .sort_values("PurchaseAmount", ascending=False)
    )
    fig_region = px.bar(
        region_rev,
        x="CustomerRegion",
        y="PurchaseAmount",
        color="CustomerRegion",
        template="plotly_white",
        labels={"CustomerRegion": "Region", "PurchaseAmount": "Revenue"},
    )
    fig_region.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_region, use_container_width=True)

with row3_col2:
    st.subheader("Revenue by Retail Channel")
    channel_rev = (
        fdf.groupby("RetailChannel")["PurchaseAmount"]
        .sum()
        .reset_index()
    )
    fig_channel = px.pie(
        channel_rev,
        names="RetailChannel",
        values="PurchaseAmount",
        hole=0.45,
        template="plotly_white",
    )
    fig_channel.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_channel, use_container_width=True)

# ----------------------------------------------------------------------------
# Row 4: Satisfaction distribution + Purchase amount vs satisfaction
# ----------------------------------------------------------------------------
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.subheader("Customer Satisfaction Distribution")
    fig_sat = px.histogram(
        fdf,
        x="CustomerSatisfaction",
        nbins=5,
        template="plotly_white",
        labels={"CustomerSatisfaction": "Satisfaction Rating"},
    )
    fig_sat.update_layout(margin=dict(l=10, r=10, t=10, b=10), bargap=0.15)
    st.plotly_chart(fig_sat, use_container_width=True)

with row4_col2:
    st.subheader("Purchase Amount vs. Customer Satisfaction")
    fig_scatter = px.scatter(
        fdf,
        x="CustomerSatisfaction",
        y="PurchaseAmount",
        color="label",
        template="plotly_white",
        labels={
            "CustomerSatisfaction": "Satisfaction Rating",
            "PurchaseAmount": "Purchase Amount",
            "label": "Segment",
        },
        hover_data=["CustomerID", "ProductCategory"],
    )
    fig_scatter.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# Top 5 Customers table
# ----------------------------------------------------------------------------
st.subheader("🏆 Top 5 Customers by Total Revenue")
top_customers = (
    fdf.groupby("CustomerID")
    .agg(
        TotalRevenue=("PurchaseAmount", "sum"),
        Transactions=("PurchaseAmount", "count"),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
        Segment=("label", lambda s: s.mode().iat[0] if not s.mode().empty else "N/A"),
        Region=("CustomerRegion", lambda s: s.mode().iat[0] if not s.mode().empty else "N/A"),
    )
    .reset_index()
    .sort_values("TotalRevenue", ascending=False)
    .head(5)
)
top_customers["TotalRevenue"] = top_customers["TotalRevenue"].round(2)
top_customers["AvgSatisfaction"] = top_customers["AvgSatisfaction"].round(2)
st.dataframe(top_customers, use_container_width=True, hide_index=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# Business Insights (auto-generated from the filtered data)
# ----------------------------------------------------------------------------
st.subheader("💡 Business Insights")

# Highest-performing customer segments
seg_perf = fdf.groupby("label")["PurchaseAmount"].sum().sort_values(ascending=False)
top_segment = seg_perf.index[0] if len(seg_perf) else "N/A"
top_segment_share = (seg_perf.iloc[0] / total_revenue * 100) if total_revenue else 0

# Strongest product categories
cat_perf = fdf.groupby("ProductCategory")["PurchaseAmount"].sum().sort_values(ascending=False)
top_category = cat_perf.index[0] if len(cat_perf) else "N/A"
top_category_share = (cat_perf.iloc[0] / total_revenue * 100) if total_revenue else 0

# Regions with growth opportunity (lowest revenue region)
region_perf = fdf.groupby("CustomerRegion")["PurchaseAmount"].sum().sort_values()
growth_region = region_perf.index[0] if len(region_perf) else "N/A"

# Warning signs — customers in decline or with low satisfaction
declining_customers = fdf[fdf["label"] == "Decline"]["CustomerID"].nunique()
low_satisfaction_pct = (fdf["CustomerSatisfaction"] <= 2).mean() * 100

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("**📈 Performance Highlights**")
    st.markdown(
        f"- **{top_segment}** is the top-performing customer segment, "
        f"contributing **{top_segment_share:.1f}%** of total revenue.\n"
        f"- **{top_category}** is the strongest product category, generating "
        f"**${cat_perf.iloc[0]:,.2f}** (**{top_category_share:.1f}%** of revenue).\n"
        f"- The **{growth_region}** region currently shows the lowest revenue "
        f"and represents a growth opportunity for targeted campaigns."
    )

with insight_col2:
    st.markdown("**⚠️ Warning Signs**")
    st.markdown(
        f"- **{declining_customers}** customer(s) are currently flagged in the "
        f"**Decline** segment and are at risk of churn.\n"
        f"- **{low_satisfaction_pct:.1f}%** of transactions have a satisfaction "
        f"rating of 2 or below, signaling dissatisfaction risk.\n"
        f"- Monitor the **{growth_region}** region and the **Decline** segment "
        f"closely — both indicate softening customer engagement."
    )

st.markdown("**🎯 Recommended Actions**")
st.markdown(
    f"""
1. **Protect at-risk revenue** — launch a targeted retention offer (loyalty
   discount, personalized outreach) for the **Decline** segment before their
   next expected purchase window.
2. **Double down on what's working** — increase marketing spend behind
   **{top_category}**, NovaRetail's strongest category, and cross-sell it to
   customers in the **{top_segment}** segment who haven't purchased it yet.
3. **Invest in the {growth_region} region** — run localized promotions or
   channel expansion (e.g., more online availability) to close the revenue
   gap with top-performing regions.
"""
)

st.markdown("---")
st.caption("NovaRetail Customer Intelligence Dashboard · Built with Streamlit & Plotly")
