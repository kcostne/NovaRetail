"""
NovaRetail Customer Intelligence Dashboard
--------------------------------------------
An executive-facing Streamlit dashboard built for NovaRetail's Director of
Customer Intelligence. Provides KPI monitoring, segment/region/category
performance views, and auto-generated business insights to support
data-driven decision-making.

Run locally with:  streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NovaRetail | Customer Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# LIGHT EXECUTIVE STYLING
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall page padding */
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

        /* KPI metric cards */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E6E6E6;
            border-radius: 10px;
            padding: 1rem 1rem 0.5rem 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetricLabel"] {font-weight: 600; color: #4B5563;}

        /* Section headers */
        h2, h3 {color: #1F2937;}

        /* Insight cards */
        .insight-card {
            background-color: #F8FAFC;
            border-left: 4px solid #2563EB;
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
        }
        .insight-card.warning {border-left-color: #DC2626; background-color: #FEF2F2;}
        .insight-card.positive {border-left-color: #16A34A; background-color: #F0FDF4;}
        .insight-card.action {border-left-color: #D97706; background-color: #FFFBEB;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "NR_dataset_edited.xlsx") -> pd.DataFrame:
    """Load the NovaRetail dataset and apply light, targeted cleaning."""
    df = pd.read_excel(path)

    # Standardize column names / rename the segment label column for clarity
    df = df.rename(columns={"label": "CustomerSegment"})

    # Convert transaction date to proper datetime
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

    # Drop rows with no transaction date at all (unusable for time-based analysis)
    df = df.dropna(subset=["TransactionDate"])

    # Fill missing categorical values with an explicit "Unknown" bucket so
    # records aren't silently dropped from the dashboard
    for col in ["CustomerSegment", "ProductCategory", "CustomerAgeGroup",
                "CustomerGender", "CustomerRegion", "RetailChannel"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Purchase amount and satisfaction should be numeric; drop rows that
    # cannot be used for revenue / satisfaction calculations
    df["PurchaseAmount"] = pd.to_numeric(df["PurchaseAmount"], errors="coerce")
    df["CustomerSatisfaction"] = pd.to_numeric(df["CustomerSatisfaction"], errors="coerce")
    df = df.dropna(subset=["PurchaseAmount", "CustomerSatisfaction"])

    return df


df = load_data()

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.title("🔎 Filters")
st.sidebar.caption("Refine the dashboard to a specific slice of customers.")

segments = sorted(df["CustomerSegment"].unique())
regions = sorted(df["CustomerRegion"].unique())
categories = sorted(df["ProductCategory"].unique())
channels = sorted(df["RetailChannel"].unique())
age_groups = sorted(df["CustomerAgeGroup"].unique())
genders = sorted(df["CustomerGender"].unique())

sel_segments = st.sidebar.multiselect("Customer Segment", segments, default=segments)
sel_regions = st.sidebar.multiselect("Customer Region", regions, default=regions)
sel_categories = st.sidebar.multiselect("Product Category", categories, default=categories)
sel_channels = st.sidebar.multiselect("Retail Channel", channels, default=channels)
sel_age_groups = st.sidebar.multiselect("Customer Age Group", age_groups, default=age_groups)
sel_genders = st.sidebar.multiselect("Customer Gender", genders, default=genders)

min_date, max_date = df["TransactionDate"].min(), df["TransactionDate"].max()
date_range = st.sidebar.date_input(
    "Transaction Date",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
# Handle the case where the user has only picked a single date so far
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

# ----------------------------------------------------------------------------
# APPLY FILTERS
# ----------------------------------------------------------------------------
mask = (
    df["CustomerSegment"].isin(sel_segments)
    & df["CustomerRegion"].isin(sel_regions)
    & df["ProductCategory"].isin(sel_categories)
    & df["RetailChannel"].isin(sel_channels)
    & df["CustomerAgeGroup"].isin(sel_age_groups)
    & df["CustomerGender"].isin(sel_genders)
    & (df["TransactionDate"].dt.date >= start_date)
    & (df["TransactionDate"].dt.date <= end_date)
)
fdf = df.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(fdf):,}** of **{len(df):,}** transactions")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🛍️ NovaRetail Customer Intelligence Dashboard")
st.caption(
    "Executive view of customer behavior, revenue performance, and retention "
    "signals — built for the Director of Customer Intelligence."
)

if fdf.empty:
    st.warning("No transactions match the selected filters. Please broaden your filter selection.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------------------------
total_revenue = fdf["PurchaseAmount"].sum()
num_customers = fdf["CustomerID"].nunique()
avg_purchase = fdf["PurchaseAmount"].mean()
avg_satisfaction = fdf["CustomerSatisfaction"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
kpi2.metric("Number of Customers", f"{num_customers:,}")
kpi3.metric("Average Purchase Amount", f"${avg_purchase:,.2f}")
kpi4.metric("Average Customer Satisfaction", f"{avg_satisfaction:.2f} / 5")

st.markdown("---")

# ----------------------------------------------------------------------------
# REVENUE TREND
# ----------------------------------------------------------------------------
st.subheader("📈 Revenue Trend Over Time")
trend = (
    fdf.groupby(fdf["TransactionDate"].dt.date)["PurchaseAmount"]
    .sum()
    .reset_index()
    .rename(columns={"TransactionDate": "Date", "PurchaseAmount": "Revenue"})
)
fig_trend = px.line(
    trend, x="Date", y="Revenue", markers=True,
    template="plotly_white",
)
fig_trend.update_traces(line_color="#2563EB")
fig_trend.update_layout(margin=dict(t=20, b=20), height=350)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# SEGMENT VIEWS
# ----------------------------------------------------------------------------
st.subheader("👥 Customer Segment Performance")
col1, col2 = st.columns(2)

with col1:
    seg_rev = (
        fdf.groupby("CustomerSegment")["PurchaseAmount"]
        .sum()
        .reset_index()
        .rename(columns={"PurchaseAmount": "Revenue"})
        .sort_values("Revenue", ascending=False)
    )
    fig_seg_rev = px.bar(
        seg_rev, x="CustomerSegment", y="Revenue", color="CustomerSegment",
        template="plotly_white", title="Revenue by Customer Segment",
    )
    fig_seg_rev.update_layout(showlegend=False, margin=dict(t=40, b=20), height=350)
    st.plotly_chart(fig_seg_rev, use_container_width=True)

with col2:
    seg_dist = fdf["CustomerSegment"].value_counts().reset_index()
    seg_dist.columns = ["CustomerSegment", "Count"]
    fig_seg_dist = px.pie(
        seg_dist, names="CustomerSegment", values="Count", hole=0.45,
        template="plotly_white", title="Customer Segment Distribution",
    )
    fig_seg_dist.update_layout(margin=dict(t=40, b=20), height=350)
    st.plotly_chart(fig_seg_dist, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# CATEGORY / REGION / CHANNEL VIEWS
# ----------------------------------------------------------------------------
st.subheader("🛒 Revenue Breakdown")
col3, col4, col5 = st.columns(3)

with col3:
    cat_rev = (
        fdf.groupby("ProductCategory")["PurchaseAmount"]
        .sum()
        .reset_index()
        .rename(columns={"PurchaseAmount": "Revenue"})
        .sort_values("Revenue", ascending=False)
        .head(10)  # top 10 categories keeps the chart readable
    )
    fig_cat = px.bar(
        cat_rev, x="Revenue", y="ProductCategory", orientation="h",
        template="plotly_white", title="Revenue by Product Category (Top 10)",
        color="Revenue", color_continuous_scale="Blues",
    )
    fig_cat.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False, coloraxis_showscale=False,
        margin=dict(t=40, b=20), height=380,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col4:
    reg_rev = (
        fdf.groupby("CustomerRegion")["PurchaseAmount"]
        .sum()
        .reset_index()
        .rename(columns={"PurchaseAmount": "Revenue"})
        .sort_values("Revenue", ascending=False)
    )
    fig_reg = px.bar(
        reg_rev, x="CustomerRegion", y="Revenue", color="CustomerRegion",
        template="plotly_white", title="Revenue by Region",
    )
    fig_reg.update_layout(showlegend=False, margin=dict(t=40, b=20), height=380)
    st.plotly_chart(fig_reg, use_container_width=True)

with col5:
    chan_rev = (
        fdf.groupby("RetailChannel")["PurchaseAmount"]
        .sum()
        .reset_index()
        .rename(columns={"PurchaseAmount": "Revenue"})
    )
    fig_chan = px.pie(
        chan_rev, names="RetailChannel", values="Revenue", hole=0.45,
        template="plotly_white", title="Revenue by Retail Channel",
    )
    fig_chan.update_layout(margin=dict(t=40, b=20), height=380)
    st.plotly_chart(fig_chan, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# SATISFACTION VIEWS
# ----------------------------------------------------------------------------
st.subheader("⭐ Customer Satisfaction")
col6, col7 = st.columns(2)

with col6:
    fig_sat_dist = px.histogram(
        fdf, x="CustomerSatisfaction", nbins=5,
        template="plotly_white", title="Customer Satisfaction Distribution",
        color_discrete_sequence=["#2563EB"],
    )
    fig_sat_dist.update_layout(
        bargap=0.15, margin=dict(t=40, b=20), height=350,
        xaxis_title="Satisfaction Score", yaxis_title="Number of Transactions",
    )
    st.plotly_chart(fig_sat_dist, use_container_width=True)

with col7:
    fig_scatter = px.scatter(
        fdf, x="PurchaseAmount", y="CustomerSatisfaction",
        color="CustomerSegment", template="plotly_white",
        title="Purchase Amount vs. Customer Satisfaction",
        opacity=0.75,
    )
    fig_scatter.update_layout(margin=dict(t=40, b=20), height=350)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# TOP 5 CUSTOMERS TABLE
# ----------------------------------------------------------------------------
st.subheader("🏆 Top 5 Customers by Total Revenue")
top_customers = (
    fdf.groupby("CustomerID")
    .agg(
        TotalRevenue=("PurchaseAmount", "sum"),
        Transactions=("PurchaseAmount", "count"),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
        PrimarySegment=("CustomerSegment", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
        PrimaryRegion=("CustomerRegion", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    )
    .reset_index()
    .sort_values("TotalRevenue", ascending=False)
    .head(5)
)
top_customers["TotalRevenue"] = top_customers["TotalRevenue"].round(2)
top_customers["AvgSatisfaction"] = top_customers["AvgSatisfaction"].round(2)

st.dataframe(
    top_customers.rename(columns={
        "CustomerID": "Customer ID",
        "TotalRevenue": "Total Revenue ($)",
        "Transactions": "# Transactions",
        "AvgSatisfaction": "Avg. Satisfaction",
        "PrimarySegment": "Primary Segment",
        "PrimaryRegion": "Primary Region",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ----------------------------------------------------------------------------
# AUTOMATED BUSINESS INSIGHTS
# ----------------------------------------------------------------------------
st.subheader("💡 Business Insights")
st.caption("Automatically generated from the currently filtered data.")

insights_col1, insights_col2 = st.columns(2)

# --- Highest-performing customer segment -----------------------------------
best_segment = seg_rev.iloc[0]["CustomerSegment"] if not seg_rev.empty else "N/A"
best_segment_rev = seg_rev.iloc[0]["Revenue"] if not seg_rev.empty else 0

# --- Strongest product category ---------------------------------------------
best_category = cat_rev.iloc[0]["ProductCategory"] if not cat_rev.empty else "N/A"
best_category_rev = cat_rev.iloc[0]["Revenue"] if not cat_rev.empty else 0

# --- Region with growth opportunity (lowest revenue among active regions) ---
growth_region_row = reg_rev.sort_values("Revenue", ascending=True).iloc[0] if not reg_rev.empty else None

# --- Declining customer warning signs ---------------------------------------
decline_df = fdf[fdf["CustomerSegment"].str.lower() == "decline"]
low_satisfaction_df = fdf[fdf["CustomerSatisfaction"] <= 2]

with insights_col1:
    st.markdown(
        f"""<div class="insight-card positive">
        <b>🏅 Top Segment:</b> The <b>{best_segment}</b> segment leads in revenue,
        generating <b>${best_segment_rev:,.0f}</b> in the current filtered view.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="insight-card positive">
        <b>🛍️ Strongest Category:</b> <b>{best_category}</b> is the top-performing
        product category, contributing <b>${best_category_rev:,.0f}</b> in revenue.
        </div>""",
        unsafe_allow_html=True,
    )
    if growth_region_row is not None:
        st.markdown(
            f"""<div class="insight-card">
            <b>🌍 Growth Opportunity:</b> The <b>{growth_region_row['CustomerRegion']}</b>
            region currently generates the least revenue
            (${growth_region_row['Revenue']:,.0f}), representing an opportunity for
            targeted regional marketing investment.
            </div>""",
            unsafe_allow_html=True,
        )

with insights_col2:
    if not decline_df.empty:
        decline_share = len(decline_df) / len(fdf) * 100
        st.markdown(
            f"""<div class="insight-card warning">
            <b>⚠️ Decline Warning:</b> Customers labeled <b>Decline</b> make up
            <b>{decline_share:.1f}%</b> of filtered transactions, with an average
            satisfaction of <b>{decline_df['CustomerSatisfaction'].mean():.2f}/5</b>.
            These accounts are at elevated churn risk.
            </div>""",
            unsafe_allow_html=True,
        )
    if not low_satisfaction_df.empty:
        st.markdown(
            f"""<div class="insight-card warning">
            <b>⚠️ Low Satisfaction Alert:</b> <b>{len(low_satisfaction_df)}</b>
            transactions have a satisfaction score of 2 or below, spanning
            <b>{low_satisfaction_df['CustomerID'].nunique()}</b> unique customers —
            a signal worth proactive outreach.
            </div>""",
            unsafe_allow_html=True,
        )
    if decline_df.empty and low_satisfaction_df.empty:
        st.markdown(
            """<div class="insight-card positive">
            <b>✅ Healthy Base:</b> No significant decline or low-satisfaction
            signals detected in the current filtered view.
            </div>""",
            unsafe_allow_html=True,
        )

# --- Recommendations ---------------------------------------------------------
st.markdown("#### 🎯 Recommended Actions")
st.markdown(
    f"""
    <div class="insight-card action">
    <b>1. Double down on {best_segment}:</b> Expand loyalty offers and
    personalized campaigns for the {best_segment} segment to reinforce the
    revenue it already drives.
    </div>
    <div class="insight-card action">
    <b>2. Launch a win-back campaign for at-risk customers:</b> Target
    customers in the Decline segment and those with satisfaction scores of
    2 or below with tailored retention offers, discounts, or proactive
    customer service outreach.
    </div>
    <div class="insight-card action">
    <b>3. Invest in regional and category expansion:</b> Increase marketing
    spend in {growth_region_row['CustomerRegion'] if growth_region_row is not None else 'underperforming regions'}
    and promote {best_category} alongside complementary categories to grow
    basket size across all channels.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
st.caption("NovaRetail Customer Intelligence Dashboard · Built with Streamlit & Plotly")
