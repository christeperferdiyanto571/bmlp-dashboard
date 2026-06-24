import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Dashboard · Chris Teper Ferdiyanto",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.3);
}
[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.85rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #fff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
}

/* Plotly charts bg */
.js-plotly-plot {
    border-radius: 16px;
    overflow: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6);
    border-radius: 8px;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* Section titles */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Info box */
.info-box {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 12px;
    padding: 16px 20px;
    color: #c7d2fe;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "data_clustering.csv"))
    df_inv = pd.read_csv(os.path.join(base, "data_clustering_inverse.csv"))
    return df, df_inv

df, df_inv = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Dashboard")
    st.markdown("**Chris Teper Ferdiyanto**")
    st.markdown("---")

    st.markdown("### 🎛️ Filter Data")
    age_group_options = sorted(df["CustomerAge_Group"].unique())
    selected_groups = st.multiselect(
        "Customer Age Group",
        options=age_group_options,
        default=age_group_options
    )

    channel_options = sorted(df["Channel"].unique())
    selected_channels = st.multiselect(
        "Channel",
        options=channel_options,
        default=channel_options
    )

    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.info(f"📊 **{len(df):,}** total records\n\n🔢 **{df.shape[1]}** features")

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df[
    df["CustomerAge_Group"].isin(selected_groups) &
    df["Channel"].isin(selected_channels)
]

# ── Plotly theme helper ───────────────────────────────────────────────────────
DARK_TEMPLATE = "plotly_dark"
GLASS_PAPER = "rgba(255,255,255,0.04)"
GLASS_PLOT  = "rgba(0,0,0,0)"
PURPLE      = "#6366f1"
CYAN        = "#22d3ee"
PINK        = "#f472b6"
AMBER       = "#fbbf24"
GREEN       = "#34d399"

def chart_layout(fig, title=""):
    fig.update_layout(
        paper_bgcolor=GLASS_PAPER,
        plot_bgcolor=GLASS_PLOT,
        font=dict(color="white", family="Inter"),
        title=dict(text=title, font=dict(size=15, color="#a5b4fc")),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        )
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size:2.8rem; font-weight:800; background: linear-gradient(135deg, #6366f1, #22d3ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        🤖 ML Analytics Dashboard
    </h1>
    <p style="color:rgba(255,255,255,0.5); font-size:1rem; margin-top:8px;">
        Financial Transaction Clustering & Analysis · Chris Teper Ferdiyanto
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📦 Total Records", f"{len(filtered):,}")
with col2:
    st.metric("👥 Avg Customer Age", f"{df_inv['CustomerAge'].mean():.1f} yr" if 'CustomerAge' in df_inv.columns else "N/A")
with col3:
    n_clusters = df["CustomerAge_Group"].nunique()
    st.metric("🔵 Clusters", n_clusters)
with col4:
    n_channels = df["Channel"].nunique()
    st.metric("📡 Channels", n_channels)
with col5:
    n_types = df["TransactionType"].nunique()
    st.metric("💳 Tx Types", n_types)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔵 Clustering", "📈 Distribution", "🔍 Data Explorer"])

# ═══════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        # Transaction Amount by Channel
        fig = px.box(
            filtered, x="Channel", y="TransactionAmount",
            color="Channel",
            color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER],
            template=DARK_TEMPLATE
        )
        chart_layout(fig, "📦 Transaction Amount by Channel")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Transaction Type Pie
        type_counts = filtered["TransactionType"].value_counts().reset_index()
        type_counts.columns = ["TransactionType", "Count"]
        fig2 = px.pie(
            type_counts, names="TransactionType", values="Count",
            color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER, GREEN],
            hole=0.5,
            template=DARK_TEMPLATE
        )
        fig2.update_traces(textfont_color="white")
        chart_layout(fig2, "💳 Transaction Type Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Login Attempts vs Account Balance
        fig3 = px.scatter(
            filtered.sample(min(500, len(filtered))),
            x="AccountBalance", y="TransactionAmount",
            color="CustomerAge_Group",
            color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER],
            opacity=0.7,
            template=DARK_TEMPLATE
        )
        chart_layout(fig3, "💰 Account Balance vs Transaction Amount")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Transaction Duration histogram
        fig4 = px.histogram(
            filtered, x="TransactionDuration",
            nbins=40,
            color_discrete_sequence=[CYAN],
            template=DARK_TEMPLATE
        )
        fig4.update_traces(marker_line_color="rgba(255,255,255,0.1)", marker_line_width=1)
        chart_layout(fig4, "⏱️ Transaction Duration Distribution")
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — Clustering
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="info-box">🔵 Visualisasi hasil <b>K-Means Clustering</b> berdasarkan fitur transaksi nasabah.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Cluster count bar
        cluster_counts = filtered["CustomerAge_Group"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        cluster_counts["Cluster"] = cluster_counts["Cluster"].apply(lambda x: f"Cluster {x}")
        fig5 = px.bar(
            cluster_counts, x="Cluster", y="Count",
            color="Cluster",
            color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER],
            template=DARK_TEMPLATE,
            text="Count"
        )
        fig5.update_traces(textposition="outside", textfont_color="white")
        chart_layout(fig5, "👥 Jumlah Anggota per Cluster")
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        # Radar chart per cluster
        numeric_cols = ["TransactionAmount", "TransactionDuration", "LoginAttempts", "AccountBalance"]
        cluster_means = filtered.groupby("CustomerAge_Group")[numeric_cols].mean().reset_index()

        fig6 = go.Figure()
        fill_colors = [
            "rgba(99, 102, 241, 0.2)",
            "rgba(34, 211, 238, 0.2)",
            "rgba(244, 114, 182, 0.2)",
            "rgba(251, 191, 36, 0.2)"
        ]
        colors = [PURPLE, CYAN, PINK, AMBER]
        for i, row in cluster_means.iterrows():
            values = [row[c] for c in numeric_cols] + [row[numeric_cols[0]]]
            cats = numeric_cols + [numeric_cols[0]]
            fig6.add_trace(go.Scatterpolar(
                r=values, theta=cats,
                fill='toself',
                name=f"Cluster {int(row['CustomerAge_Group'])}",
                line_color=colors[i % len(colors)],
                fillcolor=fill_colors[i % len(fill_colors)]
            ))
        fig6.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.03)",
                radialaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="rgba(255,255,255,0.5)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="rgba(255,255,255,0.7)")
            ),
            paper_bgcolor=GLASS_PAPER,
            font=dict(color="white", family="Inter"),
            title=dict(text="🕸️ Cluster Profile Radar", font=dict(size=15, color="#a5b4fc")),
            legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig6, use_container_width=True)

    # Heatmap cluster vs channel
    heatmap_data = filtered.groupby(["CustomerAge_Group", "Channel"]).size().reset_index(name="Count")
    pivot = heatmap_data.pivot(index="CustomerAge_Group", columns="Channel", values="Count").fillna(0)
    fig7 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"Ch {c}" for c in pivot.columns],
        y=[f"Cluster {r}" for r in pivot.index],
        colorscale=[[0, "#1e1b4b"], [0.5, "#6366f1"], [1, "#22d3ee"]],
        showscale=True,
        text=pivot.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(color="white", size=14)
    ))
    chart_layout(fig7, "🗺️ Heatmap: Cluster vs Channel")
    st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 3 — Distribution
# ═══════════════════════════════════════════════════════
with tab3:
    numeric_features = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]
    available = [c for c in numeric_features if c in df_inv.columns]

    if available:
        c1, c2 = st.columns(2)
        for i, col_name in enumerate(available):
            fig = px.histogram(
                df_inv, x=col_name,
                nbins=50,
                marginal="box",
                color_discrete_sequence=[PURPLE if i % 2 == 0 else CYAN],
                template=DARK_TEMPLATE
            )
            chart_layout(fig, f"📊 {col_name}")
            if i % 2 == 0:
                with c1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with c2:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to scaled data
        cols = ["TransactionAmount", "TransactionDuration", "AccountBalance"]
        c1, c2, c3 = st.columns(3)
        for i, (col_name, col_ref) in enumerate(zip(cols, [c1, c2, c3])):
            if col_name in filtered.columns:
                fig = px.violin(
                    filtered, y=col_name,
                    color="CustomerAge_Group",
                    color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER],
                    box=True, template=DARK_TEMPLATE
                )
                chart_layout(fig, f"🎻 {col_name}")
                with col_ref:
                    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="info-box">🔍 Eksplorasi data mentah. Gunakan filter di sidebar untuk menyaring data.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        search = st.text_input("🔎 Search / Filter (masukkan nilai kolom)", "")
    with col_b:
        n_rows = st.selectbox("Tampilkan baris", [20, 50, 100, 500], index=0)

    display_df = df_inv if len(df_inv.columns) > len(df.columns) else filtered
    if search:
        mask = display_df.astype(str).apply(lambda col: col.str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(
        display_df.head(n_rows).style.background_gradient(cmap="Blues"),
        use_container_width=True,
        height=400
    )

    st.markdown(f"**Showing {min(n_rows, len(display_df))} of {len(display_df):,} records**")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            data=filtered.to_csv(index=False),
            file_name="filtered_data.csv",
            mime="text/csv"
        )
    with col_dl2:
        st.download_button(
            "⬇️ Download Original Data (CSV)",
            data=df_inv.to_csv(index=False),
            file_name="original_data.csv",
            mime="text/csv"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.85rem; padding: 1rem 0;">
    🤖 ML Analytics Dashboard · Built with Streamlit & Plotly · Chris Teper Ferdiyanto
</div>
""", unsafe_allow_html=True)
