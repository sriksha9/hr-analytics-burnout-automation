import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Global Digital Empathy Engine",
    layout="wide"
)

# --------------------------------------------------
# SAFE LOADERS (defensive + deployment friendly)
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("burnout_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load burnout_model.pkl: {e}")
        st.write("Files found in root directory:", os.listdir("."))
        st.stop()

@st.cache_data
def load_data():
    try:
        return pd.read_csv("weekly_activity.csv")
    except Exception as e:
        st.error(f"❌ Failed to load weekly_activity.csv: {e}")
        st.write("Files found in root directory:", os.listdir("."))
        st.stop()

# --------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------
df = load_data()
model = load_model()

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("Global Digital Empathy Engine")
st.caption(
    "Burnout Risk Detection • Cultural Normalization • ONA Signals • Automated Nudges (JITAI)"
)

# --------------------------------------------------
# FEATURE DEFINITIONS (MUST MATCH TRAINING)
# --------------------------------------------------
NUM_FEATURES = [
    "total_emails_sent",
    "total_emails_received",
    "avg_email_reply_time_min",
    "total_slack_msgs_sent",
    "after_hours_msgs_count",
    "num_meetings",
    "total_meeting_hours",
    "back_to_back_meeting_blocks",
    "unique_contacts_count",
    "degree_centrality",
    "betweenness_centrality",
    "isolation_score",
    "z_after_hours_within_country",
    "z_reply_time_within_country",
    "z_meeting_load_within_country"
]

CAT_FEATURES = ["role", "team", "country", "culture_cluster"]

REQUIRED_COLS = NUM_FEATURES + CAT_FEATURES + ["employee_id", "week_start_date"]

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"❌ weekly_activity.csv is missing required columns: {missing}")
    st.stop()

# --------------------------------------------------
# MODEL INFERENCE
# --------------------------------------------------
X = df[NUM_FEATURES + CAT_FEATURES]

proba = model.predict_proba(X)
pred = model.predict(X)

df = df.copy()
df["predicted_label"] = pred
df["prob_low"] = proba[:, 0]
df["prob_medium"] = proba[:, 1]
df["prob_high"] = proba[:, 2]

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Filters")

weeks = sorted(df["week_start_date"].unique())
selected_week = st.sidebar.selectbox("Week", weeks, index=len(weeks)-1)

countries = sorted(df["country"].unique())
selected_countries = st.sidebar.multiselect(
    "Country", countries, default=countries
)

teams = sorted(df["team"].unique())
selected_teams = st.sidebar.multiselect(
    "Team", teams, default=teams
)

selected_risk = st.sidebar.multiselect(
    "Predicted Burnout Label", [0, 1, 2], default=[0, 1, 2]
)

fdf = df[
    (df["week_start_date"] == selected_week) &
    (df["country"].isin(selected_countries)) &
    (df["team"].isin(selected_teams)) &
    (df["predicted_label"].isin(selected_risk))
]

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Employees", len(fdf))
col2.metric("High Risk (Label 2)", int((fdf["predicted_label"] == 2).sum()))
col3.metric(
    "Avg After-hours Msgs",
    round(fdf["after_hours_msgs_count"].mean(), 2) if len(fdf) else 0
)
col4.metric(
    "Avg Meeting Hours",
    round(fdf["total_meeting_hours"].mean(), 2) if len(fdf) else 0
)

st.divider()

# --------------------------------------------------
# RISK TABLE
# --------------------------------------------------
st.subheader("Burnout Risk Table (sorted by High-Risk Probability)")

risk_table = fdf[
    [
        "employee_id", "role", "team", "country",
        "predicted_label", "prob_high",
        "after_hours_msgs_count", "total_meeting_hours",
        "isolation_score"
    ]
].sort_values("prob_high", ascending=False)

st.dataframe(risk_table, use_container_width=True)

# Download high-risk employees
high_risk = risk_table[risk_table["predicted_label"] == 2]
st.download_button(
    "⬇️ Download High-Risk Employees (CSV)",
    data=high_risk.to_csv(index=False).encode("utf-8"),
    file_name=f"high_risk_{selected_week}.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# EMPLOYEE DETAIL + JITAI NUDGES
# --------------------------------------------------
st.subheader("Employee Detail & Automated Nudges")

if len(fdf) == 0:
    st.warning("No employees match the selected filters.")
    st.stop()

emp_id = st.selectbox(
    "Select Employee",
    sorted(fdf["employee_id"].unique())
)

emp_hist = df[df["employee_id"] == emp_id].sort_values("week_start_date")
row = emp_hist[emp_hist["week_start_date"] == selected_week].iloc[0]

st.line_chart(emp_hist.set_index("week_start_date")["prob_high"])

st.markdown("### Weekly Snapshot")
st.json({
    "employee_id": row["employee_id"],
    "role": row["role"],
    "team": row["team"],
    "country": row["country"],
    "burnout_label": int(row["predicted_label"]),
    "high_risk_probability": float(row["prob_high"]),
    "after_hours_msgs": int(row["after_hours_msgs_count"]),
    "meeting_hours": float(row["total_meeting_hours"]),
    "isolation_score": float(row["isolation_score"])
})

# --------------------------------------------------
# JITAI NUDGES
# --------------------------------------------------
st.markdown("### Recommended Nudge(s)")

nudges = []

if row["after_hours_msgs_count"] > 20 and row["prob_high"] > 0.35:
    nudges.append(
        "📌 **Boundary Nudge:** Frequent after-hours work detected. "
        "Protect one evening this week by scheduling non-urgent work into regular hours."
    )

if row["total_meeting_hours"] > 20 or row["back_to_back_meeting_blocks"] >= 3:
    nudges.append(
        "📌 **Meeting Recovery Nudge:** Calendar overload detected. "
        "Convert one 60-minute meeting into 45 minutes to create recovery space."
    )

if row["isolation_score"] > 0.75:
    nudges.append(
        "📌 **Connection Nudge:** Collaboration appears narrow. "
        "Consider pairing with a teammate or joining a cross-functional sync."
    )

if not nudges:
    nudges.append(
        "✅ **No intervention required:** Maintain healthy routines and balanced workload."
    )

for n in nudges:
    st.markdown(f"- {n}")
