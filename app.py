import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="AI HR Burnout Automation", layout="wide")

# ---------------------------
# Cached loaders (fast + stable for deployment)
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("burnout_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("weekly_activity.csv")

# ---------------------------
# App Title
# ---------------------------
st.title("Global Digital Empathy Engine")
st.caption("Burnout Risk Detection • Cultural Normalization • ONA Signals • Automated Nudges (JITAI)")

# ---------------------------
# Load resources
# ---------------------------
try:
    df = load_data()
except Exception as e:
    st.error("❌ Could not load weekly_activity.csv. Make sure the file is in the GitHub repo root.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load burnout_model.pkl. Make sure the file is in the GitHub repo root.")
    st.stop()

# ---------------------------
# Feature columns (must match training)
# ---------------------------
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

# Safety check (prevents silent failures)
missing_cols = [c for c in (NUM_FEATURES + CAT_FEATURES + ["employee_id", "week_start_date"]) if c not in df.columns]
if missing_cols:
    st.error(f"❌ weekly_activity.csv is missing required columns: {missing_cols}")
    st.stop()

X = df[NUM_FEATURES + CAT_FEATURES].copy()

# ---------------------------
# Predictions
# ---------------------------
proba = model.predict_proba(X)
pred = model.predict(X)

df = df.copy()
df["predicted_label"] = pred
df["prob_low"] = proba[:, 0]
df["prob_medium"] = proba[:, 1]
df["prob_high"] = proba[:, 2]

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")

week_options = sorted(df["week_start_date"].unique())
selected_week = st.sidebar.selectbox("Week", week_options, index=len(week_options)-1)

country_options = sorted(df["country"].unique())
selected_countries = st.sidebar.multiselect("Country", country_options, default=country_options)

team_options = sorted(df["team"].unique())
selected_teams = st.sidebar.multiselect("Team", team_options, default=team_options)

selected_risk = st.sidebar.multiselect("Predicted Risk Label", [0, 1, 2], default=[0, 1, 2])

# Filtered view (week-specific)
fdf = df[
    (df["week_start_date"] == selected_week) &
    (df["country"].isin(selected_countries)) &
    (df["team"].isin(selected_teams)) &
    (df["predicted_label"].isin(selected_risk))
].copy()

# ---------------------------
# KPI Row
# ---------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees (Filtered)", int(len(fdf)))
col2.metric("High Risk (Label 2)", int((fdf["predicted_label"] == 2).sum()))
col3.metric("Avg After-hours Msgs", float(round(fdf["after_hours_msgs_count"].mean(), 2)) if len(fdf) else 0.0)
col4.metric("Avg Meeting Hours", float(round(fdf["total_meeting_hours"].mean(), 2)) if len(fdf) else 0.0)

st.divider()

# ---------------------------
# Risk Table
# ---------------------------
st.subheader("Risk Table (sorted by High-Risk Probability)")

table_cols = [
    "employee_id", "role", "team", "country", "week_start_date",
    "predicted_label", "prob_high",
    "after_hours_msgs_count", "total_meeting_hours",
    "back_to_back_meeting_blocks", "isolation_score"
]

risk_table = fdf[table_cols].sort_values("prob_high", ascending=False)
st.dataframe(risk_table, use_container_width=True)

# Download High Risk list
high_risk = risk_table[risk_table["predicted_label"] == 2].copy()
st.download_button(
    "⬇️ Download High-Risk List (CSV)",
    data=high_risk.to_csv(index=False).encode("utf-8"),
    file_name=f"high_risk_{selected_week}.csv",
    mime="text/csv"
)

st.divider()

# ---------------------------
# Employee Detail + Nudges
# ---------------------------
st.subheader("Employee Detail & Automated Nudges (JITAI)")

if len(fdf) == 0:
    st.warning("No employees match your filters for this week.")
    st.stop()

employee_options = sorted(fdf["employee_id"].unique())
selected_emp = st.selectbox("Select Employee", employee_options)

emp_history = df[df["employee_id"] == selected_emp].sort_values("week_start_date").copy()
selected_row = emp_history[emp_history["week_start_date"] == selected_week]

# Trend chart
st.write("High-Risk Probability Trend (by week)")
st.line_chart(emp_history.set_index("week_start_date")["prob_high"])

if selected_row.empty:
    st.warning("No record found for this employee in the selected week.")
    st.stop()

row = selected_row.iloc[0]

# Snapshot
st.markdown("### Selected Week Snapshot")
st.json({
    "employee_id": row["employee_id"],
    "role": row["role"],
    "team": row["team"],
    "country": row["country"],
    "predicted_label": int(row["predicted_label"]),
    "prob_high": float(row["prob_high"]),
    "after_hours_msgs_count": int(row["after_hours_msgs_count"]),
    "total_meeting_hours": float(row["total_meeting_hours"]),
    "back_to_back_meeting_blocks": int(row["back_to_back_meeting_blocks"]),
    "isolation_score": float(row["isolation_score"])
})

# ---------------------------
# JITAI Nudge Rules (simple + effective)
# ---------------------------
st.markdown("### Recommended Nudge(s)")

nudges = []

# Boundary nudge
if row["after_hours_msgs_count"] > 20 and row["prob_high"] > 0.35:
    nudges.append("📌 **Boundary Nudge:** Frequent after-hours activity detected. Protect one evening this week by scheduling non-urgent work into your normal hours.")

# Meeting recovery nudge
if row["back_to_back_meeting_blocks"] >= 3 or row["total_meeting_hours"] > 20:
    nudges.append(" **Meeting Recovery Nudge:** Calendar load is high. Convert one 60-min meeting into 45 mins to create recovery time between meetings.")

# Social connection nudge
if row["isolation_score"] > 0.75 and row["prob_high"] > 0.30:
    nudges.append(" **Connection Nudge:** Collaboration appears narrow. Consider pairing with a teammate or joining a cross-functional sync to stay connected.")

# If nothing triggered
if not nudges:
    nudges.append(" **No strong intervention needed:** Maintain breaks, healthy boundaries, and steady collaboration routines.")

for n in nudges:
    st.markdown(f"- {n}")

st.divider()

# ---------------------------
# Optional: show raw record for transparency
# ---------------------------
with st.expander("Show raw feature values (for transparency)"):
    st.dataframe(selected_row[NUM_FEATURES + CAT_FEATURES + ["predicted_label", "prob_low", "prob_medium", "prob_high"]],
                 use_container_width=True)
