"""
feature_engineering.py
------------------------
1. Reconstructs Logon/Logoff pairs into sessions.
2. Builds a per-user behavioral baseline (mean/std of login hour,
   session length, set of known devices) from the FIRST HALF of
   each user's timeline (the "training"/baseline period).
3. Computes deviation-from-baseline features for EVERY session
   (including the baseline period itself, for training the model,
   and the later period, for testing).

Output: data/session_features.csv
"""

import pandas as pd
import numpy as np

df = pd.read_csv("../data/logon.csv", parse_dates=["date"])
df = df.sort_values(["user", "date"])

# ---- 1. Reconstruct sessions (pair consecutive Logon -> Logoff per user) ----
sessions = []
for user, grp in df.groupby("user"):
    grp = grp.sort_values("date").reset_index(drop=True)
    open_logon = None
    for _, row in grp.iterrows():
        if row["activity"] == "Logon":
            open_logon = row
        elif row["activity"] == "Logoff" and open_logon is not None:
            duration_hrs = (row["date"] - open_logon["date"]).total_seconds() / 3600
            if 0 < duration_hrs < 24:  # sanity bound
                sessions.append({
                    "user": user,
                    "pc": open_logon["pc"],
                    "login_time": open_logon["date"],
                    "logoff_time": row["date"],
                    "login_hour": open_logon["date"].hour + open_logon["date"].minute / 60,
                    "day_of_week": open_logon["date"].dayofweek,
                    "session_duration_hrs": duration_hrs,
                    "date_only": open_logon["date"].date(),
                })
            open_logon = None

sess_df = pd.DataFrame(sessions).sort_values(["user", "login_time"]).reset_index(drop=True)

# ---- 2. Per-user baseline from first half of their session history ----
feature_rows = []

for user, grp in sess_df.groupby("user"):
    grp = grp.sort_values("login_time").reset_index(drop=True)
    n = len(grp)
    if n < 8:
        continue  # not enough history to build a meaningful baseline
    split = max(5, n // 2)
    baseline = grp.iloc[:split]

    baseline_mean_hour = baseline["login_hour"].mean()
    baseline_std_hour = max(baseline["login_hour"].std(), 0.25)
    baseline_mean_dur = baseline["session_duration_hrs"].mean()
    baseline_std_dur = max(baseline["session_duration_hrs"].std(), 0.25)
    known_pcs = set(baseline["pc"].unique())

    # sessions-per-day frequency baseline (rolling count within baseline period)
    baseline_daily_counts = baseline.groupby("date_only").size()
    baseline_mean_freq = baseline_daily_counts.mean()
    baseline_std_freq = max(baseline_daily_counts.std(), 0.5) if len(baseline_daily_counts) > 1 else 0.5

    # compute features for EVERY session of this user (baseline period + later)
    daily_counts_all = grp.groupby("date_only").size()

    for idx, row in grp.iterrows():
        day_count = daily_counts_all.get(row["date_only"], 1)
        feature_rows.append({
            "user": row["user"],
            "date": row["login_time"],
            "date_only": row["date_only"],
            "pc": row["pc"],
            "login_hour": row["login_hour"],
            "session_duration_hrs": row["session_duration_hrs"],
            "day_of_week": row["day_of_week"],
            "is_weekend": 1 if row["day_of_week"] >= 5 else 0,
            # --- deviation-from-own-baseline features (the core insight) ---
            "login_hour_zscore": (row["login_hour"] - baseline_mean_hour) / baseline_std_hour,
            "duration_zscore": (row["session_duration_hrs"] - baseline_mean_dur) / baseline_std_dur,
            "is_new_device": 0 if row["pc"] in known_pcs else 1,
            "sessions_today": day_count,
            "sessions_today_zscore": (day_count - baseline_mean_freq) / baseline_std_freq,
            "is_off_hours": 1 if (row["login_hour"] < 6 or row["login_hour"] > 21) else 0,
            "is_baseline_period": 1 if idx < split else 0,
        })

feat_df = pd.DataFrame(feature_rows)

# ---- 3. Attach ground-truth labels for evaluation ----
gt = pd.read_csv("../data/malicious_ground_truth.csv")
gt["date"] = pd.to_datetime(gt["date"]).dt.date
gt_set = set(zip(gt["user"], gt["date"]))

feat_df["is_malicious_gt"] = feat_df.apply(
    lambda r: 1 if (r["user"], r["date_only"]) in gt_set else 0, axis=1
)

feat_df.to_csv("../data/session_features.csv", index=False)

print(f"Built {len(feat_df)} session feature rows for {feat_df['user'].nunique()} users.")
print(f"Malicious (ground truth) sessions in feature set: {feat_df['is_malicious_gt'].sum()}")
print(feat_df[["user","login_hour_zscore","duration_zscore","is_new_device",
                "sessions_today_zscore","is_off_hours","is_malicious_gt"]].head(8))
