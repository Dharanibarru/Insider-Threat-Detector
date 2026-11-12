"""
generate_data.py
-----------------
Generates a synthetic dataset that mirrors the structure of the CMU CERT
Insider Threat Dataset's logon.csv (columns: id, date, user, pc, activity).

Simulates:
  - ~120 "normal" employees with consistent, role-based login rhythms
    (fixed office hours, 1-2 regular PCs, occasional normal variation)
  - 6 "malicious insider" users who, starting from a randomly chosen
    'turn' date, begin exhibiting anomalous behavior:
      * off-hours logins (late night / early morning)
      * logins from unusual/new PCs
      * abnormally long sessions (mass data access) or very short
        repeated sessions (probing)
      * spikes in login frequency

This gives us a labeled dataset so we can properly evaluate the
anomaly detection model (Precision/Recall/F1/ROC-AUC), same as you
would with the real CERT r4.2/r6.2 release.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

N_NORMAL_USERS = 120
N_MALICIOUS_USERS = 6
SIM_DAYS = 90
START_DATE = datetime(2025, 1, 1)

roles = ["engineering", "finance", "hr", "sales", "it_admin", "legal"]

def make_user_pool(n, prefix="U"):
    users = []
    for i in range(n):
        uid = f"{prefix}{i:04d}"
        role = random.choice(roles)
        # each user has a home PC and sometimes a secondary one
        home_pc = f"PC-{random.randint(1000,1999)}"
        secondary_pc = f"PC-{random.randint(1000,1999)}"
        # baseline behavioral profile
        mean_login_hour = np.random.normal(9, 1.0)   # most people arrive ~9am
        mean_session_len = np.random.normal(6.5, 1.5)  # hours
        users.append({
            "user": uid, "role": role, "home_pc": home_pc,
            "secondary_pc": secondary_pc,
            "mean_login_hour": np.clip(mean_login_hour, 6, 11),
            "mean_session_len": np.clip(mean_session_len, 2, 10),
        })
    return users

normal_users = make_user_pool(N_NORMAL_USERS, "EMP")
malicious_users = make_user_pool(N_MALICIOUS_USERS, "MAL")

records = []
rec_id = 0

def add_session(uid, pc, day, login_hour, login_minute, duration_hours):
    global rec_id
    login_dt = day.replace(hour=int(login_hour) % 24, minute=login_minute, second=0)
    logoff_dt = login_dt + timedelta(hours=duration_hours)
    records.append({"id": rec_id, "date": login_dt, "user": uid, "pc": pc, "activity": "Logon"})
    rec_id += 1
    records.append({"id": rec_id, "date": logoff_dt, "user": uid, "pc": pc, "activity": "Logoff"})
    rec_id += 1

# ---- Normal users: consistent weekday rhythm with small natural noise ----
for u in normal_users:
    for day_offset in range(SIM_DAYS):
        day = START_DATE + timedelta(days=day_offset)
        if day.weekday() >= 5:          # skip weekends (mostly)
            if random.random() > 0.03:  # rare weekend work
                continue
        if random.random() > 0.94:      # occasional day off
            continue
        login_hour = np.random.normal(u["mean_login_hour"], 0.6)
        login_minute = random.randint(0, 59)
        duration = max(0.5, np.random.normal(u["mean_session_len"], 0.8))
        pc = u["home_pc"] if random.random() > 0.08 else u["secondary_pc"]
        add_session(u["user"], pc, day, login_hour, login_minute, duration)
        # small chance of a second short session same day
        if random.random() < 0.05:
            add_session(u["user"], u["home_pc"], day,
                        login_hour + duration + 0.5, random.randint(0,59), 0.5)

# ---- Malicious insiders: normal at first, then anomalous after 'turn' day ----
insider_labels = []  # (user, date) pairs flagged as ground-truth malicious sessions

for u in malicious_users:
    turn_day = random.randint(40, 65)   # behavior turns anomalous after this day
    anomaly_type = random.choice(["off_hours", "new_device", "long_session", "burst_frequency"])
    for day_offset in range(SIM_DAYS):
        day = START_DATE + timedelta(days=day_offset)
        is_turned = day_offset >= turn_day

        if not is_turned:
            # behaves normally pre-turn
            if day.weekday() >= 5:
                continue
            if random.random() > 0.94:
                continue
            login_hour = np.random.normal(u["mean_login_hour"], 0.6)
            duration = max(0.5, np.random.normal(u["mean_session_len"], 0.8))
            add_session(u["user"], u["home_pc"], day, login_hour, random.randint(0,59), duration)
        else:
            # anomalous phase
            if anomaly_type == "off_hours":
                login_hour = random.choice([1, 2, 3, 23, 0])
                duration = np.random.normal(2, 0.5)
                pc = u["home_pc"]
            elif anomaly_type == "new_device":
                login_hour = np.random.normal(u["mean_login_hour"], 1)
                duration = np.random.normal(u["mean_session_len"], 1)
                pc = f"PC-{random.randint(2000,2999)}"  # unseen device
            elif anomaly_type == "long_session":
                login_hour = np.random.normal(u["mean_login_hour"], 1)
                duration = np.random.normal(11, 1.5)     # unusually long, mass access
                pc = u["home_pc"]
            else:  # burst_frequency
                login_hour = np.random.normal(u["mean_login_hour"], 2)
                duration = max(0.2, np.random.normal(1, 0.3))
                pc = u["home_pc"]
                # multiple bursts same day
                for _ in range(random.randint(2, 4)):
                    add_session(u["user"], pc, day, login_hour + random.uniform(-1,1),
                               random.randint(0,59), duration)
                    insider_labels.append((u["user"], day.strftime("%Y-%m-%d")))
                continue

            add_session(u["user"], pc, day, login_hour, random.randint(0,59), max(0.2, duration))
            insider_labels.append((u["user"], day.strftime("%Y-%m-%d")))

df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

df.to_csv("../data/logon.csv", index=False)

labels_df = pd.DataFrame(insider_labels, columns=["user", "date"]).drop_duplicates()
labels_df.to_csv("../data/malicious_ground_truth.csv", index=False)

print(f"Generated {len(df)} logon/logoff records for {N_NORMAL_USERS + N_MALICIOUS_USERS} users.")
print(f"Ground-truth malicious sessions: {len(labels_df)} (user-day pairs)")
print(f"Malicious users: {[u['user'] for u in malicious_users]}")
print(df.head())
